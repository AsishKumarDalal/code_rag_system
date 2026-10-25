#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3 · STEP 2 — CPT TRAINING  (docs/phase3.md · Part I)

    cpt_manifest.json (data/cpt/*_block_*.bin on HF · ~286M tokens · 4096/block)
        │
        ├─ ① LIB CHECK  python/torch/transformers/peft/accelerate/hub/numpy
        │               auto-install anything missing · torch_xla on TPU
        ├─ ② SETUP      Qwen3-4B base (bf16) + LoRA r=128 on q/k/v/o/gate/up/down
        │               (memory-honest: full 4B AdamW needs ~56GB optim state;
        │                LoRA@r128 ≈ 6.5% trainable → fits 16GB cores & T4s)
        ├─ ③ TRAIN      WSD: warmup 1% → stable @ 1e-4 → decay 20% → 5%
        │               grad-accum to 64 seqs/step = 262k tokens/step
        │               chunked cross-entropy (151k vocab: full loss OOMs)
        ├─ ④ GUARD      probe every 500 steps on 32 held-out tail blocks
        │               checkpoint (adapter+optim+position) every ~55min → HF
        ├─ ⑤ RESUME     new session finds newest ckpt on HF → loads adapter,
        │               optimizer and fast-forwards the data stream to the
        │               exact row · frozen shard ledger keeps step→data stable
        └─ ⑥ SHIP       soup decay snapshots+final → probe → merge (on CPU) →
                        models/cpt.v1/ on HF (+README lineage)

Backends (auto, --backend tpu|gpu to force):
    TPU         one process per core via xmp.spawn = true data parallel
                (each core gets different rows) · bf16 · LoRA r=128 · 1 seq/core
    CUDA/T4     QLoRA nf4 fallback · r=64 · fp16 adapters
                (1 epoch default — T4 is pilots/debug only; TPU is the path)

Run (Kaggle/Colab cell; on TPU a notebook cell auto-relaunches as a real
python process because xmp.spawn cannot pickle functions from %run):
    %run data/cpt_train.py --libs             # preflight only: versions
    %run data/cpt_train.py --pilot 60         # smoke: 60 steps + save/reload test
    %run data/cpt_train.py                    # the real run (TPU, ~9-12h)

First TPU step is slow — XLA compiles once per core (1-4 min, normal).
================================================================================
"""
import os, sys, io, gc, json, time, shutil, bisect, argparse
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

def _looks_like_tpu_env():
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU": return True
    if any(os.environ.get(k) for k in (
            "TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID",
            "TPU_WORKER_HOSTNAMES", "TPU_ACCELERATOR_TYPE",
            "TPU_CHIPS_PER_HOST_BOUNDS")):
        return True
    try:
        return any(Path("/dev").glob("accel*"))
    except Exception:
        return False

if _looks_like_tpu_env():
    os.environ.setdefault("PJRT_DEVICE", "TPU")

# HF cache on the ephemeral big disk (model + shard cache ≈ 12GB).
try:
    _H = "/tmp/code_rag_train/hf"; Path(_H).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", _H)
except Exception:
    pass


# ════════════════ ① LIB PREFLIGHT — check, install, report ════════════════════
def _ver_tuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        num = "".join(c for c in part if c.isdigit())
        out.append(int(num) if num else 0)
    while len(out) < 3: out.append(0)
    return tuple(out)

def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None

def ensure_libs(specs, quiet=False):
    """install anything missing/too old; returns [(name, version|None)]."""
    bad = []
    for spec in specs:
        name = spec.split(">=")[0].strip()
        floor = spec.split(">=")[1].strip() if ">=" in spec else None
        cur = _pkg_version(name)
        ok = cur is not None and (floor is None or _ver_tuple(cur) >= _ver_tuple(floor))
        if not ok: bad.append((spec, cur))
    if bad:
        if not quiet:
            print("(auto-installing/upgrading: " +
                  ", ".join(f"{s} (have {c})" for s, c in bad) + ")", flush=True)
        import subprocess
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                             *[s for s, _ in bad]]).returncode
        if rc != 0:
            print(f"\n❌ pip failed. On Kaggle/Colab run once:\n"
                  f"   !pip install -U {' '.join(s for s, _ in bad)}")
            raise SystemExit(1)
    return [(spec.split(">=")[0].strip(), _pkg_version(spec.split(">=")[0].strip()))
            for spec in specs]


CORE_LIBS = ["huggingface_hub>=0.23", "transformers>=4.51.0", "peft>=0.13.0",
             "accelerate", "sentencepiece", "safetensors", "numpy", "tqdm"]

try:
    import torch                                          # noqa: E402
except ImportError:
    print("❌ torch is not installed — attach a GPU/TPU runtime "
          "(Kaggle/Colab) or `pip install torch`")
    raise
try:
    import numpy as np                                    # noqa: E402
except ImportError:
    ensure_libs(["numpy"])
    import numpy as np
ensure_libs(CORE_LIBS)
from huggingface_hub import HfApi, hf_hub_download        # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
try:
    from tqdm.auto import tqdm
except Exception:                                         # tqdm ships with hub
    class _MiniBar:
        def __init__(self, total=None, desc="", **k):
            self.total, self.n = total or 0, 0
        def update(self, n=1): self.n += n
        def set_postfix(self, **k): pass
        def close(self): pass
    tqdm = _MiniBar

def libs_report(extra=()):
    rows = [f"python {sys.version.split()[0]}", f"torch {torch.__version__}"]
    for spec in CORE_LIBS:
        n = spec.split(">=")[0]
        rows.append(f"{n} {_pkg_version(n) or '✗'}")
    for n in extra:
        rows.append(f"{n} {_pkg_version(n) or '✗'}")
    if TPU:
        rows.append(f"torch_xla {_pkg_version('torch_xla') or '✗'}")
        rows.append(f"world {WORLD} · rank {RANK}")
    log("libs: " + " · ".join(rows))
    if not TPU and torch.cuda.is_available():
        log(f"cuda: {torch.version.cuda} · {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GB) "
            f"×{torch.cuda.device_count()}")

# ══════════════════════════════════ CONFIG ════════════════════════════════════
HF_TOKEN    = os.environ.get("HF_TOKEN", "")     # or Kaggle/Colab secret
REPO        = "sddfhy546y/code_rag_dataset"      # dataset repo (blocks live here)
DATA_SUB    = "data/cpt"                         # folder the prep banks to
CKPT_SUB    = "ckpt/cpt"                         # hourly checkpoints
SHIP_SUB    = "models/cpt.v1"                    # final merged weights
BASE        = "Qwen/Qwen3-4B"                    # the one lineage
TARGETS     = ["q_proj", "k_proj", "v_proj", "o_proj",   # explicit + version-proof
               "gate_proj", "up_proj", "down_proj"]
SEED        = 3407
BLOCK       = 4096                               # tokens/block (from prep)
EPOCHS_TPU  = 3
EPOCHS_GPU  = 1
LR_PEAK     = 1.0e-4                             # LoRA peak (docs/phase3.md §WSD)
DECAY_FRAC  = 0.20
DECAY_FLOOR = 0.05
WARM_FRAC   = 0.01
G_BATCH     = 64                                 # seqs / optimizer step (262k tok)
LBS_TPU     = 1              # 1 seq per core macro-batch (4096-tok logits ≈1.2GB)
LBS_GPU     = 2                                  # T4 micro-batch
LOG_EVERY   = 20
PROBE_EVERY = 500
PROBE_ROWS  = 32
HOLDOUT     = 512              # tail blocks never trained on (eval only)
CKPT_MIN    = 55                                 # ≤ minutes between checkpoints
EST_TOK_S   = 27000                              # v5e-8 planning number
TMP_ROOT    = Path(os.environ.get("CPT_TRAIN_TMP", "/tmp/code_rag_train"))
ROOT_CANDIDATES = [
    (Path("/tmp/code_rag_train"),            "ephemeral big disk"),
    (Path("/kaggle/working/code_rag_train"), "kaggle working disk (20GB quota)"),
]

# ════════════════════ plumbing: same idioms as cpt_prep.py ════════════════════
T0 = time.time()
def now():  return datetime.now(timezone.utc).strftime("%H:%M:%S")
def log(msg):
    if IS_MASTER:
        print(f"[{now()}] {msg}", flush=True)
def gfmt(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} TB"
def bar(desc, total=None, unit="it", scale=False):
    try:
        return tqdm(total=total, desc=desc, unit=unit, unit_scale=scale,
                    dynamic_ncols=True, disable=not IS_MASTER)
    except TypeError:
        return tqdm(total=total, desc=desc)
def disk_free(p):
    try: return shutil.disk_usage(str(p)).free
    except Exception: return 0
def free_gb(p): return disk_free(p) / 1024**3
def pick_root():
    global TMP_ROOT
    best, why = None, ""
    for cand, why_ in ROOT_CANDIDATES:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            if best is None or free_gb(cand) > free_gb(best):
                best, why = cand, why_
        except Exception:
            continue
    TMP_ROOT = best or Path(".")
    os.environ["CPT_TRAIN_TMP"] = str(TMP_ROOT)
    log(f"disk: root {TMP_ROOT} (auto: {why}) · free {free_gb(TMP_ROOT):.0f}GB")

# ── HF plumbing (env → Kaggle secret → Colab secret → stored login) ───────────
_API = None
def get_token():
    t = os.environ.get("HF_TOKEN", "")
    if t: return t
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t:
            os.environ["HF_TOKEN"] = t
            log("token: loaded from Kaggle Secrets")
            return t
    except Exception:
        pass
    try:
        from google.colab import userdata
        t = userdata.get("HF_TOKEN")
        if t:
            os.environ["HF_TOKEN"] = t
            log("token: loaded from Colab userdata")
            return t
    except Exception:
        pass
    return ""

def api():
    global _API
    if _API is None:
        tok = HF_TOKEN or get_token()
        if tok:
            try:
                from huggingface_hub import login
                login(token=tok)
            except Exception as e:
                log(f"login warning ({str(e)[:80]})")
        else:
            log("⚠ no HF_TOKEN found — pushes will only work if already logged in")
        _API = HfApi()
    return _API

def repo_tree():
    try:
        items = api().list_repo_tree(REPO, repo_type="dataset", recursive=True)
        return {i.path: getattr(i, "size", None) for i in items}
    except Exception:
        try:
            return {p: None for p in api().list_repo_files(REPO, repo_type="dataset")}
        except Exception as e:
            log(f"⚠ repo tree unavailable ({str(e)[:80]}) — uploads will re-run")
            return {}

class ProgressReader(io.BufferedReader):
    """real file-like + progress bar; the hub rejects hand-rolled wrappers."""
    def __init__(self, path, b):
        self._pb, self._f = b, open(path, "rb")
        super().__init__(self._f, buffer_size=1024 * 1024)
    def read(self, size=-1):
        d = super().read(size)
        if d: self._pb.update(len(d))
        return d
    def close(self):
        try: super().close()
        except Exception: pass
        try: self._f.close()
        except Exception: pass

def bank(local: Path, repo_path: str, tree: dict, desc="⬆", force=False):
    local = Path(local)
    if not local.is_file():
        log(f"⚠ bank skip (missing): {local}")
        return False
    need = local.stat().st_size
    if not force and tree.get(repo_path) == need:
        return True
    rd = None
    try:
        pb = bar(f"{desc} {local.name}", need, "B", True)
        rd = ProgressReader(local, pb)
        api().upload_file(path_or_fileobj=rd, path_in_repo=repo_path,
                          repo_id=REPO, repo_type="dataset",
                          commit_message="cpt-train")
        tree[repo_path] = need
        return True
    except Exception as e:
        log(f"   (upload deferred: {str(e)[:90]})")
        return False
    finally:
        if rd is not None: rd.close()

def bank_dir(local_dir: Path, repo_sub: str, tree: dict, desc="⬆ ckpt"):
    path = Path(local_dir)
    order = {"adapter.pt": 0, "optim.pt": 1, "meta.json": 2, "README.md": 3}
    files = sorted((p for p in path.rglob("*") if p.is_file()),
                   key=lambda p: (order.get(p.name, 9), str(p)))
    oks = [bank(p, f"{repo_sub}/{p.relative_to(path)}", tree, desc=desc)
           for p in files]
    if not all(oks):
        log(f"⚠ some files of {repo_sub} deferred — rerun retries them")
    return all(oks)

# ═══════════════ TPU launch: one process per core (true data parallel) ════════
def _ensure_torch_xla():
    from importlib import import_module
    try:
        import_module("torch_xla")
        return
    except Exception as e:
        log(f"torch_xla missing ({str(e)[:70]}) — pip install torch_xla[tpu] …")
    import subprocess
    want = "torch_xla[tpu]==" + torch.__version__.split("+")[0]
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                         want]).returncode
    if rc != 0:
        log("❌ torch_xla install failed. Colab: Runtime → Change runtime → TPU; "
            "Kaggle: Settings → Accelerator → TPU VM v5e-8")
        raise SystemExit(1)
    try:
        import_module("torch_xla")
    except Exception as e:
        log(f"❌ torch_xla still not importable ({str(e)[:80]})")
        raise SystemExit(1)

def _in_ipython():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

def _relaunch_if_notebook():
    """xmp.spawn needs a real script; %run keeps everything in the kernel.

    Returns True when a child process already did the work (parent must stop).
    """
    if not _in_ipython() or os.environ.get("CPT_RELAUNCHED"):
        return False
    f = globals().get("__file__")
    if not f or not Path(f).is_file():
        log("❌ cannot auto-relaunch (no script path). Run from a terminal cell:\n"
            "   !python data/cpt_train.py --pilot 60")
        raise SystemExit(1)
    script = Path(f).resolve()
    log("notebook cell → relaunching as a plain python process for xmp.spawn")
    import subprocess
    env = dict(os.environ, CPT_RELAUNCHED="1")
    rc = subprocess.call([sys.executable, "-u", str(script), *sys.argv[1:]],
                         env=env)
    if rc:
        raise SystemExit(rc)
    return True

def _train_worker(index, cfg):
    """one TPU core: own process, own device, its shard of every global batch."""
    global ARGS, TPU, DEV, XM, WORLD, RANK, IS_MASTER
    import torch_xla.core.xla_model as xm
    try:
        import torch_xla.runtime as xr
        WORLD, RANK = int(xr.world_size()), int(xr.global_ordinal())
    except Exception:
        WORLD, RANK = int(index) + 1, int(index)
    ARGS = argparse.Namespace(**cfg)
    TPU, XM = True, xm
    IS_MASTER = (RANK == 0)
    DEV = xm.xla_device()
    try:
        _run(ARGS)
    except Exception as e:
        log(f"❌ worker {RANK} failed: {type(e).__name__}: {str(e)[:200]}")
        raise

# ═════════════════════════════════ main state ═════════════════════════════════
ARGS = None
TPU  = False
DEV  = None
XM   = None                        # torch_xla.core.xla_model when TPU
WORLD, RANK = 1, 0                 # data-parallel size / this process's index
IS_MASTER = True                   # rank 0: logging, checkpoints, HF uploads
TRAIN = dict(step=0, rows=0, loss=None, probe=[], started=time.time(),
             last_ckpt=time.time(), clock=time.time(), tok_s=None)

def load_manifest():
    """server copy is always truth (fresh download, ~KB) → shard ledger."""
    try:
        p = hf_hub_download(REPO, "cpt_manifest.json", repo_type="dataset")
        m = json.loads(Path(p).read_text())
    except Exception as e:
        log(f"❌ cpt_manifest.json unreachable ({str(e)[:80]})")
        log("   → run  %run data/cpt_prep.py  first (it banks the manifest)")
        raise SystemExit(1)
    tree = repo_tree()
    def _blk_no(name):
        return int(name.split("_")[-1].split(".")[0])   # "<fp>_block_000N.bin"
    names = sorted(m["pack"].get("shards", {}), key=_blk_no)
    blk = {n: int(m["pack"]["shards"][n]["blocks"]) for n in names}
    usable = []
    for name in names:
        present = f"{DATA_SUB}/{name}" in tree or \
                  m["pack"]["shards"][name].get("uploaded")
        if present:
            usable.append((name, blk[name]))
    if not usable:
        log("❌ no banked blocks found in the manifest — prep run incomplete")
        log("   → rerun data/cpt_prep.py until all shards are uploaded")
        raise SystemExit(1)
    if len(usable) < len(names):
        log(f"⚠ {len(names) - len(usable)} shard(s) not on repo yet — "
            "training continues on the banked subset (frozen at first ckpt)")
    return m, tree, usable, blk

# ═══════════════════════════ model (base + LoRA adapter) ══════════════════════
def _check_wrapped(model, n_linear):
    wrapped = [n for n, _ in model.named_modules()
               if "lora_A" in n or "lora_B" in n]
    if not wrapped:
        log(f"❌ peft wrapped 0 modules ({n_linear} Linear found) — "
            "target names did not match this model")
        raise SystemExit(1)
    log(f"peft: {len(wrapped)} lora matrices on {n_linear} Linear layers")

def rewire_lora_trainable(model):
    """XLA device moves can drop requires_grad — set it explicitly."""
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    n_tr = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    if n_tr == 0:
        heads = [f"{n}·{tuple(p.shape)}" for n, p in list(model.named_parameters())[:8]]
        log("❌ 0 trainable params even after rewire — first params:")
        for h in heads: log(f"   {h}")
        raise SystemExit(1)
    log(f"requires_grad rewired · trainable {n_tr/1e6:,.0f}M (lora-only)")

def build_model(rank):
    from peft import LoraConfig, get_peft_model
    attn = ARGS.attn
    cfg = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                     target_modules=list(TARGETS), bias="none",
                     task_type="CAUSAL_LM")
    if TPU:
        log(f"model: {BASE} · bf16 · attn {attn} → {DEV} …")
        model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, attn_implementation=attn)
        model.config.use_cache = False
        n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        torch.manual_seed(SEED)                # every rank: identical LoRA init
        model = get_peft_model(model, cfg)
        _check_wrapped(model, n_lin)
        model = model.to(DEV)
        rewire_lora_trainable(model)          # flags FIRST — XLA .to may drop them
        for n, p in model.named_parameters():  # bf16 adapters (16GB-core math)
            if "lora_" in n: p.data = p.data.to(torch.bfloat16)
        if WORLD > 1:                          # all ranks must print the SAME sha
            import hashlib
            q = next(p for n, p in model.named_parameters() if "lora_A" in n)
            raw = q.detach().to("cpu", torch.float32).numpy().tobytes()
            print(f"[rank {RANK}] init sha1 {hashlib.sha1(raw).hexdigest()[:12]}"
                  f" ({q.numel():,} floats)", flush=True)
    else:
        log(f"model: {BASE} · QLoRA nf4 · attn {attn} → cuda:0 …")
        ensure_libs(["bitsandbytes"], quiet=True)
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=bnb, device_map={"": 0},
            attn_implementation=attn)
        model.config.use_cache = False
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(     # canonical QLoRA order:
            model, use_gradient_checkpointing=True)  # prepare base, THEN LoRA
        n_lin = sum(1 for m in model.modules()
                    if isinstance(m, torch.nn.Linear)
                    or m.__class__.__name__.endswith("4bit"))
        torch.manual_seed(SEED)                # deterministic adapters
        model = get_peft_model(model, cfg)
        _check_wrapped(model, n_lin)
        rewire_lora_trainable(model)
        for n, p in model.named_parameters():  # fp16 adapters on the 4-bit base
            if "lora_" in n: p.data = p.data.to(torch.float16)
    if not getattr(model, "is_gradient_checkpointing", False) and not ARGS.no_ckpt:
        try:                                   # frozen base + GC needs grad inputs
            model.enable_input_require_grads()
        except Exception:
            try:
                model.base_model.model.enable_input_require_grads()
            except Exception as e:
                log(f"⚠ input-grad hook unavailable ({str(e)[:60]}) — "
                    "non-reentrant checkpointing should still work")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"model ready · trainable {n_tr/1e6:,.0f}M / {n_all/1e9:.1f}B "
        f"({100*n_tr/n_all:.1f}% · LoRA r={rank}) · "
        f"gradient_checkpointing={getattr(model, 'is_gradient_checkpointing', False)}")
    return model

# ═════════════ chunked cross-entropy (151k-vocab memory trick) ════════════════
CHUNK = 512       # tokens per loss chunk: B×512×151k×4B ≈ 0.3GB for B=1
def token_ce(logits, labels):
    """mean CE over next-token positions; fp32 chunks (full logits would OOM).

    Chunks over the *time* axis so we never materialise a full [B·T, V] copy.
    """
    B, T, V = logits.shape
    tot, n = None, 0
    for t0 in range(0, T - 1, CHUNK):
        t1 = min(t0 + CHUNK, T - 1)
        s = logits[:, t0:t1, :].float().reshape(-1, V)
        t = labels[:, t0 + 1:t1 + 1].reshape(-1)
        ce = torch.nn.functional.cross_entropy(s, t, reduction="sum")
        tot = ce if tot is None else (tot + ce)
        n += int(t.numel())
    return tot / float(max(n, 1))

# ═════════════════════════════ data stream (deterministic) ════════════════════
class RowStream:
    """
    frozen shard list → flat row iterator.
    • order = shard filename order (prep already seed-shuffled spans → stable
      mixture; no extra shuffle). Row identity is byte-stable across sessions.
    • the last eval_rows are never yielded (eval probe lives there).
    • start= skips rows on resume (cheap, no device work).
    """
    def __init__(self, shards):
        self.shards = shards                      # [(name, blocks)]
        self.R = sum(b for _, b in shards)
        self.eval_rows = min(HOLDOUT, max(0, self.R // 20))
        self.starts = []
        t = 0
        for _, b in shards:
            self.starts.append(t); t += b
        self._mm, self._si = None, -1

    def shard_of(self, i):
        k = bisect.bisect_right(self.starts, i) - 1
        return k, i - self.starts[k]

    def _mmap(self, k):
        if k != self._si:
            if self._mm is not None:
                try: self._mm._mmap.close()
                except Exception: pass
            name, b = self.shards[k]
            p = Path(hf_hub_download(REPO, f"{DATA_SUB}/{name}",
                                     repo_type="dataset"))
            self._mm = np.memmap(p, dtype=np.int32, mode="r", shape=(b, BLOCK))
            self._si = k
        return self._mm

    def row(self, i):
        k, off = self.shard_of(i)
        arr = np.asarray(self._mmap(k)[off], dtype=np.int64)
        return torch.from_numpy(arr)

    def close(self):
        if self._mm is not None:
            try: self._mm._mmap.close()
            except Exception: pass
            self._mm, self._si = None, -1

    def iter_shard(self, base, n_steps, skip_steps, accum, world, rank, lb):
        """this rank's rows for steps [skip_steps, n_steps) of an epoch.

        One global optimizer step consumes accum*world*lb consecutive rows:
        micro-batch m, rank r → rows  base + s*step_rows + m*world*lb + r*lb …
        Every rank computes the same (base, s, m, r) plan, takes a different
        slice, and gradients are all-reduced at optimizer_step → true DP.
        """
        step_rows = accum * world * lb
        try:
            for s in range(skip_steps, n_steps):
                b0 = base + s * step_rows
                for m in range(accum):
                    off = b0 + m * world * lb + rank * lb
                    for j in range(lb):
                        i = off + j
                        try:
                            y = self.row(i)
                        except Exception as e:
                            log(f"   row {i} unreadable → row 0 "
                                f"({str(e)[:50]})")
                            y = self.row(0)
                        yield {"input_ids": y, "labels": y.clone()}
        finally:
            self.close()

    def probe_rows(self):
        k = max(1, self.eval_rows // PROBE_ROWS)
        return list(range(self.R - self.eval_rows, self.R, k))[:PROBE_ROWS]

# ══════════════════════════════ WSD LR schedule ═══════════════════════════════
def wsd_factor(step, total):
    if step < WARM_STEPS: return (step + 1) / max(WARM_STEPS, 1)
    ds = int(total * (1.0 - DECAY_FRAC))
    if step < ds: return 1.0
    f = (total - step) / max(total - ds, 1)
    return DECAY_FLOOR + (1.0 - DECAY_FLOOR) * max(f, 0.0)

def wsd_milestones(total):
    ds = int(total * (1.0 - DECAY_FRAC))
    span = max(total - ds, 1)
    return sorted({min(ds + span // 3, total), min(ds + 2 * span // 3, total)})

# ══════════════════════════════ probe (eval gate) ═════════════════════════════
def probe(model, stream):
    rows = stream.probe_rows()
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        pbq = bar("probe", len(rows), unit="row")
        for r in pbq:
            try:
                x = stream.row(r).unsqueeze(0).to(DEV)
            except Exception:
                continue
            out = model(input_ids=x)
            tot += float(token_ce(out.logits, x).item()); cnt += 1
            del out
    model.train()
    if cnt < len(rows): log(f"⚠ probe used only {cnt} of {len(rows)} rows")
    return round(tot / max(cnt, 1), 4)

# ══════════════════════════ checkpoint save / resume ══════════════════════════
def _cpu(o):
    if isinstance(o, torch.Tensor): return o.detach().cpu()
    if isinstance(o, dict): return {k: _cpu(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return type(o)(_cpu(v) for v in o)
    return o

def lora_state(model):
    from peft import get_peft_model_state_dict
    return _cpu(get_peft_model_state_dict(model))

def set_lora(model, sd):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, sd)

def save_ckpt(model, optim, tree, stream, cfg_keys, final=False):
    if not IS_MASTER:                    # replicas are identical after allreduce
        TRAIN["last_ckpt"] = time.time()
        return True
    step = TRAIN["step"]
    d = TMP_ROOT / f"step_{step:06d}"; d.mkdir(parents=True, exist_ok=True)
    log(f"ckpt → saving step {step:,} (adapter + optimizer) …")
    torch.save(lora_state(model), d / "adapter.pt")
    torch.save(_cpu(optim.state_dict()), d / "optim.pt")
    meta = dict(step=step, rows=TRAIN["rows"], loss=TRAIN["loss"],
                probe=TRAIN["probe"][-32:], started=TRAIN["started"],
                saved=time.time(),
                shards=[n for n, _ in stream.shards],
                rows_epoch=stream.R - stream.eval_rows,
                holdout=stream.eval_rows, cfg=cfg_keys, final=final)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    if ARGS.pilot:                            # pilot: local only, no upload
        TRAIN["last_ckpt"] = time.time()
        log(f"pilot ckpt saved locally at {d} (no upload)")
        return True
    ok = bank_dir(d, f"{CKPT_SUB}/step_{step:06d}", tree)
    if ok:
        pt = TMP_ROOT / "latest.txt"
        pt.write_text(f"step_{step:06d}\n")
        bank(pt, f"{CKPT_SUB}/latest.txt", tree, desc="⬆ pointer", force=True)
        for p in d.iterdir(): p.unlink()
        d.rmdir()
        log(f"ckpt ✔ step {step:,} → {CKPT_SUB}/step_{step:06d} banked")
    else:
        log(f"⚠ ckpt step {step:,} kept local at {d} — rerun uploads it")
    TRAIN["last_ckpt"] = time.time()
    return ok

def _step_sort(name):
    try: return int(name.split("_")[-1])
    except Exception: return -1

def resume_peek(tree):
    """newest COMPLETE checkpoint on HF; download ONLY meta.json (cheap)."""
    cands = set()
    pref = f"{CKPT_SUB}/"
    for full in tree:
        if full.startswith(pref):
            tail = full[len(pref):]
            if "/" in tail: cands.add(tail.split("/")[0])
    try:
        p = Path(hf_hub_download(REPO, f"{CKPT_SUB}/latest.txt",
                                 repo_type="dataset"))
        cands.add(p.read_text().strip())
    except Exception:
        pass
    steps = sorted((c for c in cands if c.startswith("step_")),
                   key=_step_sort, reverse=True)
    need = ("adapter.pt", "optim.pt", "meta.json")
    for nam in steps:
        if tree and not all(f"{CKPT_SUB}/{nam}/{f}" in tree for f in need):
            log(f"resume: {nam} incomplete on HF — trying older")
            continue
        try:
            p = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/meta.json",
                                     repo_type="dataset"))
            meta = json.loads(p.read_text())
        except Exception as e:
            log(f"⚠ meta of {nam} unreadable ({str(e)[:70]}) — trying older")
            continue
        log(f"resume: newest checkpoint {nam} (meta loaded)")
        return nam, meta
    return None

def resume_load(model, optim, sched, nam, meta, stream):
    """load adapter + optimizer + position into the already-built objects."""
    cfg = meta.get("cfg", {})
    if cfg.get("shards") and cfg["shards"] != [n for n, _ in stream.shards]:
        log("⚠ ckpt shard list differs from stream — continuing anyway "
            "(the launcher froze the list from this ckpt)")
    if cfg.get("rank") and cfg.get("rank") != ARGS.rank:
        log(f"⚠ rank differs (ckpt {cfg.get('rank')} vs arg {ARGS.rank})")
    if cfg.get("lr") and cfg["lr"] != getattr(ARGS, "lr", None):
        log(f"⚠ lr differs (ckpt {cfg['lr']} vs now {ARGS.lr}) — "
            "the resumed schedule will not match the original")
    if cfg.get("max_steps_planned") and cfg["max_steps_planned"] != MAX_STEPS:
        log(f"⚠ max_steps_planned differs (ckpt {cfg['max_steps_planned']} "
            f"vs now {MAX_STEPS}) — schedule mismatch")
    try:
        ad = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/adapter.pt",
                                  repo_type="dataset"))
        set_lora(model, torch.load(ad, map_location="cpu", weights_only=False))
        log("resume: adapter loaded ✔")
    except Exception as e:
        log(f"⚠ adapter load failed ({str(e)[:70]}) — flat restart weights")
        return False
    try:
        op = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/optim.pt",
                                  repo_type="dataset"))
        optim.load_state_dict(torch.load(op, map_location="cpu", weights_only=False))
        for s in optim.state.values():
            for k, v in list(s.items()):
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(DEV)          # exp_avg, exp_avg_sq, step …
        log("resume: optimizer state loaded ✔")
    except Exception as e:
        log(f"⚠ optimizer state skipped ({str(e)[:70]}) — short re-warm")
    TRAIN["step"] = int(meta["step"])
    rps = int(cfg.get("rows_per_step") or ROWS_PER_STEP)
    TRAIN["rows"] = int(meta.get("rows", TRAIN["step"] * rps))
    if rps != ROWS_PER_STEP:
        TRAIN["rows"] = TRAIN["step"] * ROWS_PER_STEP   # re-anchor to this build
        log(f"⚠ rows/step changed ({rps} → {ROWS_PER_STEP}) — data position "
            "re-anchored to step × rows_per_step")
    TRAIN["loss"] = meta.get("loss")
    TRAIN["probe"] = list(meta.get("probe", []))
    TRAIN["started"] = meta.get("started", time.time())
    TRAIN["last_ckpt"] = time.time()
    for _ in range(TRAIN["step"]):            # deterministic WSD replay
        sched.step()                          # fresh LambdaLR is at epoch 0
    last = TRAIN["probe"][-1][1] if TRAIN["probe"] else "—"
    log(f"resume ✔ at step {TRAIN['step']:,} · rows {TRAIN['rows']:,} · "
        f"last probe {last}")
    return True

# ═══════════════════════════ optimizer + scheduler ════════════════════════════
def build_optim(model):
    ps = [p for p in model.parameters() if p.requires_grad]
    if not ps:
        lora_n = sum(1 for n, _ in model.named_modules() if "lora_" in n)
        log(f"❌ optimizer got 0 trainable params · lora modules {lora_n}")
        raise SystemExit(1)
    opt = torch.optim.AdamW(ps, lr=LR_PEAK, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: wsd_factor(st, MAX_STEPS))
    return opt, sched

# ═════════════════════════════ soup (decay-end average) ═══════════════════════
SNAP = {}
def snapshot(model, key):
    if not IS_MASTER: return                      # rank 0 ships the model
    SNAP[key] = lora_state(model)                 # CPU dict (~0.5GB/shot)
    log(f"soup: snapshot '{key}' captured ({len(SNAP)} held)")

def soup_state():
    if len(SNAP) < 2: return None
    ref = SNAP[list(SNAP)[0]]
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in ref.items()}
    for s in SNAP.values():
        for k, v in s.items(): avg[k] += v.float()
    return {k: (avg[k] / len(SNAP)).to(v.dtype) for k, v in ref.items()}

def try_soup_ship(model, stream):
    """probe(final) vs probe(soup); leave the winner loaded."""
    if not IS_MASTER: return
    SNAP["fin"] = lora_state(model)
    if len(SNAP) < 2:
        SNAP.pop("fin", None)
        log("soup: fewer than 2 decay snapshots (crash earlier?) — skip")
        return
    final_probe = probe(model, stream)            # measure final fresh
    TRAIN["probe"].append([TRAIN["step"], final_probe])
    log(f"soup: final probe {final_probe}")
    cur = lora_state(model)
    set_lora(model, soup_state())
    soup_probe = probe(model, stream)
    if soup_probe <= final_probe:
        log(f"soup: soup wins ({soup_probe} ≤ final {final_probe}) → shipping soup")
        TRAIN["probe"].append([TRAIN["step"], soup_probe])
    else:
        log(f"soup: final wins ({final_probe} < soup {soup_probe}) → shipping final")
        set_lora(model, cur)
        TRAIN["probe"].append([TRAIN["step"], final_probe])

# ══════════════════════════ merge + ship cpt.v1 ═══════════════════════════════
def ship(model, optim, stream, tree, cfg_keys):
    if not IS_MASTER: return
    if ARGS.pilot:
        log("pilot: ship skipped (the full run banks models/cpt.v1/)")
        return
    out = TMP_ROOT / "cpt_v1"; shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    if not TPU:
        log("GPU/QLoRA: 4-bit base cannot merge → banking adapter only")
        save_ckpt(model, optim, tree, stream, cfg_keys, final=True)
        return
    readme = (f"# cpt.v1 · code_rag lineage\n\n"
              f"- parent: {BASE} · method: LoRA r={ARGS.rank} "
              f"(q/k/v/o/gate/up/down)\n"
              f"- data: {len(stream.shards)} shards · {stream.R:,} blocks × "
              f"{BLOCK} tok · probe {TRAIN['probe'][-1][1] if TRAIN['probe'] else '—'}\n"
              f"- schedule: WSD peak {LR_PEAK:g} · decay {DECAY_FRAC:.2f} → "
              f"{DECAY_FLOOR} · final step {TRAIN['step']:,}\n")
    if free_gb(TMP_ROOT) < 12:
        log(f"⚠ only {free_gb(TMP_ROOT):.0f}GB free — need ~12GB to merge; "
            "banking the adapter only")
        save_ckpt(model, optim, tree, stream, cfg_keys, final=True)
        return
    (out / "README.md").write_text(readme)
    try:
        log("merge: moving model to CPU first (avoids 2×8GB device peak) …")
        model.to("cpu"); gc.collect()
        log("merge: unloading adapters into base weights …")
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True, max_shard_size="3GB")
        AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
        log(f"ship: banking merged bf16 → {SHIP_SUB}/ (~8GB, file-skip safe)")
        if bank_dir(out, SHIP_SUB, tree, desc="⬆ model"):
            log(f"ship ✔ {SHIP_SUB}/ complete — cpt.v1 ready for Phase 5")
    except Exception as e:
        log(f"⚠ merged ship failed ({str(e)[:140]}) — the final adapter "
            "checkpoint is already banked; rerun to retry the merge")
    finally:
        shutil.rmtree(out, ignore_errors=True); gc.collect()

# ═════════════════════════════ training drive ═════════════════════════════════
class _EpochDS(torch.utils.data.IterableDataset):
    """IterableDataset wrapper so DataLoader accepts the row generator."""
    def __init__(self, stream, base, n_steps, skip_steps, accum, world, rank, lb):
        self.p = (stream, base, n_steps, skip_steps, accum, world, rank, lb)
    def __iter__(self):
        return self.p[0].iter_shard(*self.p[1:])

def train_session(model, optim, sched, stream, pstream, tree, cfg_keys):
    step, rows_done = TRAIN["step"], TRAIN["rows"]
    per_epoch = stream.R - stream.eval_rows
    steps_epoch = max(0, per_epoch // ROWS_PER_STEP)
    total = MAX_STEPS                              # pilot cap honoured here
    log(f"plan: {total:,} steps · {steps_epoch:,} steps/epoch · "
        f"{EPOCHS} epochs · {ROWS_PER_STEP} seqs/step "
        f"({ROWS_TOKENS/1000:,.0f}k tok/step)")
    log(f"      world {WORLD} × micro {LOADER_B} × accum {ACCUM} · "
        f"usable {steps_epoch*ROWS_PER_STEP:,}/{per_epoch:,} rows/epoch · "
        f"WSD peak {LR_PEAK:g} (warm {int(100*WARM_FRAC)}% · "
        f"decay {int(100*DECAY_FRAC)}% → {DECAY_FLOOR})")
    if step > 0:
        log(f"      continuing at step {step:,} "
            f"({100*step/max(total,1):.0f}%)")
    milestones = set(wsd_milestones(total))
    pb = bar("train", max(0, total - step), unit="step")
    micro, run_loss_t, nrun = 0, None, 0
    for ep in range(EPOCHS):
        if step >= total: break
        consumed = max(0, (rows_done - ep * steps_epoch * ROWS_PER_STEP)
                       // ROWS_PER_STEP)
        if consumed >= steps_epoch:                # epoch fully consumed
            continue
        ds = _EpochDS(stream, 0, steps_epoch, consumed,
                      ACCUM, WORLD, RANK, LOADER_B)
        dl = torch.utils.data.DataLoader(ds, batch_size=LOADER_B,
                                         drop_last=True, num_workers=0)
        if TPU:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader
            inner = MpDeviceLoader(dl, DEV)
        else:
            inner = dl
        broke = False
        for batch in inner:
            if step >= total:
                broke = True; break
            x, y = batch["input_ids"], batch["labels"]
            out = model(input_ids=x)
            loss = token_ce(out.logits, y)
            del out                          # free ~1.2GB logits before backward
            (loss / ACCUM).backward()
            run_loss_t = loss.detach() if run_loss_t is None \
                else run_loss_t + loss.detach()   # device-side, 1 sync per log
            nrun += 1
            if nrun == 1:                    # fail fast on silent zero-grad wiring
                tr = [p for p in model.parameters() if p.requires_grad]
                ng = sum(1 for p in tr if p.grad is not None)
                if tr and ng == 0:
                    log("❌ no LoRA gradients after the first backward — "
                        "checkpointing/input-grad wiring dropped the graph")
                    raise SystemExit(1)
                if tr: log(f"grad check ✔ {ng} LoRA tensors received grads")
            micro += 1
            if micro == ACCUM:
                if TPU:
                    XM.optimizer_step(optim, barrier=True)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                micro = 0; step += 1; rows_done += ROWS_PER_STEP
                TRAIN["step"], TRAIN["rows"] = step, rows_done
                pb.update(1)
                if step % LOG_EVERY == 0:
                    ema = float(run_loss_t.item()) / max(nrun, 1)
                    tok_s = ROWS_TOKENS * LOG_EVERY / max(
                        time.time() - TRAIN["clock"], 1e-9)
                    TRAIN["loss"] = round(ema, 4)
                    TRAIN["tok_s"] = tok_s
                    pb.set_postfix(loss=f"{ema:.3f}",
                                   lr=f"{sched.get_last_lr()[0]:0.2e}",
                                   tok_s=f"{tok_s/1000:.1f}k")
                    TRAIN["step"], TRAIN["rows"] = step, rows_done
                    TRAIN["clock"] = time.time()
                    run_loss_t, nrun = None, 0
                if step % PROBE_EVERY == 0:
                    v = probe(model, pstream)   # separate memmap: no races
                    TRAIN["probe"].append([step, v])
                    log(f"probe @ {step:,}: loss {v} (history "
                        f"{[p[1] for p in TRAIN['probe'][-4:]]})")
                    save_ckpt(model, optim, tree, stream, cfg_keys)
                    pb.close(); pb = bar("train", total - step, unit="step")
                if step in milestones and not ARGS.pilot:
                    snapshot(model, f"m{len(SNAP)}")
                if time.time() - TRAIN["last_ckpt"] > CKPT_MIN * 60:
                    save_ckpt(model, optim, tree, stream, cfg_keys)
                    pb.close(); pb = bar("train", total - step, unit="step")
        del dl, ds, inner
        gc.collect()
        if not broke:            # loader threads are done → safe to unmap
            stream.close()
    TRAIN["step"], TRAIN["rows"] = step, rows_done
    pb.close()
    return step >= total

def pilot_selftest(model, optim, stream, cfg_keys):
    """prove the resume path: save locally → reload adapters → compare."""
    save_ckpt(model, optim, {}, stream, cfg_keys)     # pilot → local only
    d = TMP_ROOT / f"step_{TRAIN['step']:06d}"
    try:
        cur = lora_state(model)
        set_lora(model, torch.load(d / "adapter.pt", map_location="cpu",
                                   weights_only=False))
        re = lora_state(model)
        same = set(cur) == set(re) and all(torch.equal(cur[k], re[k]) for k in cur)
        n_tensors = len(torch.load(d / "optim.pt", map_location="cpu",
                                   weights_only=False).get("state", {}))
        log("selftest: adapter save→load "
            + ("✅ equal" if same else "❌ DIFFERS — resume would be corrupt"))
        log(f"selftest: optimizer round-trip ok ({n_tensors} tensors)")
        if not same: log("selftest ❌ — set_peft_model_state_dict key mismatch")
    finally:
        shutil.rmtree(d, ignore_errors=True)

# ════════════════════════════════════ main ════════════════════════════════════
def summary(step, total, done):
    log("══════════ CPT TRAIN SUMMARY ══════════")
    h = (time.time() - TRAIN["started"]) / 3600
    log(f"   steps {step:,}/{total:,} · rows {TRAIN['rows']:,} · "
        f"tokens {TRAIN['rows']*BLOCK:,.0f}")
    log(f"   losses: ema {TRAIN['loss']} · probes "
        f"{[p[1] for p in TRAIN['probe'][-5:]]}")
    log(f"   elapsed {h:.1f}h (session {time.time()-T0:.0f}s)")
    if done:
        log("✔ CPT done — lineage: Qwen3-4B → cpt.v1 (your ~286M-token mix)")
        log("next → data/rag_build.py (Phase 4 index) then Phase 5 SFT")
    else:
        hrs = (total - step) * ROWS_TOKENS / (
            (TRAIN.get("tok_s") or EST_TOK_S)) / 3600
        log(f"⚠ NOT finished · est remaining ≈ {hrs:.1f}h")
        log("  → rerun the SAME script on a fresh session (auto-resume)")

def _run(args):
    """one full training process (single GPU, or one rank of the TPU mesh)."""
    global ARGS, LR_PEAK, EPOCHS, MAX_STEPS, WARM_STEPS, LOADER_B, ACCUM, \
           ROWS_PER_STEP, ROWS_TOKENS, TRAIN
    ARGS, LR_PEAK = args, args.lr
    TRAIN = dict(step=0, rows=0, loss=None, probe=[], started=time.time(),
                 last_ckpt=time.time(), clock=time.time(), tok_s=None)
    pick_root()
    libs_report()
    if TPU:
        log(f"backend: TPU · world {WORLD} · rank {RANK} · bf16 · "
            f"one process per core")
    else:
        log(f"backend: CUDA · {torch.cuda.get_device_name(0)} · QLoRA")

    manifest, tree, shards_all, blk = load_manifest()
    peek = None if args.no_resume else resume_peek(tree)

    # data: a checkpoint's frozen shard list wins over the live manifest
    shards = shards_all
    if peek:
        frozen = (peek[1].get("cfg") or {}).get("shards") or []
        if frozen:
            missing = [n for n in frozen if n not in blk]
            if missing:
                log(f"❌ checkpoint {peek[0]} froze {len(missing)} shard(s) "
                    "missing from the manifest — step→data map would break")
                log("   → rerun with --no-resume for a fresh lineage")
                raise SystemExit(1)
            if frozen != [n for n, _ in shards_all]:
                log(f"data: freezing to the checkpoint's shard list "
                    f"({len(frozen)} shards)")
            shards = [(n, blk[n]) for n in frozen]

    stream = RowStream(shards)          # training rows
    pstream = RowStream(shards)         # eval rows — own memmap (no races)
    EPOCHS = args.epochs or (EPOCHS_TPU if TPU else EPOCHS_GPU)
    ARGS.rank = args.rank or (128 if TPU else 64)
    LOADER_B = args.micro_batch or (LBS_TPU if TPU else LBS_GPU)
    ACCUM = args.accum or max(1, G_BATCH // (LOADER_B * WORLD))
    ROWS_PER_STEP = LOADER_B * ACCUM * WORLD
    ROWS_TOKENS = ROWS_PER_STEP * BLOCK
    per_epoch = stream.R - stream.eval_rows
    steps_epoch = per_epoch // ROWS_PER_STEP
    planned = steps_epoch * EPOCHS
    MAX_STEPS = int(args.pilot) if args.pilot else planned
    if MAX_STEPS < 1:
        log(f"❌ stream too small ({per_epoch:,} rows → 0 steps). "
            "Rerun data/cpt_prep.py to bank more blocks.")
        raise SystemExit(1)
    WARM_STEPS = max(4, int(MAX_STEPS * WARM_FRAC))
    log(f"stream: {len(stream.shards)} shards · {stream.R:,} rows "
        f"({stream.R*BLOCK/1e6:.0f}M tok) · {stream.eval_rows:,} held-out · "
        f"{per_epoch:,} rows/epoch")
    log(f"plan: {MAX_STEPS:,} steps · {EPOCHS} epochs · "
        f"world {WORLD} × micro {LOADER_B} × accum {ACCUM} = "
        f"{ROWS_PER_STEP} seqs/step ({ROWS_TOKENS/1000:,.0f}k tok) · "
        f"est {MAX_STEPS*ROWS_TOKENS/EST_TOK_S/3600:.1f}h @ "
        f"{EST_TOK_S/1000:.0f}k tok/s")
    if ROWS_PER_STEP != G_BATCH:
        log(f"⚠ global batch {ROWS_PER_STEP} ≠ target {G_BATCH} "
            "(override with --micro-batch/--accum if unintended)")

    if TPU and RANK:
        time.sleep(min(RANK * 8, 60))   # stagger 8×9GB host loads a little
    model = build_model(ARGS.rank)
    optim, sched = build_optim(model)
    cfg_keys = dict(base=BASE, rank=ARGS.rank, lr=args.lr, epochs=EPOCHS,
                    accum=ACCUM, loader_b=LOADER_B, world=WORLD, seed=SEED,
                    rows_per_step=ROWS_PER_STEP, max_steps_planned=planned,
                    shards=[n for n, _ in stream.shards])
    if peek:
        try:
            loaded = resume_load(model, optim, sched, peek[0], peek[1], stream)
        except SystemExit:
            raise
        except Exception as e:
            log(f"❌ resume failed: {type(e).__name__}: {str(e)[:100]}")
            log("   → rerun (transient?) or use --no-resume for a fresh lineage")
            raise SystemExit(1)
        if not loaded:
            log("❌ checkpoint exists but its adapter did not load — refusing "
                "to silently restart into the same lineage")
            log("   → rerun (transient download?) or use --no-resume for a "
                "fresh lineage")
            raise SystemExit(1)
    elif args.no_resume:
        log("starting a FRESH lineage (--no-resume given)")

    if not TRAIN["probe"]:
        v = probe(model, pstream)
        TRAIN["probe"].append([TRAIN["step"], v])
        log(f"probe baseline: loss {v}")
    TRAIN["clock"] = time.time()

    done = False
    try:
        done = train_session(model, optim, sched, stream, pstream,
                             tree, cfg_keys)
        if args.pilot and done and IS_MASTER:
            pilot_selftest(model, optim, stream, cfg_keys)
        if done and not args.pilot:
            try: try_soup_ship(model, pstream)
            except Exception as e: log(f"soup skipped ({str(e)[:80]})")
            save_ckpt(model, optim, tree, stream, cfg_keys, final=True)
            ship(model, optim, stream, tree, cfg_keys)
        elif not args.pilot and TRAIN["step"] > 0:
            save_ckpt(model, optim, tree, stream, cfg_keys)   # crash guard
    except KeyboardInterrupt:
        log("⚠ interrupted — saving a checkpoint first …")
        try: save_ckpt(model, optim, tree, stream, cfg_keys)
        except Exception: pass
        raise
    except Exception as e:
        log(f"❌ train loop failed: {type(e).__name__}: {str(e)[:200]}")
        low = str(e).lower()
        if "out of memory" in low or "resource exhausted" in low:
            log("   → OOM: retry with --rank 64 (or 32) · keep gradient "
                "checkpointing · GPU: --micro-batch 1")
        try: save_ckpt(model, optim, tree, stream, cfg_keys)
        except Exception: pass
        raise
    summary(TRAIN["step"], MAX_STEPS, done)

def main():
    """parse → resolve token → TPU spawn (one process/core) or single GPU."""
    global ARGS, LR_PEAK, TPU, DEV, XM, WORLD, RANK, IS_MASTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", action="store_true", help="preflight report only")
    ap.add_argument("--pilot", type=int, default=0,
                    help=">0 = run N steps only (smoke test, local-only ckpt)")
    ap.add_argument("--backend", choices=["auto", "tpu", "gpu"], default="auto")
    ap.add_argument("--rank", type=int, default=None, help="LoRA rank override")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--lr", type=float, default=LR_PEAK)
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"default: {EPOCHS_TPU} on TPU, {EPOCHS_GPU} on GPU")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="seqs per core per forward (default: 1 TPU, 2 GPU)")
    ap.add_argument("--accum", type=int, default=None,
                    help="micro-batches per optimizer step (auto → 64 seqs)")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing (escape hatch)")
    ap.add_argument("--single-core", action="store_true",
                    help="TPU: 1 core in-process (fallback if xmp.spawn fails)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints; restart the lineage")
    args = ap.parse_args()
    ARGS, LR_PEAK = args, args.lr
    log(f"════════════ CPT TRAIN · seed {SEED} · block {BLOCK} · "
        f"pilot={args.pilot or 'off'} ════════════")

    # resolve the token here so spawned/relaunched children inherit it
    tok = get_token()
    if tok:
        os.environ["HF_TOKEN"] = tok

    if args.libs:
        libs_report(extra=["torch_xla"]); return

    tpu = (args.backend == "tpu") or \
          (args.backend == "auto" and _looks_like_tpu_env())
    if tpu:
        _ensure_torch_xla()
        if args.single_core:
            import torch_xla.core.xla_model as xm
            TPU, XM, DEV = True, xm, xm.xla_device()
            WORLD, RANK, IS_MASTER = 1, 0, True
            log("TPU single-core mode (1 core — ~8× slower; pilots only)")
            _run(args)
            return
        if _relaunch_if_notebook():
            return
        # Kaggle pre-sets these for its single-process runtime; xmp.spawn must
        # write its own multi-worker values or TPU init fails ("expected 8
        # worker addresses, got 1").
        for k in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
            os.environ.pop(k, None)
        # one shared-cache download beats eight cores racing the same HF locks
        try:
            from huggingface_hub import snapshot_download
            log(f"model: pre-downloading {BASE} to the shared cache …")
            snapshot_download(BASE)
        except Exception as e:
            log(f"⚠ pre-download skipped ({str(e)[:80]}) — cores will fetch")
        import torch_xla.distributed.xla_multiprocessing as xmp
        log("TPU: spawning one process per core (data-parallel) …")
        try:
            xmp.spawn(_train_worker, args=(vars(args),), nprocs=None)
        except Exception as e:
            log(f"❌ TPU launch failed: {type(e).__name__}: {str(e)[:180]}")
            log("   → attach a TPU runtime; or retry with --single-core "
                "(1 core, slow, pilot only); or --backend gpu")
            raise SystemExit(1)
        return

    if not torch.cuda.is_available():
        log("❌ no TPU detected and no CUDA GPU — in the notebook: "
            "Settings/Runtime → Accelerator → GPU or TPU")
        log("   (on a TPU runtime without env markers, rerun with --backend tpu)")
        raise SystemExit(1)
    TPU, DEV, XM = False, torch.device("cuda:0"), None
    WORLD, RANK, IS_MASTER = 1, 0, True
    log(f"backend: CUDA · {torch.cuda.get_device_name(0)} · QLoRA")
    log("⚠ T4 path is for pilots/debug — the free TPU is ~15-30× faster")
    _run(args)

if __name__ == "__main__":
    main()
