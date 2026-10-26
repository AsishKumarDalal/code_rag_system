#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 5 · STEP 2 — SFT TRAINING  (docs/phase_5_sft.md §6)

    cpt.v1 (frozen)  +  data/sft/episodes_*.jsonl  ->  models/sft.v2
        │
        ├─ 1 BASE     models/cpt.v1 from HF, fp16 (T4 has no bf16), FROZEN
        │             (the freeze is the preservation trick - Phase-3
        │              knowledge physically cannot wash out)
        ├─ 2 ADAPTER  fresh LoRA r=64 on q/k/v/o/gate/up/down (fp16)
        │             (r=128 on >=22GB GPUs; T4 needs r=64 to fit)
        ├─ 3 MASK     loss ONLY on generated tokens (<plan>/<search>/
        │             <thinking>/<answer>); question+obs are context -100
        ├─ 4 PACK     episodes -> deterministic bins of SEQ=2048 tokens
        │             (mid-episode continuation is legal; mask follows)
        ├─ 5 SCHED    cosine, 3% warmup, floor 10%, LR 2e-5, wd 0, 1 epoch
        │             (Nemotron Cascade's own card: ~1.5 epochs is optimal;
        │              T4 budget says 1 solid pass > 2 starved ones)
        ├─ 6 GUARD    val masked-CE probe every 400 steps on the fixed val
        │             split; hourly adapter+optimizer ckpt -> HF; auto-resume
        └─ 7 SHIP     best-by-val adapter -> merge -> models/sft.v2 on HF
                      (+README with parent/data-fp/recipe/val stats)

GPU reality (Oct 2026: TPU off, 2xT4 16GB, ~30h GPU quota / week):
    * single GPU from %run; BOTH GPUs via a terminal-style cell:
        !python -m torch.distributed.run --nproc_per_node=2 \
            sft/sft_train.py
    * fp16 base + fp16 LoRA (no 4-bit needed: adapters are tiny; plain fp16
      is faster than nf4 on T4 tensor cores and fits: ~13GB at seq 2048)
    * measured tok/s from the pilot drives the ETA banner - no fantasy ETAs
    * /kaggle/working stays untouched; caches+ckpts live in /tmp and on HF

Run (Kaggle GPU cell):
    %run sft/sft_train.py --libs           # versions only
    %run sft/sft_train.py --pilot 40       # 40 steps + save/reload selftest
    %run sft/sft_train.py                  # the real run (resume-safe)
================================================================================
"""
import os, sys, io, gc, json, math, time, shutil, argparse
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
try:
    _H = "/tmp/code_rag_sft/hf"; Path(_H).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", _H)
except Exception:
    pass

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
    bad = []
    for spec in specs:
        name = spec.split(">=")[0].strip()
        floor = spec.split(">=")[1].strip() if ">=" in spec else None
        cur = _pkg_version(name)
        ok = cur is not None and (floor is None or
                                  _ver_tuple(cur) >= _ver_tuple(floor))
        if not ok: bad.append((spec, cur))
    if bad:
        if not quiet:
            print("(auto-installing/upgrading: " +
                  ", ".join(f"{s} (have {c})" for s, c in bad) + ")",
                  flush=True)
        import subprocess
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                             *[s for s, _ in bad]]).returncode
        if rc != 0:
            print("\n! pip failed. On Kaggle run once:\n   !pip install -U " +
                  " ".join(s for s, _ in bad), flush=True)
            raise SystemExit(1)
    return [(spec.split(">=")[0], _pkg_version(spec.split(">=")[0]))
            for spec in specs]

CORE_LIBS = ["huggingface_hub>=0.23", "transformers>=4.51.0", "peft>=0.13.0",
             "accelerate", "sentencepiece", "safetensors", "numpy", "tqdm"]

try:
    import torch
except ImportError:
    print("! torch missing - attach a Kaggle GPU runtime or pip install torch")
    raise
ensure_libs(CORE_LIBS)
from huggingface_hub import HfApi, hf_hub_download        # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
try:
    from tqdm.auto import tqdm
except Exception:
    class _MiniBar:
        def __init__(self, total=None, desc="", **k):
            self.total, self.n = total or 0, 0
        def update(self, n=1): self.n += n
        def set_postfix(self, **k): pass
        def set_postfix_str(self, s): pass
        def close(self): pass
    tqdm = _MiniBar

# ════════════════════════════════ CONFIG ══════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO = "sddfhy546y/code_rag_dataset"     # same bank as Phase 3/5 build
DATA_MANIFEST = "sft_manifest.json"      # repo root (build writes it)
CKPT_SUB = "ckpt/sft"
SHIP_SUB = "models/sft.v2"
BASE_ID = "Qwen/Qwen3-4B"                # only used if cpt.v1 is missing (pilot)
CPT_SUB = "models/cpt.v1"                # frozen parent (Phase 3 ship)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]
SEED = 3407
SEQ = 2048                    # tokens per packed bin (T4-safe; --seq 4096 ok)
EPOCHS = 1                    # doc says 2; T4 budget says 1 (use --epochs 2)
LR_PEAK = 2.0e-5              # docs/phase_5_sft.md 6.2 (LoRA 1e-5..2e-5)
WARM_FRAC = 0.03
LR_FLOOR = 0.10
ACCUM = 16                    # micro-bins per optimizer step (16x2048 = 33k tok)
LOG_EVERY = 10
PROBE_EVERY = 100
PROBE_BINS = 16
CKPT_MIN = 55                 # <= minutes between kill-safe checkpoints
EST_TOK_S = 700               # T4 planning number (fp16 LoRA, measured in pilot)
TMP_ROOT = Path(os.environ.get("SFT_TRAIN_TMP", "/tmp/code_rag_sft"))
ROOT_CANDIDATES = [
    (Path("/tmp/code_rag_sft"), "ephemeral big disk"),
    (Path("/kaggle/working/.sft_stage"), "kaggle working disk (20GB quota)"),
]

# ════════════════════════════ small plumbing ══════════════════════════════════
T0 = time.time()
def now(): return datetime.now(timezone.utc).strftime("%H:%M:%S")
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
    os.environ["SFT_TRAIN_TMP"] = str(TMP_ROOT)
    log(f"disk: root {TMP_ROOT} (auto: {why}) · free {free_gb(TMP_ROOT):.0f}GB")

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

_API = None
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
            log("! no HF_TOKEN found - pushes only work if already logged in")
        _API = HfApi()
    return _API

def repo_tree():
    try:
        items = api().list_repo_tree(REPO, repo_type="dataset", recursive=True)
        return {i.path: getattr(i, "size", None) for i in items}
    except Exception:
        try:
            return {p: None for p in api().list_repo_files(REPO,
                                                           repo_type="dataset")}
        except Exception as e:
            log(f"! repo tree unavailable ({str(e)[:80]}) - uploads re-run")
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
        log(f"! bank skip (missing): {local}")
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
                          commit_message="sft-train")
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
        log(f"! some files of {repo_sub} deferred - rerun retries them")
    return all(oks)

# ═════════════════════════════ main state (rank-aware) ════════════════════════
ARGS = None
DEV = None
WORLD, RANK, LOCAL_RANK = 1, 0, 0
IS_MASTER = True
TRAIN = dict(step=0, loss=None, probe=[], started=time.time(),
             last_ckpt=time.time(), clock=time.time(), tok_s=None)

def detect_world():
    try:
        w = int(os.environ.get("WORLD_SIZE", "1"))
        r = int(os.environ.get("RANK", "0"))
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        return w, r, lr
    except Exception:
        return 1, 0, 0

# ═══════════════ base resolution: frozen parent (cpt.v1) or fallback ══════════
def resolve_base(args):
    """local folder for models/cpt.v1 (HF dataset repo), else hub fallback."""
    if args.base and args.base != "auto":
        if os.path.isdir(args.base):
            log(f"base: local dir {args.base}")
            return args.base
        log(f"base: hub {args.base} (no local dir)")
        return args.base
    tree = repo_tree()
    has = any(k.startswith(CPT_SUB + "/") for k in tree)
    if has:
        try:
            from huggingface_hub import snapshot_download
            d = snapshot_download(REPO, repo_type="dataset",
                                  allow_patterns=[f"{CPT_SUB}/*"])
            p = Path(d) / CPT_SUB       # snapshot keeps models/cpt.v1/ paths
            if p.is_dir() and (p / "config.json").exists():
                log(f"base: {CPT_SUB} from HF bank -> {p}")
                return str(p)
            log(f"! {CPT_SUB} banked but incomplete at {p} - checking fallback")
        except Exception as e:
            log(f"! {CPT_SUB} download failed ({str(e)[:90]})")
    if args.pilot:
        log(f"! {CPT_SUB} not found - PILOT uses {BASE_ID} (machinery test "
            "only; do not judge quality)")
        return BASE_ID
    log(f"X {CPT_SUB} is not on the HF repo - Phase 5 needs the Phase-3 ship.")
    log("  -> finish CPT first:  %run cpt/cpt_train.py")
    log("  -> or run a machinery pilot now:  %run sft/sft_train.py --pilot 40")
    raise SystemExit(1)

# ═══════════════════════════ model: frozen base + LoRA ════════════════════════
def _check_wrapped(model, n_linear):
    wrapped = [n for n, _ in model.named_modules()
               if "lora_A" in n or "lora_B" in n]
    if not wrapped:
        log(f"X peft wrapped 0 modules ({n_linear} Linear found) - "
            "target names did not match this model")
        raise SystemExit(1)
    log(f"peft: {len(wrapped)} lora matrices on {n_linear} Linear layers")

def rewire_lora_trainable(model):
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    n_tr = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    if n_tr == 0:
        log("X 0 trainable params even after rewire")
        raise SystemExit(1)
    return n_tr

def unwrap(model):
    return model.module if hasattr(model, "module") else model

def build_model(base_path, rank):
    from peft import LoraConfig, get_peft_model
    log(f"model: {base_path} · fp16 · attn {ARGS.attn} -> {DEV} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16,
        attn_implementation=ARGS.attn)
    model.config.use_cache = False
    n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    cfg = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                     target_modules=list(TARGETS), bias="none",
                     task_type="CAUSAL_LM")
    torch.manual_seed(SEED)                 # identical init on every rank
    model = get_peft_model(model, cfg)
    _check_wrapped(model, n_lin)
    model = model.to(DEV)
    n_tr = rewire_lora_trainable(model)
    for n, p in model.named_parameters():   # fp16 adapters on the frozen base
        if "lora_" in n: p.data = p.data.to(torch.float16)
    if not ARGS.no_ckpt:
        try:
            model.enable_input_require_grads()   # frozen base + GC needs it
        except Exception as e:
            log(f"! input-grad hook unavailable ({str(e)[:60]})")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_all = sum(p.numel() for p in model.parameters())
    log(f"model ready · trainable {n_tr/1e6:,.0f}M / {n_all/1e9:.1f}B "
        f"({100*n_tr/n_all:.1f}% · LoRA r={rank}) · "
        f"gradient_checkpointing={getattr(model, 'is_gradient_checkpointing', False)}")
    if WORLD > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], find_unused_parameters=False)
        log(f"ddp: wrapped rank {RANK}/{WORLD} on cuda:{LOCAL_RANK}")
    return model

# ═══════════════════ chunked masked cross-entropy (T4 memory) ═════════════════
CHUNK = 512      # time-chunk: 1 x 512 x 151936 x 4B ~= 0.31GB transient

def masked_ce(logits, labels):
    """(sum-over-masked-tokens, count). labels already -100 on context."""
    B, T, V = logits.shape
    tot, n = None, 0
    for t0 in range(0, T - 1, CHUNK):
        t1 = min(t0 + CHUNK, T - 1)
        s = logits[:, t0:t1, :].float().reshape(-1, V)
        t = labels[:, t0 + 1:t1 + 1].reshape(-1)
        keep = t != -100
        if not bool(keep.any()):
            continue
        ce = torch.nn.functional.cross_entropy(s[keep], t[keep],
                                               reduction="sum")
        tot = ce if tot is None else (tot + ce)
        n += int(keep.sum().item())
    return tot, n

# ══════════════════════ data: episodes -> packed bins ═════════════════════════
def load_data_manifest():
    try:
        p = Path(hf_hub_download(REPO, DATA_MANIFEST, repo_type="dataset"))
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"X {DATA_MANIFEST} unreachable ({str(e)[:80]})")
        log("  -> run  %run sft/sft_build.py  first (it banks the manifest)")
        raise SystemExit(1)

class EpisodeStream:
    """banked JSONL episodes -> deterministic token bins of exactly SEQ.

    Mid-episode bin cuts are legal (causal LM): the loss mask travels with
    the tokens. Bins are globally numbered from 0 (across epochs) so resume
    only needs one integer (start_bin) - no fragile byte offsets.
    """
    def __init__(self, files, tok, seq=SEQ):
        self.files, self.tok, self.seq = list(files), tok, seq
        self.eos = tok.eos_token_id
    def episodes(self):
        for name in self.files:
            p = Path(hf_hub_download(REPO, f"data/sft/{name}",
                                     repo_type="dataset"))
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
    def encode(self, ep):
        ids, mask = [], []
        for tag, text, masked in ep["parts"]:
            block = f"<{tag}>{text}</{tag}>\n"
            t = self.tok(block, add_special_tokens=False)["input_ids"]
            ids.extend(t)
            mask.extend([0 if masked else 1] * len(t))
        if self.eos is not None:
            ids.append(self.eos)
            mask.append(1)                  # learn to emit the stop token
        return ids, mask
    def count_tokens(self):
        n, k = 0, 0
        for name in self.files:
            p = Path(hf_hub_download(REPO, f"data/sft/{name}",
                                     repo_type="dataset"))
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        n += len(self.encode(json.loads(line))[0])
                    except Exception:
                        continue
            k += 1
            if k % 2 == 0 or k == len(self.files):
                log(f"tokenize: {k}/{len(self.files)} shards · "
                    f"{n/1e6:.1f}M tokens")
        return n
    def bins(self, epochs, start_bin=0):
        buf_i, buf_m, gbin = [], [], 0
        for _ep_i in range(int(epochs)):
            for ep in self.episodes():
                i, mm = self.encode(ep)
                buf_i.extend(i); buf_m.extend(mm)
                while len(buf_i) >= self.seq:
                    bi, bm = buf_i[:self.seq], buf_m[:self.seq]
                    del buf_i[:self.seq]; del buf_m[:self.seq]
                    if gbin >= start_bin:
                        yield gbin, bi, bm
                    gbin += 1

def mk_batch(ids, mask):
    x = torch.tensor([ids], dtype=torch.long)
    y = x.clone()
    y[0, torch.tensor([m == 0 for m in mask], dtype=torch.bool)] = -100
    return {"input_ids": x, "labels": y}

def build_val_bins(m, tok, n=PROBE_BINS):
    name = m.get("val_file")
    if not name:
        return []
    st = EpisodeStream([name], tok, SEQ)
    out = []
    for k, (gbin, bi, bm) in zip(range(n), st.bins(1, 0)):
        out.append(mk_batch(bi, bm))
    return out

def probe(model, val_bins):
    """masked-CE on the fixed val split; '' means no val (pilot build)."""
    if not val_bins:
        return None
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in val_bins:
            ids = batch["input_ids"].to(DEV)
            lab = batch["labels"].to(DEV)
            out = model(input_ids=ids)
            s, c = masked_ce(out.logits, lab)
            del out
            if s is not None:
                tot += float(s.item())
                n += c
    model.train()
    return round(tot / max(n, 1), 4)

def vram_report():
    try:
        p = torch.cuda.get_device_properties(0)
        tot = p.total_memory / 2**30
        log(f"cuda: {torch.version.cuda} · {p.name} · {tot:.0f}GB x"
            f"{torch.cuda.device_count()}")
        return tot
    except Exception:
        return 0

def mem_guard(vram_gb):
    """honest preflight: warn when the planned footprint is tight."""
    est = 8.6 + (0.9 if ARGS.rank <= 64 else 1.8) + 1.0 + \
          (1.2 if ARGS.seq >= 4096 else 0.6) + 1.2
    log(f"memory plan: ~{est:.1f}GB (base 8.6 · lora+optim "
        f"{'0.9' if ARGS.rank <= 64 else '1.8'} · activations+ckpt 1.0 · "
        f"logits {1.2 if ARGS.seq >= 4096 else 0.6} · headroom 1.2) "
        f"of {vram_gb:.0f}GB")
    if est > vram_gb - 0.5:
        log("! tight: if OOM -> lower --seq (2048/1024), --rank 32, "
            "or --micro-batch 1 (already default)")

# ══════════════════════ checkpoint save / resume (kill-safe) ══════════════════
def _barrier():
    if WORLD > 1:
        import torch.distributed as dist
        dist.barrier()

def _cpu(o):
    if isinstance(o, torch.Tensor): return o.detach().cpu()
    if isinstance(o, dict): return {k: _cpu(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return type(o)(_cpu(v) for v in o)
    return o

def lora_state(model):
    from peft import get_peft_model_state_dict
    return _cpu(get_peft_model_state_dict(unwrap(model)))

def set_lora(model, sd):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(unwrap(model), sd)

def save_ckpt(model, optim, scaler, tree, files, cfg_keys, final=False):
    step = TRAIN["step"]
    _barrier()
    if not IS_MASTER:
        TRAIN["last_ckpt"] = time.time()
        _barrier()
        return True
    d = TMP_ROOT / f"step_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    log(f"ckpt -> saving step {step:,} (adapter + optimizer + scaler) ...")
    torch.save(lora_state(model), d / "adapter.pt")
    torch.save({"optim": _cpu(optim.state_dict()),
                "scaler": _cpu(scaler.state_dict()) if scaler else {}},
               d / "optim.pt")
    meta = dict(step=step, loss=TRAIN["loss"], probe=TRAIN["probe"][-32:],
                started=TRAIN["started"], saved=time.time(), files=files,
                cfg=cfg_keys, final=final, world=WORLD)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    if ARGS.pilot:
        TRAIN["last_ckpt"] = time.time()
        log(f"pilot ckpt saved locally at {d} (no upload)")
        _barrier()
        return True
    ok = bank_dir(d, f"{CKPT_SUB}/step_{step:06d}", tree)
    if ok:
        pt = TMP_ROOT / "latest.txt"
        pt.write_text(f"step_{step:06d}\n")
        bank(pt, f"{CKPT_SUB}/latest.txt", tree, desc="⬆ pointer", force=True)
        for p in d.iterdir():
            p.unlink()
        d.rmdir()
        log(f"ckpt OK step {step:,} -> {CKPT_SUB}/step_{step:06d} banked")
    else:
        log(f"! ckpt step {step:,} kept local at {d} - rerun uploads it")
    TRAIN["last_ckpt"] = time.time()
    _barrier()
    return ok

def _step_sort(name):
    try: return int(name.split("_")[-1])
    except Exception: return -1

def resume_peek(tree):
    cands = set()
    pref = f"{CKPT_SUB}/"
    for full in tree:
        if full.startswith(pref):
            tail = full[len(pref):]
            if "/" in tail:
                cands.add(tail.split("/")[0])
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
            log(f"resume: {nam} incomplete on HF - trying older")
            continue
        try:
            p = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/meta.json",
                                     repo_type="dataset"))
            meta = json.loads(p.read_text())
        except Exception as e:
            log(f"! meta of {nam} unreadable ({str(e)[:70]}) - trying older")
            continue
        log(f"resume: newest checkpoint {nam} (meta loaded)")
        return nam, meta
    return None

def resume_load(model, optim, scaler, nam, meta, files):
    cfg = meta.get("cfg", {})
    hard = []
    if sorted(cfg.get("files") or []) != sorted(files):
        hard.append("data shard list changed")
    if int(cfg.get("seq") or 0) != SEQ:
        hard.append(f"seq {cfg.get('seq')} vs now {SEQ}")
    if int(cfg.get("accum") or 0) != ACCUM:
        hard.append(f"accum {cfg.get('accum')} vs now {ACCUM}")
    if int(cfg.get("world") or 1) != WORLD:
        hard.append(f"world {cfg.get('world')} vs now {WORLD}")
    if hard:
        log("X resume refused: " + " · ".join(hard))
        log("   the step-to-data map would be wrong - rerun with --no-resume "
            "for a fresh lineage")
        raise SystemExit(1)
    try:
        ad = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/adapter.pt",
                                  repo_type="dataset"))
        set_lora(model, torch.load(ad, map_location="cpu", weights_only=False))
        log("resume: adapter loaded OK")
    except Exception as e:
        log(f"X adapter load failed ({str(e)[:70]})")
        raise SystemExit(1)
    try:
        op = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/optim.pt",
                                  repo_type="dataset"))
        blob = torch.load(op, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "optim" in blob:
            optim.load_state_dict(blob["optim"])
            if scaler is not None and isinstance(blob.get("scaler"), dict) \
               and blob["scaler"]:
                try: scaler.load_state_dict(blob["scaler"])
                except Exception: pass
        else:
            optim.load_state_dict(blob)      # legacy adapter-only format
        for s in optim.state.values():
            for k, v in list(s.items()):
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(DEV)
        log("resume: optimizer state loaded OK")
    except Exception as e:
        log(f"! optimizer state skipped ({str(e)[:70]}) - short re-warm")
    TRAIN["step"] = int(meta["step"])
    TRAIN["loss"] = meta.get("loss")
    TRAIN["probe"] = list(meta.get("probe", []))
    TRAIN["started"] = meta.get("started", time.time())
    TRAIN["last_ckpt"] = time.time()
    log(f"resume OK at step {TRAIN['step']:,} · last probe "
        f"{TRAIN['probe'][-1][1] if TRAIN['probe'] else '-'}")
    return True

# ══════════════════════════ optimizer + cosine schedule ═══════════════════════
def cos_factor(step, total):
    warm = max(4, int(total * WARM_FRAC))
    if step < warm:
        return (step + 1) / warm
    prog = min(1.0, max(0.0, (step - warm) / max(total - warm, 1)))
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * prog))

def build_optim(model):
    ps = [p for p in unwrap(model).parameters() if p.requires_grad]
    if not ps:
        log("X optimizer got 0 trainable params")
        raise SystemExit(1)
    opt = torch.optim.AdamW(ps, lr=LR_PEAK, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: cos_factor(st, MAX_STEPS))
    return opt, sched

# ═══════════════════════════════ training drive ════════════════════════════════
SNAP = {}

def _bank_best(tree):
    """persist the best-val adapter with each hourly ckpt (crash-safe ship)."""
    if not IS_MASTER or not SNAP.get("dirty") or SNAP.get("best") is None:
        return
    p = TMP_ROOT / "best.pt"
    try:
        torch.save(SNAP["best"], p)
        if bank(p, f"{CKPT_SUB}/best.pt", tree, desc="⬆ best", force=True):
            SNAP["dirty"] = False
            log(f"banked best-val adapter (v={SNAP.get('best_v')})")
    finally:
        try: p.unlink()
        except Exception: pass

def train_session(model, optim, sched, scaler, stream, val_bins, tree,
                  files, cfg_keys):
    step = TRAIN["step"]
    start_bin = step * ACCUM * WORLD
    total = MAX_STEPS
    tr_params = [p for p in unwrap(model).parameters() if p.requires_grad]
    log(f"plan: {total:,} steps · {steps_epoch:,} steps/epoch · {EPOCHS} "
        f"epochs · {ACCUM}x{WORLD} bins/step = "
        f"{ACCUM*WORLD*SEQ/1000:,.0f}k tok/step")
    log(f"      cosine peak {LR_PEAK:g} · warm {int(100*WARM_FRAC)}% · "
        f"floor {int(100*LR_FLOOR)}% · wd 0 · seq {SEQ}")
    if step:
        log(f"      continuing at step {step:,} ({100*step/max(total,1):.0f}%)")
    def rank_bins():
        for gbin, bi, bm in stream.bins(EPOCHS, start_bin):
            if gbin % WORLD == RANK:
                yield gbin, bi, bm
    it = iter(rank_bins())
    pb = bar("train", max(0, total - step), unit="step")
    micro, run_t, nrun, done = 0, None, 0, False
    while step < total and not done:
        for _ in range(ACCUM):
            try:
                _gbin, bi, bm = next(it)
            except StopIteration:
                done = True
                break
            batch = mk_batch(bi, bm)
            ids = batch["input_ids"].to(DEV, non_blocking=True)
            lab = batch["labels"].to(DEV, non_blocking=True)
            out = model(input_ids=ids)
            s, n = masked_ce(out.logits, lab)
            if s is None or n == 0:
                # fully-masked bin (long question/obs): still run a connected
                # zero-loss backward so DDP keeps every rank's collectives in
                # lockstep (a skipped backward would desync NCCL)
                loss = out.logits[:, :1, :1].sum() * 0.0
                del out
                scaler.scale(loss / ACCUM).backward()
                continue
            loss = s / n
            del out
            scaler.scale(loss / ACCUM).backward()
            run_t = loss.detach() if run_t is None else run_t + loss.detach()
            nrun += 1
            micro += 1
            if nrun == 1:                       # fail fast on dead wiring
                ng = sum(1 for p in tr_params if p.grad is not None)
                if tr_params and ng == 0:
                    log("X no LoRA gradients after the first backward - "
                        "checkpointing/input-grad wiring dropped the graph")
                    raise SystemExit(1)
                log(f"grad check OK {ng} LoRA tensors received grads")
        if done:
            break
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(tr_params, 1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        sched.step()
        micro = 0
        step += 1
        TRAIN["step"] = step
        pb.update(1)
        if step % LOG_EVERY == 0:
            ema = float(run_t.item()) / max(nrun, 1)
            dt = max(time.time() - TRAIN["clock"], 1e-9)
            tok_s = ACCUM * WORLD * SEQ * LOG_EVERY / dt
            TRAIN["loss"] = round(ema, 4)
            TRAIN["tok_s"] = tok_s
            eta_h = (total - step) * ACCUM * WORLD * SEQ / max(tok_s, 1) / 3600
            pb.set_postfix(loss=f"{ema:.3f}",
                           lr=f"{sched.get_last_lr()[0]:0.2e}",
                           tok_s=f"{tok_s/1000:.1f}k",
                           eta=f"{eta_h:.1f}h")
            TRAIN["clock"] = time.time()
            run_t, nrun = None, 0
        if step % PROBE_EVERY == 0:
            v = probe(model, val_bins)
            TRAIN["probe"].append([step, v])
            log(f"probe @ {step:,}: val masked-CE {v} (history "
                f"{[p[1] for p in TRAIN['probe'][-4:]]})")
            if v is not None and IS_MASTER:
                if SNAP.get("best") is None or v < SNAP.get("best_v", 1e9):
                    SNAP["best"] = lora_state(model)
                    SNAP["best_v"] = v
                    SNAP["dirty"] = True
                    log(f"snap: best-val adapter kept (v={v})")
        if time.time() - TRAIN["last_ckpt"] > CKPT_MIN * 60:
            save_ckpt(model, optim, scaler, tree, files, cfg_keys)
            _bank_best(tree)
            pb.close()
            pb = bar("train", total - step, unit="step")
    pb.close()
    if done and step < total:
        log(f"! data stream ended at step {step:,}/{total:,} "
            "(bins exhausted; run stays resumable)")
    return step >= total

def pilot_selftest(model, optim, scaler, stream, files, cfg_keys):
    save_ckpt(model, optim, scaler, {}, files, cfg_keys)
    d = TMP_ROOT / f"step_{TRAIN['step']:06d}"
    try:
        cur = lora_state(model)
        set_lora(model, torch.load(d / "adapter.pt", map_location="cpu",
                                   weights_only=False))
        re = lora_state(model)
        same = set(cur) == set(re) and all(torch.equal(cur[k], re[k])
                                           for k in cur)
        n_t = len(torch.load(d / "optim.pt", map_location="cpu",
                             weights_only=False).get("optim", {}).get("state", {}))
        log("selftest: adapter save->load " +
            ("OK equal" if same else "X DIFFERS - resume would be corrupt"))
        log(f"selftest: optimizer round-trip ok ({n_t} tensors)")
        if not same:
            log("selftest X - set_peft_model_state_dict key mismatch")
    finally:
        shutil.rmtree(d, ignore_errors=True)

def format_smoke(model, tok, val_stream, n=192):
    """greedy one-sample sanity: does the loop grammar appear at all?"""
    m = unwrap(model)
    q = "Name three sorting algorithms and their worst-case complexity."
    try:
        if val_stream is not None:
            ep = next(val_stream.episodes())
            q = ep["parts"][0][1][:300]
    except Exception:
        pass
    prompt = f"<question>{q}</question>\n<plan>"
    m.eval()
    old_cache = getattr(m.config, "use_cache", False)
    m.config.use_cache = True          # generation wants the KV cache
    try:
        ids = tok(prompt, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        full = "<plan>" + txt
        checks = {
            "plan_closed": "</plan>" in full,
            "search_used": "<search>" in full,
            "thinking": "<thinking>" in full,
            "answer_closed": "</answer>" in full,
            "obs_hallucinated": "<obs>" in full,
        }
        log("format smoke: " + ", ".join(f"{k}={v}"
                                          for k, v in checks.items()))
        log("format smoke sample: " + txt[:220].replace("\n", " | "))
    except Exception as e:
        log(f"! format smoke failed ({str(e)[:90]})")
    finally:
        m.config.use_cache = old_cache
        m.train()

# ══════════════════════════ merge + ship sft.v2 ═══════════════════════════════
TOK = None
BASE_PATH = ""
SHIPPED = False

def ship(model, tree, files, cfg_keys):
    global SHIPPED
    if not IS_MASTER:
        return
    if ARGS.pilot:
        log("pilot: ship skipped (the full run banks models/sft.v2/)")
        return
    if SNAP.get("best") is not None:
        set_lora(model, SNAP["best"])
        log(f"ship: best-val adapter selected (val masked-CE "
            f"{SNAP.get('best_v')})")
    if free_gb(TMP_ROOT) < 14:
        log(f"! only {free_gb(TMP_ROOT):.0f}GB free - need ~14GB to merge; "
            "the final adapter checkpoint is banked, rerun to retry the ship")
        return
    out = TMP_ROOT / "sft_v2"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(
        "# sft.v2 · code_rag lineage\n\n"
        f"- parent: cpt.v1 (frozen) · method: LoRA r={ARGS.rank} "
        "(q/k/v/o/gate/up/down)\n"
        f"- data: {DATA_MANIFEST} fp "
        f"{cfg_keys.get('data_fp')} · {len(files)} shards · "
        f"{ACCUM*WORLD*SEQ} tok/step · {EPOCHS} epoch(s)\n"
        f"- schedule: cosine peak {LR_PEAK:g} · warm 3% · floor 10% · wd 0\n"
        f"- val: {TRAIN['probe'][-1][1] if TRAIN['probe'] else '-'} "
        "(masked CE on the fixed val split)\n"
        f"- final step {TRAIN['step']:,}\n")
    try:
        m = unwrap(model)
        log("merge: moving model to CPU first (avoids 2x8GB device peak) ...")
        m.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        log("merge: unloading adapters into the base weights ...")
        merged = m.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True,
                               max_shard_size="3GB")
        if TOK is not None:
            TOK.save_pretrained(out)
        log(f"ship: banking merged fp16 -> {SHIP_SUB}/ (~8GB, file-skip safe)")
        if bank_dir(out, SHIP_SUB, tree, desc="⬆ model"):
            log(f"ship OK {SHIP_SUB}/ complete - sft.v2 ready for Phase 6")
            SHIPPED = True
    except Exception as e:
        log(f"! merged ship failed ({str(e)[:140]}) - the final adapter "
            "checkpoint is already banked; rerun to retry the merge")
    finally:
        shutil.rmtree(out, ignore_errors=True)
        gc.collect()

# ════════════════════════════════ main ════════════════════════════════════════
MAX_STEPS = 0
steps_epoch = 0

def summary(step, total, done):
    log("══════════ SFT TRAIN SUMMARY ══════════")
    h = (time.time() - TRAIN["started"]) / 3600
    log(f"   steps {step:,}/{total:,} · tok/step {ACCUM*WORLD*SEQ:,} · "
        f"approx tokens {step*ACCUM*WORLD*SEQ/1e6:,.0f}M")
    log(f"   losses: ema {TRAIN['loss']} · val probes "
        f"{[p[1] for p in TRAIN['probe'][-5:]]}")
    log(f"   elapsed {h:.1f}h (session {time.time()-T0:.0f}s) · "
        f"tok/s {TRAIN.get('tok_s') and int(TRAIN['tok_s']) or '-'}")
    if done:
        log("OK SFT done - lineage: Qwen3-4B -> cpt.v1 -> sft.v2")
        if SHIPPED:
            log("next -> Phase 6 (GRPO): the policy now speaks the loop grammar")
        else:
            log("next -> adapter is banked; rerun to retry the merged ship")
    else:
        est = (total - step) * ACCUM * WORLD * SEQ / max(
            TRAIN.get("tok_s") or EST_TOK_S, 1) / 3600
        log(f"! NOT finished · est remaining ~{est:.1f}h")
        log("  -> rerun the SAME script on a fresh session (auto-resume)")

def main():
    global ARGS, DEV, WORLD, RANK, LOCAL_RANK, IS_MASTER, LR_PEAK, SEQ, \
           EPOCHS, ACCUM, MAX_STEPS, steps_epoch, TOK, BASE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", action="store_true", help="preflight report only")
    ap.add_argument("--pilot", type=int, default=0,
                    help=">0 = N steps only (smoke; local ckpt, no ship)")
    ap.add_argument("--base", default="auto",
                    help="auto = models/cpt.v1 from HF · or a hub id / local dir")
    ap.add_argument("--rank", type=int, default=64, help="LoRA rank (T4: 64)")
    ap.add_argument("--lr", type=float, default=LR_PEAK)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seq", type=int, default=SEQ, help="tokens per bin")
    ap.add_argument("--accum", type=int, default=ACCUM,
                    help="micro-bins per optimizer step")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing (escape hatch)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints; restart the lineage")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap total steps (budget control)")
    ap.add_argument("--backend", choices=["cuda"], default="cuda")
    args = ap.parse_args()
    ARGS = args
    LR_PEAK, SEQ, EPOCHS = args.lr, args.seq, args.epochs
    ACCUM = args.accum
    WORLD, RANK, LOCAL_RANK = detect_world()
    IS_MASTER = (RANK == 0)
    TRAIN.update(step=0, loss=None, probe=[], started=time.time(),
                 last_ckpt=time.time(), clock=time.time(), tok_s=None)
    log(f"════════════ SFT TRAIN · seed {SEED} · pilot={args.pilot or 'off'} "
        f"════════════")
    tok0 = get_token()
    if tok0:
        os.environ["HF_TOKEN"] = tok0
    if args.libs:
        log("libs: python " + sys.version.split()[0] + " · torch " +
            torch.__version__ + " · transformers " +
            str(_pkg_version("transformers")) + " · peft " +
            str(_pkg_version("peft")))
        vram_report()
        return
    if not torch.cuda.is_available():
        log("X no CUDA GPU detected. Kaggle: Settings -> Accelerator -> "
            "GPU T4 x2 (TPU is not used in this phase)")
        raise SystemExit(1)
    torch.cuda.set_device(LOCAL_RANK)
    DEV = torch.device(f"cuda:{LOCAL_RANK}")
    if WORLD > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        log(f"ddp: world {WORLD} · rank {RANK} · local {LOCAL_RANK}")
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    pick_root()
    vram = vram_report()
    mem_guard(vram)
    BASE_PATH = resolve_base(args)
    TOK = AutoTokenizer.from_pretrained(
        BASE_PATH if os.path.isdir(str(BASE_PATH)) else BASE_ID)
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token
    tree = repo_tree()
    m = load_data_manifest()
    files = sorted(m.get("files") or {})
    if not files:
        log("X sft_manifest.json has no data shards - run sft/sft_build.py")
        raise SystemExit(1)
    log(f"data: {len(files)} shards · {m.get('episodes', 0):,} train eps · "
        f"val {m.get('val', 0):,} · fp {m.get('fp')} · "
        f"manifest tokens approx {m.get('token_est', 0)/1e6:.0f}M")
    if not m.get("val_file"):
        log("! no val split in the manifest (pilot build?) - probes will be "
            "skipped; the pilot still exercises everything else")
    peek = None if args.no_resume else resume_peek(tree)

    total_tok = 0
    if IS_MASTER:
        log("tokenize: one honest CPU pass to plan exact steps ...")
        total_tok = EpisodeStream(files, TOK, SEQ).count_tokens()
        log(f"tokens: {total_tok/1e6:.1f}M train tokens in {len(files)} shards")
    if WORLD > 1:
        import torch.distributed as dist
        t = torch.tensor([total_tok], dtype=torch.long, device=DEV)
        dist.broadcast(t, src=0)
        total_tok = int(t.item())
    bpe = total_tok // SEQ
    steps_epoch = bpe // (ACCUM * WORLD)
    planned = steps_epoch * EPOCHS
    MAX_STEPS = int(args.pilot) if args.pilot else int(args.max_steps or planned)
    if MAX_STEPS < 1:
        log(f"X data too small: {total_tok:,} tokens -> 0 steps "
            "(rebuild with more episodes)")
        raise SystemExit(1)
    log(f"plan: {MAX_STEPS:,} steps · {steps_epoch:,} steps/epoch · "
        f"{EPOCHS} epoch(s) · world {WORLD} x accum {ACCUM} x seq {SEQ} = "
        f"{ACCUM*WORLD*SEQ/1000:,.0f}k tok/step")
    log(f"      est {MAX_STEPS*ACCUM*WORLD*SEQ/EST_TOK_S/3600:.1f}h @ "
        f"{EST_TOK_S/1000:.1f}k tok/s (T4 planning number; pilot refines)")

    model = build_model(BASE_PATH, args.rank)
    optim, sched = build_optim(model)
    try:
        scaler = torch.amp.GradScaler("cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler()
    cfg_keys = dict(base=str(BASE_PATH), rank=args.rank, lr=LR_PEAK,
                    seq=SEQ, accum=ACCUM, world=WORLD, epochs=EPOCHS,
                    files=files, max_steps_planned=planned, data_fp=m.get("fp"))

    if peek:
        try:
            resume_load(model, optim, scaler, peek[0], peek[1], files)
        except SystemExit:
            raise
        except Exception as e:
            log(f"X resume failed: {type(e).__name__}: {str(e)[:120]}")
            log("  -> rerun (transient?) or use --no-resume for a fresh lineage")
            raise SystemExit(1)
        for _ in range(TRAIN["step"]):
            sched.step()
        try:
            bp = Path(hf_hub_download(REPO, f"{CKPT_SUB}/best.pt",
                                      repo_type="dataset"))
            SNAP["best"] = torch.load(bp, map_location="cpu",
                                      weights_only=False)
            vals = [p[1] for p in TRAIN["probe"] if p[1] is not None]
            SNAP["best_v"] = min(vals) if vals else 1e9
            SNAP["dirty"] = False
            log(f"resume: best-val adapter restored (best_v={SNAP['best_v']})")
        except Exception:
            pass
    elif args.no_resume:
        log("starting a FRESH lineage (--no-resume given)")

    val_bins = build_val_bins(m, TOK)
    val_stream = EpisodeStream([m["val_file"]], TOK, SEQ) if m.get("val_file") \
        else None
    if not TRAIN["probe"]:
        v = probe(model, val_bins)
        if v is not None:
            TRAIN["probe"].append([TRAIN["step"], v])
            log(f"probe baseline: val masked-CE {v}")
    TRAIN["clock"] = time.time()

    done = False
    try:
        done = train_session(model, optim, sched, scaler,
                             EpisodeStream(files, TOK, SEQ),
                             val_bins, tree, files, cfg_keys)
        if IS_MASTER:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            format_smoke(model, TOK, val_stream)
        if args.pilot and done and IS_MASTER:
            pilot_selftest(model, optim, scaler, None, files, cfg_keys)
        if done and not args.pilot:
            save_ckpt(model, optim, scaler, tree, files, cfg_keys, final=True)
            _bank_best(tree)
            ship(model, tree, files, cfg_keys)
        elif not args.pilot and TRAIN["step"] > 0:
            save_ckpt(model, optim, scaler, tree, files, cfg_keys)
    except KeyboardInterrupt:
        log("! interrupted - saving a checkpoint first ...")
        try:
            save_ckpt(model, optim, scaler, tree, files, cfg_keys)
        except Exception:
            pass
        raise
    except Exception as e:
        log(f"X train loop failed: {type(e).__name__}: {str(e)[:200]}")
        low = str(e).lower()
        if "out of memory" in low:
            log("   -> OOM: retry --seq 1024 --rank 32 (or lower --accum); "
                "keep gradient checkpointing on")
        try:
            save_ckpt(model, optim, scaler, tree, files, cfg_keys)
        except Exception:
            pass
        raise
    summary(TRAIN["step"], MAX_STEPS, done)
    _barrier()          # workers wait here while the master finishes the ship
    if WORLD > 1:
        try:
            import torch.distributed as dist
            dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()



