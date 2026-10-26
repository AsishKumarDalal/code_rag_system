#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3 · STEP 2 — CPT TRAINING  (docs/phase3.md · Part I)

    cpt_manifest.json (data/cpt/*_block_*.bin on HF · ~286M tokens · 4096/block)
        │
        ├─ ① LIB CHECK  torch/transformers/peft/accelerate/hub/numpy/tqdm
        ├─ ② SETUP      Qwen3-4B: LoRA r=128 on TPU (bf16) / QLoRA r=64 on T4
        ├─ ③ TRAIN      WSD · grad-accum to 64 seqs/step · chunked LM-head CE
        ├─ ④ GUARD      probe on held-out tail blocks every 500 steps
        ├─ ⑤ RESUME     newest HF checkpoint → adapter + optimizer + row position
        └─ ⑥ SHIP       soup decay snapshots → probe → merge (TPU) → models/cpt.v1

Backends (auto):
    TPU         one process per core via xmp.spawn (true data parallel)
    2×T4        one process per GPU via torch mp.spawn + NCCL (data parallel)
    1×T4        single-process QLoRA (pilots/debug only)

Memory safety:
    • micro-batch 1 (GPU), gradient checkpointing, expandable CUDA segments
    • decoder run directly + frozen LM head applied in small chunks under
      torch.checkpoint → no full [1, 4096, 151k] logits tensor
    • pre-Ampere SDPA: repeat KV heads so the fused kernel is used (native GQA
      otherwise falls back to math attention and needs ~13 GiB on a T4)

Run:
    %run data/cpt_train.py --libs
    %run data/cpt_train.py --pilot 20
    %run data/cpt_train.py
"""
import os, sys, io, gc, json, time, shutil, bisect, argparse, hashlib, socket
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _looks_like_tpu_env():
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
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

try:
    _H = "/tmp/code_rag_train/hf"
    Path(_H).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", _H)
except Exception:
    pass


# ════════════════════════════ ① library preflight ════════════════════════════
def _ver_tuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        num = "".join(c for c in part if c.isdigit())
        out.append(int(num) if num else 0)
    while len(out) < 3:
        out.append(0)
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
        ok = cur is not None and (floor is None or _ver_tuple(cur) >= _ver_tuple(floor))
        if not ok:
            bad.append((spec, cur))
    if bad:
        if not quiet:
            print("(auto-installing/upgrading: " +
                  ", ".join(f"{s} (have {c})" for s, c in bad) + ")", flush=True)
        import subprocess
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                             *[s for s, _ in bad]]).returncode
        if rc != 0:
            print(f"\n❌ pip failed. Run once:\n"
                  f"   !pip install -U {' '.join(s for s, _ in bad)}")
            raise SystemExit(1)
    return [(spec.split(">=")[0].strip(),
             _pkg_version(spec.split(">=")[0].strip())) for spec in specs]


CORE_LIBS = ["huggingface_hub>=0.23", "transformers>=4.51.0", "peft>=0.13.0",
             "accelerate", "sentencepiece", "safetensors", "numpy", "tqdm"]

try:
    import torch
except ImportError:
    print("❌ torch is not installed — attach a GPU/TPU runtime (Kaggle/Colab)")
    raise
try:
    import numpy as np
except ImportError:
    ensure_libs(["numpy"])
    import numpy as np

ensure_libs(CORE_LIBS)
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from tqdm.auto import tqdm
except Exception:
    class _MiniBar:
        def __init__(self, total=None, desc="", **k):
            self.total, self.n = total or 0, 0
        def update(self, n=1): self.n += n
        def set_postfix(self, **k): pass
        def close(self): pass
    tqdm = _MiniBar


# ══════════════════════════════════ config ═══════════════════════════════════
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
REPO        = "sddfhy546y/code_rag_dataset"
DATA_SUB    = "data/cpt"
CKPT_SUB    = "ckpt/cpt"
SHIP_SUB    = "models/cpt.v1"
BASE        = "Qwen/Qwen3-4B"
TARGETS     = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
SEED        = 3407
BLOCK       = 4096
EPOCHS_TPU  = 3
EPOCHS_GPU  = 1
LR_PEAK     = 1.0e-4
DECAY_FRAC  = 0.20
DECAY_FLOOR = 0.05
WARM_FRAC   = 0.01
G_BATCH     = 64
LBS_TPU     = 1
LBS_GPU     = 1
LOG_EVERY   = 20
PROBE_EVERY = 500
PROBE_ROWS  = 32
HOLDOUT     = 512
CKPT_MIN    = 55
EST_TOK_S   = 27000
LOSS_CHUNK  = 256
TMP_ROOT    = Path(os.environ.get("CPT_TRAIN_TMP", "/tmp/code_rag_train"))
ROOT_CANDIDATES = [
    (Path("/tmp/code_rag_train"),            "ephemeral big disk"),
    (Path("/kaggle/working/code_rag_train"), "kaggle working disk (20GB quota)"),
]


# ═══════════════════════════════ plumbing ════════════════════════════════════
T0 = time.time()


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(msg):
    if IS_MASTER:
        print(f"[{now()}] {msg}", flush=True)


def gfmt(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} TB"


def bar(desc, total=None, unit="it", scale=False):
    try:
        return tqdm(total=total, desc=desc, unit=unit, unit_scale=scale,
                    dynamic_ncols=True, disable=not IS_MASTER)
    except TypeError:
        return tqdm(total=total, desc=desc)


def disk_free(p):
    try:
        return shutil.disk_usage(str(p)).free
    except Exception:
        return 0


def free_gb(p):
    return disk_free(p) / 1024 ** 3


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


def _retry(fn, tries=3, delay=3.0, what="hf"):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i + 1 < tries:
                log(f"⚠ {what} retry {i + 1}/{tries}: {str(e)[:80]}")
                time.sleep(delay * (i + 1))
    raise last


# ═══════════════════════════════ HF plumbing ═════════════════════════════════
_API = None


def get_token():
    t = os.environ.get("HF_TOKEN", "")
    if t:
        return t
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
            log("⚠ no HF_TOKEN found — pushes need an existing login")
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


def bank(local, repo_path, tree, desc="⬆", force=False):
    local = Path(local)
    if not local.is_file():
        log(f"⚠ bank skip (missing): {local}")
        return False
    need = local.stat().st_size
    if not force and tree.get(repo_path) == need:
        return True
    try:
        api().upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                          repo_id=REPO, repo_type="dataset",
                          commit_message="cpt-train")
        tree[repo_path] = need
        log(f"{desc} {local.name} → {repo_path} ({gfmt(need)})")
        return True
    except Exception as e:
        log(f"   (upload deferred {local.name}: {str(e)[:90]})")
        return False


def bank_dir(local_dir, repo_sub, tree, desc="⬆ ckpt", force=False):
    path = Path(local_dir)
    order = {"adapter.pt": 0, "optim.pt": 1, "meta.json": 2, "README.md": 3}
    files = sorted((p for p in path.rglob("*") if p.is_file()),
                   key=lambda p: (order.get(p.name, 9), str(p)))
    oks = [bank(p, f"{repo_sub}/{p.relative_to(path)}", tree,
                desc=desc, force=force) for p in files]
    if not all(oks):
        log(f"⚠ some files of {repo_sub} deferred — rerun retries them")
    return all(oks)


# ═══════════════════════════════ workers ═════════════════════════════════════
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
        log("❌ torch_xla install failed. Choose a TPU runtime.")
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
    if not _in_ipython() or os.environ.get("CPT_RELAUNCHED"):
        return False
    f = globals().get("__file__")
    if not f or not Path(f).is_file():
        log("❌ cannot auto-relaunch (no script path). Run:\n"
            "   !python data/cpt_train.py --pilot 20")
        raise SystemExit(1)
    script = Path(f).resolve()
    log("notebook cell → relaunching as a plain python process for spawn")
    import subprocess
    env = dict(os.environ, CPT_RELAUNCHED="1")
    rc = subprocess.call([sys.executable, "-u", str(script), *sys.argv[1:]],
                         env=env)
    if rc:
        raise SystemExit(rc)
    return True


def _train_worker(index, cfg):
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
        print(f"[tpu {RANK}] ❌ worker failed: {type(e).__name__}: "
              f"{str(e)[:300]}", flush=True)
        raise


def _gpu_worker(index, world, cfg):
    global ARGS, TPU, DEV, XM, WORLD, RANK, IS_MASTER
    import torch.distributed as dist
    ARGS = argparse.Namespace(**cfg)
    TPU, XM = False, None
    WORLD, RANK = int(world), int(index)
    IS_MASTER = (RANK == 0)
    torch.cuda.set_device(RANK)
    DEV = torch.device(f"cuda:{RANK}")
    dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD,
                            timeout=timedelta(hours=2))
    try:
        _run(ARGS)
    except Exception as e:
        print(f"[gpu {RANK}] ❌ worker failed: {type(e).__name__}: "
              f"{str(e)[:300]}", flush=True)
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# ══════════════════════════════ main state ═══════════════════════════════════
ARGS = None
TPU = False
DEV = None
XM = None
WORLD, RANK = 1, 0
IS_MASTER = True
TRAIN = dict(step=0, rows=0, loss=None, probe=[], started=time.time(),
             last_ckpt=time.time(), clock=time.time(), tok_s=None)


# ═════════════════════════════ manifest + rows ═══════════════════════════════
def _block_no(name):
    try:
        return int(name.split("_")[-1].split(".")[0])
    except Exception:
        return 0


def load_manifest():
    try:
        p = _retry(lambda: hf_hub_download(REPO, "cpt_manifest.json",
                                           repo_type="dataset"),
                   what="manifest")
        m = json.loads(Path(p).read_text())
    except Exception as e:
        log(f"❌ cpt_manifest.json unreachable ({str(e)[:80]})")
        log("   → run data/cpt_prep.py first (it banks the manifest)")
        raise SystemExit(1)
    tree = repo_tree()
    spec = (m.get("pack") or {}).get("shards") or {}
    names = sorted(spec, key=_block_no)
    blk = {n: int(spec[n]["blocks"]) for n in names}
    usable = []
    for name in names:
        if f"{DATA_SUB}/{name}" in tree or spec[name].get("uploaded"):
            usable.append((name, blk[name]))
    if not usable:
        log("❌ no banked blocks found in the manifest — prep run incomplete")
        raise SystemExit(1)
    if len(usable) < len(names):
        log(f"⚠ {len(names) - len(usable)} shard(s) not on repo yet — "
            "training continues on the banked subset")
    return m, tree, usable, blk


def predownload_data():
    try:
        _, _, shards, _ = load_manifest()
        for name, _ in shards:
            _retry(lambda n=name: hf_hub_download(
                REPO, f"{DATA_SUB}/{n}", repo_type="dataset"),
                what=f"shard {name}")
        log(f"data: cached {len(shards)} shard(s) in the shared HF cache")
    except Exception as e:
        log(f"⚠ data pre-download skipped ({str(e)[:90]})")


class RowStream:
    def __init__(self, shards):
        self.shards = list(shards)
        self.R = sum(b for _, b in self.shards)
        self.eval_rows = min(HOLDOUT, max(0, self.R // 20))
        self.starts = []
        t = 0
        for _, b in self.shards:
            self.starts.append(t)
            t += b
        self._mm, self._si = None, -1

    def shard_of(self, i):
        if i < 0 or i >= self.R:
            raise IndexError(f"row {i} outside [0,{self.R})")
        k = bisect.bisect_right(self.starts, i) - 1
        return k, i - self.starts[k]

    def _mmap(self, k):
        if k != self._si:
            if self._mm is not None:
                try:
                    self._mm._mmap.close()
                except Exception:
                    pass
            name, b = self.shards[k]
            p = _retry(lambda: hf_hub_download(
                REPO, f"{DATA_SUB}/{name}", repo_type="dataset"),
                what=f"shard {name}")
            self._mm = np.memmap(p, dtype=np.int32, mode="r", shape=(b, BLOCK))
            self._si = k
        return self._mm

    def row(self, i):
        k, off = self.shard_of(i)
        return torch.from_numpy(np.asarray(self._mmap(k)[off], dtype=np.int64))

    def close(self):
        if self._mm is not None:
            try:
                self._mm._mmap.close()
            except Exception:
                pass
            self._mm, self._si = None, -1

    def iter_batch(self, base, n_steps, skip_steps, accum, world, rank, lb):
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
                            log(f"   row {i} unreadable → row 0 ({str(e)[:50]})")
                            y = self.row(0)
                        yield {"input_ids": y, "labels": y.clone()}
        finally:
            self.close()

    def probe_rows(self):
        if self.eval_rows <= 0:
            return []
        n = max(1, min(PROBE_ROWS, self.eval_rows))
        k = max(1, self.eval_rows // n)
        return list(range(self.R - self.eval_rows, self.R, k))[:n]


# ═════════════════════════════ WSD schedule ══════════════════════════════════
def wsd_factor(step, total):
    if step < WARM_STEPS:
        return (step + 1) / max(WARM_STEPS, 1)
    ds = int(total * (1.0 - DECAY_FRAC))
    if step < ds:
        return 1.0
    f = (total - step) / max(total - ds, 1)
    return DECAY_FLOOR + (1.0 - DECAY_FLOOR) * max(f, 0.0)


def wsd_milestones(total):
    ds = int(total * (1.0 - DECAY_FRAC))
    span = max(total - ds, 1)
    return sorted({min(ds + span // 3, total), min(ds + 2 * span // 3, total)})


# ═════════════════════════ memory-safe causal-LM loss ════════════════════════
def _peft_core(model):
    core = model
    if hasattr(core, "base_model") and hasattr(core.base_model, "model"):
        core = core.base_model.model
    return core


def _decoder_and_head(model):
    core = _peft_core(model)
    dec = getattr(core, "model", None)
    if dec is None:
        try:
            dec = core.get_decoder()
        except Exception:
            dec = None
    head = None
    try:
        head = core.get_output_embeddings()
    except Exception:
        head = getattr(core, "lm_head", None)
    if dec is None or head is None:
        raise RuntimeError("cannot locate decoder/lm_head on this model")
    return dec, head


def lm_loss(model, input_ids, labels):
    """mean next-token CE without ever materialising [B,T,V] logits."""
    dec, head = _decoder_and_head(model)
    out = dec(input_ids=input_ids, use_cache=False)
    h = out.last_hidden_state
    del out
    T = h.shape[1]
    tot, n = None, 0
    use_ckpt = torch.is_grad_enabled()
    for t0 in range(0, T - 1, LOSS_CHUNK):
        t1 = min(t0 + LOSS_CHUNK, T - 1)
        hs = h[:, t0:t1, :]
        lab = labels[:, t0 + 1:t1 + 1]

        def _ce(hs_, lab_):
            logits = head(hs_).float()
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), lab_.reshape(-1),
                reduction="sum")

        if use_ckpt:
            ce = torch.utils.checkpoint.checkpoint(
                _ce, hs, lab, use_reentrant=False)
        else:
            ce = _ce(hs, lab)
        tot = ce if tot is None else tot + ce
        n += int(lab.numel())
    return tot / float(max(n, 1))


def probe(model, stream):
    rows = stream.probe_rows()
    if not rows:
        log("⚠ probe: no held-out rows")
        return 0.0
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        pbq = bar("probe", len(rows), unit="row")
        try:
            for r in rows:
                pbq.update(1)
                try:
                    x = stream.row(r).unsqueeze(0).to(DEV)
                except Exception as e:
                    log(f"⚠ probe row {r} skipped ({str(e)[:60]})")
                    continue
                loss = lm_loss(model, x, x)
                tot += float(loss.item())
                cnt += 1
                del loss
        finally:
            pbq.close()
    model.train()
    if cnt < len(rows):
        log(f"⚠ probe used only {cnt} of {len(rows)} rows")
    return round(tot / max(cnt, 1), 4)


# ══════════════════════════════ checkpoints ══════════════════════════════════
def _cpu(o):
    if torch.is_tensor(o):
        return o.detach().to("cpu").clone()
    if isinstance(o, dict):
        return {k: _cpu(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_cpu(v) for v in o)
    return o


def lora_state(model):
    from peft import get_peft_model_state_dict
    return _cpu(get_peft_model_state_dict(model))


def set_lora(model, sd):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, sd)


def save_ckpt(model, optim, tree, stream, cfg_keys, final=False):
    if not IS_MASTER:
        TRAIN["last_ckpt"] = time.time()
        return True
    step = TRAIN["step"]
    d = TMP_ROOT / f"step_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
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
    if ARGS.pilot:
        TRAIN["last_ckpt"] = time.time()
        log(f"pilot ckpt saved locally at {d} (no upload)")
        return True
    ok = bank_dir(d, f"{CKPT_SUB}/step_{step:06d}", tree, force=final)
    if ok:
        pt = TMP_ROOT / "latest.txt"
        pt.write_text(f"step_{step:06d}\n")
        bank(pt, f"{CKPT_SUB}/latest.txt", tree, desc="⬆ pointer", force=True)
        for p in d.iterdir():
            p.unlink()
        d.rmdir()
        log(f"ckpt ✔ step {step:,} → {CKPT_SUB}/step_{step:06d} banked")
    else:
        log(f"⚠ ckpt step {step:,} kept local at {d} — rerun uploads it")
    TRAIN["last_ckpt"] = time.time()
    return ok


def _step_sort(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return -1


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
    cfg = meta.get("cfg", {})
    if cfg.get("shards") and cfg["shards"] != [n for n, _ in stream.shards]:
        log("⚠ ckpt shard list differs from stream — continuing anyway")
    if cfg.get("rank") and cfg.get("rank") != ARGS.rank:
        log(f"⚠ rank differs (ckpt {cfg.get('rank')} vs arg {ARGS.rank})")
    if cfg.get("lr") and cfg["lr"] != getattr(ARGS, "lr", None):
        log(f"⚠ lr differs (ckpt {cfg['lr']} vs now {ARGS.lr})")
    if cfg.get("max_steps_planned") and cfg["max_steps_planned"] != MAX_STEPS:
        log(f"⚠ max_steps_planned differs (ckpt {cfg['max_steps_planned']} "
            f"vs now {MAX_STEPS}) — schedule mismatch")
    try:
        ad = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/adapter.pt",
                                  repo_type="dataset"))
        set_lora(model, torch.load(ad, map_location="cpu", weights_only=False))
        log("resume: adapter loaded ✔")
    except Exception as e:
        log(f"⚠ adapter load failed ({str(e)[:70]}) — refusing to restart")
        return False
    try:
        op = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/optim.pt",
                                  repo_type="dataset"))
        optim.load_state_dict(torch.load(op, map_location="cpu",
                                         weights_only=False))
        for s in optim.state.values():
            for k, v in list(s.items()):
                if torch.is_tensor(v):
                    s[k] = v.to(DEV)
        log("resume: optimizer state loaded ✔")
    except Exception as e:
        log(f"⚠ optimizer state skipped ({str(e)[:70]}) — short re-warm")
    TRAIN["step"] = int(meta["step"])
    rps = int(cfg.get("rows_per_step") or ROWS_PER_STEP)
    TRAIN["rows"] = int(meta.get("rows", TRAIN["step"] * rps))
    if rps != ROWS_PER_STEP:
        TRAIN["rows"] = TRAIN["step"] * ROWS_PER_STEP
        log(f"⚠ rows/step changed ({rps} → {ROWS_PER_STEP}) — re-anchored")
    TRAIN["loss"] = meta.get("loss")
    TRAIN["probe"] = list(meta.get("probe", []))
    TRAIN["started"] = meta.get("started", time.time())
    TRAIN["last_ckpt"] = time.time()
    for _ in range(TRAIN["step"]):
        sched.step()
    last = TRAIN["probe"][-1][1] if TRAIN["probe"] else "—"
    log(f"resume ✔ at step {TRAIN['step']:,} · rows {TRAIN['rows']:,} · "
        f"last probe {last}")
    return True


# ══════════════════════════════ model ════════════════════════════════════════
def _prep_sdpa():
    if TPU or not torch.cuda.is_available():
        return
    try:
        cap = torch.cuda.get_device_capability(DEV)
        if cap[0] >= 8:
            return
        from transformers.integrations import sdpa_attention as sa
        sa.use_gqa_in_sdpa = lambda mask, key, value: False
        log(f"sdpa: sm{cap[0]}{cap[1]} has no fused GQA kernel — "
            "repeating KV heads (avoids math fallback)")
    except Exception as e:
        log(f"⚠ sdpa GQA patch skipped ({str(e)[:60]})")


def _prep_peft():
    try:
        import peft.import_utils as iu
        try:
            iu.is_torchao_available()
        except Exception:
            iu.is_torchao_available = lambda: False
            try:
                import peft.tuners.lora.torchao as tao
                tao.is_torchao_available = lambda: False
            except Exception:
                pass
            log("peft: disabled incompatible torchao backend")
    except Exception:
        pass


def _check_wrapped(model, n_linear):
    wrapped = [n for n, _ in model.named_modules()
               if "lora_A" in n or "lora_B" in n]
    if not wrapped:
        log(f"❌ peft wrapped 0 modules ({n_linear} Linear found) — "
            "target names did not match this model")
        raise SystemExit(1)
    log(f"peft: {len(wrapped)} lora matrices on {n_linear} Linear layers")


def rewire_lora_trainable(model):
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    n_tr = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    if n_tr == 0:
        log("❌ 0 trainable params after rewire")
        raise SystemExit(1)
    log(f"requires_grad rewired · trainable {n_tr/1e6:,.0f}M (lora-only)")


def _log_init_hash(model):
    if WORLD <= 1:
        return
    q = next(p for n, p in model.named_parameters() if "lora_A" in n)
    raw = q.detach().to("cpu", torch.float32).numpy().tobytes()
    print(f"[rank {RANK}] init sha1 {hashlib.sha1(raw).hexdigest()[:12]}"
          f" ({q.numel():,} floats)", flush=True)


def build_model(rank):
    _prep_peft()
    _prep_sdpa()
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                     target_modules=list(TARGETS), bias="none",
                     task_type="CAUSAL_LM")
    if TPU:
        log(f"model: {BASE} · bf16 · attn {ARGS.attn} → {DEV} …")
        model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, attn_implementation=ARGS.attn)
        model.config.use_cache = False
        n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        torch.manual_seed(SEED)
        model = get_peft_model(model, cfg)
        _check_wrapped(model, n_lin)
        model = model.to(DEV)
        rewire_lora_trainable(model)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = p.data.to(torch.bfloat16)
        _log_init_hash(model)
    else:
        log(f"model: {BASE} · QLoRA nf4 · attn {ARGS.attn} → {DEV} …")
        ensure_libs(["bitsandbytes"], quiet=True)
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=bnb, device_map={"": RANK},
            attn_implementation=ARGS.attn)
        model.config.use_cache = False
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False)
        n_lin = sum(1 for m in model.modules()
                    if isinstance(m, torch.nn.Linear)
                    or m.__class__.__name__.endswith("4bit"))
        torch.manual_seed(SEED)
        model = get_peft_model(model, cfg)
        _check_wrapped(model, n_lin)
        rewire_lora_trainable(model)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = p.data.to(torch.float16)
        _log_init_hash(model)
    if not ARGS.no_ckpt:
        try:
            model.enable_input_require_grads()
        except Exception:
            try:
                model.base_model.model.enable_input_require_grads()
            except Exception as e:
                log(f"⚠ input-grad hook unavailable ({str(e)[:60]})")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"model ready · trainable {n_tr/1e6:,.0f}M / {n_all/1e9:.1f}B "
        f"({100*n_tr/max(n_all,1):.1f}% · LoRA r={rank}) · "
        f"gradient_checkpointing="
        f"{getattr(model, 'is_gradient_checkpointing', False)}")
    return model


# ═══════════════════════════ optimizer + sync ════════════════════════════════
def build_optim(model):
    ps = [p for p in model.parameters() if p.requires_grad]
    if not ps:
        log("❌ optimizer got 0 trainable params")
        raise SystemExit(1)
    opt = torch.optim.AdamW(ps, lr=LR_PEAK, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: wsd_factor(st, MAX_STEPS))
    return opt, sched


def sync_grads(model):
    ps = [p for p in model.parameters() if p.requires_grad]
    if not TPU and WORLD > 1:
        import torch.distributed as dist
        for p in ps:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(WORLD)
    torch.nn.utils.clip_grad_norm_(ps, 1.0)


# ═════════════════════════════ soup + ship ═══════════════════════════════════
SNAP = {}


def snapshot(model, key):
    if not IS_MASTER:
        return
    SNAP[key] = lora_state(model)
    log(f"soup: snapshot '{key}' captured ({len(SNAP)} held)")


def soup_state():
    if len(SNAP) < 2:
        return None
    ref = SNAP[list(SNAP)[0]]
    avg = {k: torch.zeros_like(v, dtype=torch.float32)
           for k, v in ref.items()}
    for s in SNAP.values():
        for k, v in s.items():
            avg[k] += v.float()
    n = len(SNAP)
    return {k: (avg[k] / n).to(ref[k].dtype) for k in ref}


def try_soup_ship(model, stream):
    if not IS_MASTER:
        return
    SNAP["fin"] = lora_state(model)
    if len(SNAP) < 2:
        SNAP.pop("fin", None)
        log("soup: fewer than 2 decay snapshots — skip")
        return
    final_probe = probe(model, stream)
    TRAIN["probe"].append([TRAIN["step"], final_probe])
    log(f"soup: final probe {final_probe}")
    cur = lora_state(model)
    set_lora(model, soup_state())
    soup_probe = probe(model, stream)
    if soup_probe <= final_probe:
        log(f"soup: soup wins ({soup_probe} ≤ final {final_probe})")
        TRAIN["probe"].append([TRAIN["step"], soup_probe])
    else:
        log(f"soup: final wins ({final_probe} < soup {soup_probe})")
        set_lora(model, cur)
        TRAIN["probe"].append([TRAIN["step"], final_probe])


def ship(model, optim, stream, tree, cfg_keys):
    if not IS_MASTER:
        return
    if ARGS.pilot:
        return
    out = TMP_ROOT / "cpt_v1"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    if not TPU:
        log("GPU/QLoRA: 4-bit base cannot merge → adapter already banked")
        return
    readme = (f"# cpt.v1 · code_rag lineage\n\n"
              f"- parent: {BASE} · method: LoRA r={ARGS.rank} "
              f"(q/k/v/o/gate/up/down)\n"
              f"- data: {len(stream.shards)} shards · {stream.R:,} blocks × "
              f"{BLOCK} tok · probe "
              f"{TRAIN['probe'][-1][1] if TRAIN['probe'] else '—'}\n"
              f"- schedule: WSD peak {LR_PEAK:g} · decay {DECAY_FRAC:.2f} → "
              f"{DECAY_FLOOR} · final step {TRAIN['step']:,}\n")
    if free_gb(TMP_ROOT) < 12:
        log(f"⚠ only {free_gb(TMP_ROOT):.0f}GB free — need ~12GB to merge")
        return
    (out / "README.md").write_text(readme)
    try:
        log("merge: moving model to CPU first …")
        model.to("cpu")
        gc.collect()
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True,
                               max_shard_size="3GB")
        AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
        log(f"ship: banking merged bf16 → {SHIP_SUB}/ (~8GB) …")
        if bank_dir(out, SHIP_SUB, tree, desc="⬆ model"):
            log(f"ship ✔ {SHIP_SUB}/ complete — cpt.v1 ready for Phase 5")
    except Exception as e:
        log(f"⚠ merged ship failed ({str(e)[:140]}) — adapter is banked")
    finally:
        shutil.rmtree(out, ignore_errors=True)
        gc.collect()


# ═════════════════════════════ training drive ════════════════════════════════
class _EpochDS(torch.utils.data.IterableDataset):
    def __init__(self, stream, base, n_steps, skip_steps, accum, world,
                 rank, lb):
        self.p = (stream, base, n_steps, skip_steps, accum, world, rank, lb)

    def __iter__(self):
        return self.p[0].iter_batch(*self.p[1:])


def _fingerprint(sd):
    acc = 0
    for k in sorted(sd):
        v = sd[k]
        if v.dtype.is_floating_point and v.element_size() == 2:
            iv = v.contiguous().view(torch.int16).to(torch.int64)
        else:
            iv = v.contiguous().view(torch.int32).to(torch.int64)
        acc = (acc + int(iv.sum())) & 0xFFFFFFFFFFFFFFFF
    return f"{acc:016x}"


def train_session(model, optim, sched, stream, pstream, tree, cfg_keys):
    step, rows_done = TRAIN["step"], TRAIN["rows"]
    per_epoch = stream.R - stream.eval_rows
    steps_epoch = max(0, per_epoch // ROWS_PER_STEP)
    total = MAX_STEPS
    log(f"plan: {total:,} steps · {steps_epoch:,} steps/epoch · "
        f"{EPOCHS} epochs · {ROWS_PER_STEP} seqs/step "
        f"({ROWS_TOKENS/1000:,.0f}k tok/step)")
    log(f"      world {WORLD} × micro {LOADER_B} × accum {ACCUM} · "
        f"usable {steps_epoch*ROWS_PER_STEP:,}/{per_epoch:,} rows/epoch · "
        f"WSD peak {LR_PEAK:g}")
    if step > 0:
        log(f"      continuing at step {step:,} "
            f"({100*step/max(total,1):.0f}%)")
    milestones = set(wsd_milestones(total))
    pb = bar("train", max(0, total - step), unit="step")
    micro = 0
    run_loss, nrun = None, 0
    for ep in range(EPOCHS):
        if step >= total:
            break
        consumed = max(0, (rows_done - ep * steps_epoch * ROWS_PER_STEP)
                       // ROWS_PER_STEP)
        if consumed >= steps_epoch:
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
                broke = True
                break
            x, y = batch["input_ids"], batch["labels"]
            if not TPU:
                x = x.to(DEV, non_blocking=True)
                y = y.to(DEV, non_blocking=True)
            loss = lm_loss(model, x, y)
            (loss / ACCUM).backward()
            run_loss = loss.detach() if run_loss is None \
                else run_loss + loss.detach()
            nrun += 1
            if nrun == 1:
                tr = [p for p in model.parameters() if p.requires_grad]
                ng = sum(1 for p in tr if p.grad is not None)
                if tr and ng == 0:
                    log("❌ no LoRA gradients after the first backward")
                    raise SystemExit(1)
                log(f"grad check ✔ {ng} LoRA tensors received grads")
            micro += 1
            if micro == ACCUM:
                if TPU:
                    XM.optimizer_step(optim, barrier=True)
                else:
                    sync_grads(model)
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                micro = 0
                step += 1
                rows_done += ROWS_PER_STEP
                TRAIN["step"], TRAIN["rows"] = step, rows_done
                pb.update(1)
                if step % LOG_EVERY == 0:
                    ema = float(run_loss.item()) / max(nrun, 1)
                    tok_s = ROWS_TOKENS * LOG_EVERY / max(
                        time.time() - TRAIN["clock"], 1e-9)
                    TRAIN["loss"] = round(ema, 4)
                    TRAIN["tok_s"] = tok_s
                    pb.set_postfix(loss=f"{ema:.3f}",
                                   lr=f"{sched.get_last_lr()[0]:0.2e}",
                                   tok_s=f"{tok_s/1000:.1f}k")
                    TRAIN["clock"] = time.time()
                    run_loss, nrun = None, 0
                if step % PROBE_EVERY == 0:
                    v = probe(model, pstream)
                    TRAIN["probe"].append([step, v])
                    log(f"probe @ {step:,}: loss {v} (history "
                        f"{[p[1] for p in TRAIN['probe'][-4:]]})")
                    save_ckpt(model, optim, tree, stream, cfg_keys)
                    pb.close()
                    pb = bar("train", total - step, unit="step")
                if step in milestones and not ARGS.pilot:
                    snapshot(model, f"m{len(SNAP)}")
                if time.time() - TRAIN["last_ckpt"] > CKPT_MIN * 60:
                    save_ckpt(model, optim, tree, stream, cfg_keys)
                    pb.close()
                    pb = bar("train", total - step, unit="step")
        del dl, ds, inner
        gc.collect()
        if not broke:
            stream.close()
    TRAIN["step"], TRAIN["rows"] = step, rows_done
    pb.close()
    return step >= total


def pilot_selftest(model, optim, stream, cfg_keys):
    if not IS_MASTER:
        return
    save_ckpt(model, optim, {}, stream, cfg_keys)
    d = TMP_ROOT / f"step_{TRAIN['step']:06d}"
    try:
        cur = lora_state(model)
        zero = {k: torch.zeros_like(v) for k, v in cur.items()}
        set_lora(model, zero)
        mid = lora_state(model)
        setter_ok = all(
            torch.allclose(mid[k].float(), torch.zeros_like(mid[k].float()),
                           rtol=0, atol=0)
            for k in mid)
        set_lora(model, torch.load(d / "adapter.pt", map_location="cpu",
                                   weights_only=False))
        re = lora_state(model)
        same = set(cur) == set(re) and all(
            torch.allclose(cur[k].float(), re[k].float(), rtol=0, atol=0,
                           equal_nan=True) for k in cur)
        n_tensors = len(torch.load(d / "optim.pt", map_location="cpu",
                                   weights_only=False).get("state", {}))
        log("selftest: adapter setter " + ("✅" if setter_ok else "❌"))
        log("selftest: adapter save→load " + ("✅ equal" if same
                                              else "❌ DIFFERS"))
        log(f"selftest: optimizer round-trip ok ({n_tensors} tensors)")
        if not same or not setter_ok:
            log("selftest ❌ — resume would be corrupt")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ════════════════════════════════ summary ════════════════════════════════════
def summary(step, total, done):
    log("══════════ CPT TRAIN SUMMARY ══════════")
    h = (time.time() - TRAIN["started"]) / 3600
    log(f"   steps {step:,}/{total:,} · rows {TRAIN['rows']:,} · "
        f"tokens {TRAIN['rows']*BLOCK:,.0f}")
    log(f"   losses: ema {TRAIN['loss']} · probes "
        f"{[p[1] for p in TRAIN['probe'][-5:]]}")
    log(f"   elapsed {h:.1f}h (session {time.time()-T0:.0f}s)")
    if done:
        log("✔ CPT done — lineage: Qwen3-4B → cpt.v1")
        log("next → data/rag_build.py (Phase 4) then Phase 5 SFT")
    else:
        hrs = (total - step) * ROWS_TOKENS / (
            (TRAIN.get("tok_s") or EST_TOK_S)) / 3600
        log(f"⚠ NOT finished · est remaining ≈ {hrs:.1f}h")
        log("  → rerun the SAME script on a fresh session (auto-resume)")


# ════════════════════════════════ driver ═════════════════════════════════════
def _run(args):
    global ARGS, LR_PEAK, EPOCHS, MAX_STEPS, WARM_STEPS, LOADER_B, ACCUM, \
        ROWS_PER_STEP, ROWS_TOKENS, PROBE_ROWS, TRAIN
    ARGS, LR_PEAK = args, args.lr
    PROBE_ROWS = args.probe_rows or (4 if args.pilot else PROBE_ROWS)
    TRAIN = dict(step=0, rows=0, loss=None, probe=[], started=time.time(),
                 last_ckpt=time.time(), clock=time.time(), tok_s=None)
    pick_root()
    libs_report()
    if TPU:
        log(f"backend: TPU · world {WORLD} · rank {RANK} · bf16")
    else:
        dp = f" · world {WORLD} (NCCL)" if WORLD > 1 else ""
        log(f"backend: CUDA · {torch.cuda.get_device_name(DEV)} · QLoRA{dp}")

    manifest, tree, shards_all, blk = load_manifest()
    peek = None if args.no_resume else resume_peek(tree)
    shards = shards_all
    if peek:
        frozen = (peek[1].get("cfg") or {}).get("shards") or []
        if frozen:
            missing = [n for n in frozen if n not in blk]
            if missing:
                log(f"❌ checkpoint {peek[0]} froze {len(missing)} shard(s) "
                    "missing from the manifest")
                log("   → use --no-resume for a fresh lineage")
                raise SystemExit(1)
            if frozen != [n for n, _ in shards_all]:
                log(f"data: freezing to checkpoint shard list "
                    f"({len(frozen)} shards)")
            shards = [(n, blk[n]) for n in frozen]

    stream = RowStream(shards)
    pstream = RowStream(shards)
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
        log(f"❌ stream too small ({per_epoch:,} rows → 0 steps)")
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
        log(f"⚠ global batch {ROWS_PER_STEP} ≠ target {G_BATCH}")

    if TPU and RANK:
        time.sleep(min(RANK * 8, 60))
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
            raise SystemExit(1)
        if not loaded:
            log("❌ checkpoint exists but its adapter did not load")
            log("   → rerun (transient?) or use --no-resume for a fresh lineage")
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
        if args.pilot:
            if WORLD > 1:
                print(f"[rank {RANK}] adapter fingerprint "
                      f"{_fingerprint(lora_state(model))}", flush=True)
            pilot_selftest(model, optim, stream, cfg_keys)
        if done and not args.pilot:
            try:
                try_soup_ship(model, pstream)
            except Exception as e:
                log(f"soup skipped ({str(e)[:80]})")
            save_ckpt(model, optim, tree, stream, cfg_keys, final=True)
            ship(model, optim, stream, tree, cfg_keys)
        elif not args.pilot and TRAIN["step"] > 0:
            save_ckpt(model, optim, tree, stream, cfg_keys)
    except KeyboardInterrupt:
        log("⚠ interrupted — saving a checkpoint first …")
        try:
            save_ckpt(model, optim, tree, stream, cfg_keys)
        except Exception:
            pass
        raise
    except Exception as e:
        log(f"❌ train loop failed: {type(e).__name__}: {str(e)[:200]}")
        low = str(e).lower()
        if "out of memory" in low or "resource exhausted" in low:
            log("   → OOM: keep --micro-batch 1 · lower --rank 32 · "
                "close other GPU notebooks")
        try:
            save_ckpt(model, optim, tree, stream, cfg_keys)
        except Exception:
            pass
        raise
    finally:
        stream.close()
        pstream.close()
    summary(TRAIN["step"], MAX_STEPS, done)


# ═════════════════════════════════ main ══════════════════════════════════════
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
        dev = DEV if DEV is not None else torch.device("cuda:0")
        log(f"cuda: {torch.version.cuda} · "
            f"{torch.cuda.get_device_name(dev)} "
            f"({torch.cuda.get_device_properties(dev).total_memory/2**30:.0f}GB) "
            f"×{torch.cuda.device_count()}")


def main():
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
                    help="seqs per core per forward (default: 1)")
    ap.add_argument("--accum", type=int, default=None,
                    help="micro-batches per optimizer step (auto → 64 seqs)")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing (escape hatch)")
    ap.add_argument("--gpus", type=int, default=0,
                    help="CUDA workers (0 = all visible GPUs, 1 = single)")
    ap.add_argument("--probe-rows", type=int, default=None,
                    help="held-out rows per probe (default: 32 · 4 in pilot)")
    ap.add_argument("--single-core", action="store_true",
                    help="TPU: 1 core in-process (fallback if xmp.spawn fails)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints; restart the lineage")
    args = ap.parse_args()
    ARGS, LR_PEAK = args, args.lr
    log(f"════════════ CPT TRAIN · seed {SEED} · block {BLOCK} · "
        f"pilot={args.pilot or 'off'} ════════════")

    tok = get_token()
    if tok:
        os.environ["HF_TOKEN"] = tok

    if args.libs:
        libs_report(extra=["torch_xla"])
        return

    tpu = (args.backend == "tpu") or \
          (args.backend == "auto" and _looks_like_tpu_env())
    if tpu:
        _ensure_torch_xla()
        if args.single_core:
            import torch_xla.core.xla_model as xm
            TPU, XM, DEV = True, xm, xm.xla_device()
            WORLD, RANK, IS_MASTER = 1, 0, True
            log("TPU single-core mode (pilots only)")
            _run(args)
            return
        if _relaunch_if_notebook():
            return
        for k in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
            os.environ.pop(k, None)
        try:
            from huggingface_hub import snapshot_download
            log(f"model: pre-downloading {BASE} to the shared cache …")
            snapshot_download(BASE)
        except Exception as e:
            log(f"⚠ pre-download skipped ({str(e)[:80]})")
        predownload_data()
        import torch_xla.distributed.xla_multiprocessing as xmp
        log("TPU: spawning one process per core (data-parallel) …")
        try:
            xmp.spawn(_train_worker, args=(vars(args),), nprocs=None)
        except Exception as e:
            log(f"❌ TPU launch failed: {type(e).__name__}: {str(e)[:180]}")
            log("   → retry with --single-core (slow) or --backend gpu")
            raise SystemExit(1)
        return

    if not torch.cuda.is_available():
        log("❌ no TPU detected and no CUDA GPU — choose a GPU/TPU accelerator")
        raise SystemExit(1)
    n_gpu = torch.cuda.device_count()
    want = n_gpu if args.gpus <= 0 else min(args.gpus, n_gpu)
    if want > 1:
        if _relaunch_if_notebook():
            return
        try:
            from huggingface_hub import snapshot_download
            log(f"model: pre-downloading {BASE} to the shared cache …")
            snapshot_download(BASE)
        except Exception as e:
            log(f"⚠ pre-download skipped ({str(e)[:80]})")
        predownload_data()
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()
        import torch.multiprocessing as mp
        log(f"CUDA: spawning {want} workers (data-parallel · NCCL "
            f":{os.environ['MASTER_PORT']}) …")
        try:
            mp.spawn(_gpu_worker, args=(want, vars(args)), nprocs=want)
        except Exception as e:
            log(f"❌ multi-GPU launch failed: {type(e).__name__}: "
                f"{str(e)[:180]}")
            log("   → retry with --gpus 1 (single GPU)")
            raise SystemExit(1)
        return
    TPU, DEV, XM = False, torch.device("cuda:0"), None
    WORLD, RANK, IS_MASTER = 1, 0, True
    log(f"backend: CUDA · {torch.cuda.get_device_name(0)} · QLoRA")
    log("⚠ T4 path is for pilots/debug — the free TPU is ~15-30× faster")
    _run(args)


if __name__ == "__main__":
    main()
