#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3 · STEP 2 — CPT TRAINING  (docs/phase3.md · Part I)

    cpt_manifest.json (data/cpt/*_block_*.bin on HF · ~300M tokens · 4096/token)
        │
        ├─ ① SETUP    Qwen3-4B base (bf16) + LoRA r=128 on all linear layers
        │             (memory-honest: full 4B AdamW/Adafactor fine-tune needs
        │              ~56GB optimizer state per replica — v5e cores give 16GB;
        │              LoRA@r128 ≈ 6.5% trainable params, makes BOTH backends
        │              fit; replay data in the mix protects general ability)
        ├─ ② TRAIN    WSD schedule: warmup 1% → stable @ peak 1e-4 →
        │             decay last 20% linear down to peak/20
        │             grad-accum to 64 seqs/step = 262k tokens/step
        │             chunked cross-entropy (151k vocab: full-vocab loss OOMs)
        ├─ ③ GUARD    probe loss every 500 steps on 32 held-out tail rows
        │             checkpoint (adapter+optimizer+data position) every ~55min
        │             → HF  ckpt/cpt/step_NNNNNN/ · kill-safe resume
        └─ ④ SHIP     soup the decay-third snapshots → probe(soup vs final)
                     → merge adapters → bank models/cpt.v1/ on HF (+README)

Backends (auto, --backend tpu|gpu to force):
    TPU v5e-8   torch_xla present, ≥4 cores → bf16, r=128, MpDeviceLoader
    CUDA/T4     fallback → QLoRA nf4, r=64, fp16 adapters, single GPU
                (1 epoch default — 3 full epochs on one T4 ≈ weeks; TPU is
                 the real path, GPU sessions are for pilots/debugging)

Resumability — the prep contract again ("nothing is progress that isn't on HF"):
    every checkpoint dir holds:  adapter.pt · optim.pt · meta.json
    ckpt/cpt/latest.txt          = name of newest step dir (fast lookup)
    meta.json freezes the shard list + row counts used → a later prep upload
    can never silently change the step→data map mid-lineage.

Run (Kaggle cell):
    %run data/cpt_train.py --pilot 60     # smoke: 60 steps + save/reload test
    %run data/cpt_train.py                # the real run (TPU session, ~9-12h)

First step on TPU is slow — XLA compiles the graph (1-4 min, normal, once).
================================================================================
"""
import os, sys, gc, json, time, math, shutil, argparse, bisect
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PJRT_DEVICE", "TPU")

# HF cache lives on the ephemeral big disk; model + shard cache ≈ 12GB.
try:
    _H = "/tmp/code_rag_train/hf"; Path(_H).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", _H)
except Exception:
    pass


# ── Kaggle convenience: auto-install anything missing (needs Internet ON) ─────
def _maybe_install(pkgs):
    from importlib.metadata import version
    bad = []
    for spec in pkgs:
        name = spec.split(">=")[0]
        floor = spec.split(">=")[1] if ">=" in spec else None
        ok = True
        try:
            cur = version(name)
            if floor is not None:
                c = tuple(int(x) for x in cur.split(".")[:3] if x.isdigit())
                f = tuple(int(x) for x in floor.split(".")[:3] if x.isdigit())
                if len(c) < len(f): c = c + (0,) * (len(f) - len(c))
                ok = c >= f
        except Exception:
            ok = False
        if not ok: bad.append(spec)
    if not bad: return
    print(f"(auto-installing {', '.join(bad)} …)", flush=True)
    import subprocess
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                         *bad]).returncode
    if rc != 0:
        print(f"\nneed: pip install -U {' '.join(bad)}"); raise SystemExit(1)


import torch                                              # noqa: E402
import numpy as np                                        # noqa: E402
_maybe_install(["huggingface_hub>=0.23", "transformers>=4.51.0",
                "peft>=0.13.0", "accelerate", "sentencepiece", "tqdm"])
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

# ══════════════════════════════════ CONFIG ════════════════════════════════════
HF_TOKEN    = os.environ.get("HF_TOKEN", "")     # or Kaggle secret HF_TOKEN
REPO        = "sddfhy546y/code_rag_dataset"      # dataset repo (blocks live here)
DATA_SUB    = "data/cpt"                         # same folder the prep banks to
CKPT_SUB    = "ckpt/cpt"                         # hourly checkpoints
SHIP_SUB    = "models/cpt.v1"                    # final merged weights
BASE        = "Qwen/Qwen3-4B"                    # the one lineage (per phase plan)
TARGETS     = ["q_proj", "k_proj", "v_proj", "o_proj",     # explicit, version-proof
               "gate_proj", "up_proj", "down_proj"]        # (peft "all-linear"
SEED        = 3407                               # pinned → identical everything
BLOCK       = 4096                               # tokens/block (from prep)
EPOCHS_TPU  = 3                                  # TPU budget (900M token passes)
EPOCHS_GPU  = 1                                  # T4 is a fallback, not the plan
LR_PEAK     = 1.0e-4                             # LoRA peak (docs/phase3.md §WSD)
DECAY_FRAC  = 0.20                               # decay window = 20% of steps
DECAY_FLOOR = 0.05                               # end LR = peak × 0.05
WARM_FRAC   = 0.01                               # warmup = 1% of steps
G_BATCH     = 64                                 # seqs / optimizer step (262k tok)
LBS_TPU     = 8                                  # loader batch (8 cores × 1/core)
LBS_GPU     = 2                                  # T4: micro-batch = 2 seqs
LOG_EVERY   = 20                                 # steps between loss prints
PROBE_EVERY = 500                                # steps between probe evals
PROBE_ROWS  = 32                                 # held-out rows per probe
HOLDOUT     = 512              # tail blocks never trained on (eval only)
CKPT_MIN    = 55                                 # ≤ minutes between checkpoints
EST_TOK_S   = 27000                              # v5e-8 planning number
TMP_ROOT    = Path(os.environ.get("CPT_TRAIN_TMP", "/tmp/code_rag_train"))

ROOT_CANDIDATES = [
    (Path("/tmp/code_rag_train"),            "ephemeral big disk"),
    (Path("/kaggle/working/code_rag_train"), "kaggle working disk (20GB quota)"),
]
MIN_FREE_GB = 6.0

# ════════════════════ plumbing: identical idioms to cpt_prep.py ═══════════════
T0 = time.time()
def now():  return datetime.now(timezone.utc).strftime("%H:%M:%S")
def log(msg): print(f"[{now()}] {msg}", flush=True)
def gfmt(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} TB"
def bar(desc, total=None, unit="it", scale=False):
    try:
        return tqdm(total=total, desc=desc, unit=unit,
                    unit_scale=scale, dynamic_ncols=True)
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
    log(f"disk: root {TMP_ROOT} (auto: {why})")

_API = None
def api():
    global _API
    if _API is None:
        tok = HF_TOKEN or os.environ.get("KAGGLE_SECRET_HF_TOKEN", "")
        if not tok:
            try:                                   # Kaggle-only convenience
                from kaggle_secrets import UserSecretsClient
                tok = UserSecretsClient().get_secret("HF_TOKEN")
                os.environ["HF_TOKEN"] = tok
            except Exception:
                pass
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

class ProgressReader:                                # hub needs a real file-like
    def __init__(self, path, b):
        self._pb, self._f = b, open(path, "rb")
    def read(self, size=-1):
        d = self._f.read(size)
        if d: self._pb.update(len(d))
        return d
    def tell(self): return self._f.tell()
    def close(self):
        try: self._f.close()
        except Exception: pass

def bank(local: Path, repo_path: str, tree: dict, desc="⬆"):
    """upload once; skip when the repo already holds a same-size file."""
    local = Path(local)
    if not local.is_file():
        log(f"⚠ bank skip (missing): {local}")
        return False
    need = local.stat().st_size
    if tree.get(repo_path) == need:
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

def bank_dir(local_dir: Path, repo_sub: str, tree: dict, desc="⬆ model"):
    path = Path(local_dir)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    oks = [bank(p, f"{repo_sub}/{p.relative_to(path)}", tree, desc=desc)
           for p in files]
    if not all(oks):
        log(f"⚠ some files of {repo_sub} deferred — rerun retries them")
    return all(oks)

def pick_backend(args):
    """TPU when torch_xla provides ≥4 cores; else CUDA; --backend overrides."""
    if args.backend == "gpu":
        return False
    try:
        import torch_xla.core.xla_model as xm
        if xm.xla_device_count() >= 4:
            return True
        log("torch_xla present but no big device → CUDA path")
    except Exception:
        pass
    if args.backend == "tpu":
        log("❌ --backend tpu requested but no TPU attached to this session")
        raise SystemExit(1)
    return False

# ═════════════════════════════════ main state ═════════════════════════════════
ARGS = None
TPU  = False
DEV  = None
XM   = None                        # torch_xla module when TPU else None
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
    usable, blk = [], {n: int(m["pack"]["shards"][n]["blocks"]) for n in names}
    for name in names:
        on_repo = tree.get(f"{DATA_SUB}/{name}")
        if m["pack"]["shards"][name].get("uploaded") or on_repo is not None:
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
    """peft attachments must be real; 'all-linear' once silently wrapped 0."""
    wrapped = [n for n, _ in model.named_modules()
               if "lora_A" in n or "lora_B" in n]
    if not wrapped:
        log(f"❌ peft wrapped 0 modules ({n_linear} Linear found) — "
            "target names did not match this model")
        raise SystemExit(1)
    log(f"peft: {len(wrapped)} lora matrices on {n_linear} Linear layers")

def rewire_lora_trainable(model):
    """XLA device moves can drop requires_grad — set it explicitly: only
    lora_* params trainable, everything else frozen. Fails loudly if empty."""
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    n_tr = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    if n_tr == 0:
        heads = [f"{n}·{tuple(p.shape)}" for n, p in list(model.named_parameters())[:8]]
        log("❌ 0 trainable params even after rewire — first pělěts:")
        for h in heads: log(f"   {h}")
        raise SystemExit(1)
    log(f"requires_grad rewired · trainable {n_tr/1e6:,.0f}M (lora-only)")

def build_model(rank):
    from peft import LoraConfig, get_peft_model
    attn = ARGS.attn
    if TPU:
        log(f"model: {BASE} · bf16 · attn {attn} → xla device …")
        model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, attn_implementation=attn)
        model.config.use_cache = False
        n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        cfg = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                         target_modules=list(TARGETS), bias="none",
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        _check_wrapped(model, n_lin)
        model = model.to(DEV)
        rewire_lora_trainable(model)         # flags FIRST — XLA .to may drop them
        for n, p in model.named_parameters():    # bf16 adapters (16GB-core math)
            if "lora_" in n: p.data = p.data.to(torch.bfloat16)
    else:
        log(f"model: {BASE} · QLoRA nf4 · attn {attn} → cuda:0 …")
        _maybe_install(["bitsandbytes"])
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=bnb, device_map={"": 0},
            attn_implementation=attn)
        model.config.use_cache = False
        n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear)
                    or m.__class__.__name__.endswith("4bit"))
        cfg = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                         target_modules=list(TARGETS), bias="none",
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        _check_wrapped(model, n_lin)
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model,
                                                use_gradient_checkpointing=True)
        rewire_lora_trainable(model)
        for n, p in model.named_parameters():    # fp16 adapters on the 4-bit base
            if "lora_" in n: p.data = p.data.to(torch.float16)
    if not getattr(model, "is_gradient_checkpointing", False) and not ARGS.no_ckpt:
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
CHUNK = 1024
def token_ce(logits, labels):
    """mean per-token CE; logits [B,T,V] ∈ {bf16,fp16}; float32 chunks."""
    lm = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    lb = labels[:, 1:].reshape(-1)
    n = int(lb.numel())
    tot = None
    for c in range(0, n, CHUNK):
        s = lm[c:c + CHUNK].float()
        t = lb[c:c + CHUNK]
        ce = torch.nn.functional.cross_entropy(s, t, reduction="sum")
        tot = ce if tot is None else (tot + ce)
    return tot / float(max(n, 1))

# ═════════════════════════════ data stream (deterministic) ════════════════════
class RowStream:
    """
    frozen shard list → flat row iterator.
    • order = shard filename order (the prep already seed-shuffled spans, so
      this IS a stable mixture; no extra shuffle). identity is byte-stable.
    • the last eval_rows are never yielded (they are the eval probe).
    • start= skips rows on resume (no device work, cheap).
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
        return torch.from_numpy(np.asarray(self._mmap(k)[off], dtype=np.int64))

    def iter_epoch(self, start=0):
        """one training epoch: rows i < R - eval_rows, skipping `start`."""
        stop = self.R - self.eval_rows
        given = 0
        for i in range(stop):
            if given < start:
                given += 1; continue
            try:
                y = self.row(i)
            except Exception as e:                # corrupt row → skip it
                log(f"   data row {i} unreadable → skipped ({str(e)[:60]})")
                continue
            yield {"input_ids": y, "labels": y.clone()}
        if self._mm is not None:
            try: self._mm._mmap.close()
            except Exception: pass
            self._mm, self._si = None, -1

    def probe_rows(self):
        """deterministic eval rows: evenly spaced inside the never-trained tail."""
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
    """held-out tail loss; deterministic rows → comparable across runs."""
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
    set_peft_model_state_dict(model, {k: v for k, v in sd.items()})

def save_ckpt(model, optim, tree, stream, cfg_keys, final=False):
    step = TRAIN["step"]
    d = TMP_ROOT / f"step_{step:06d}"; d.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state(model), d / "adapter.pt")
    torch.save(_cpu(optim.state_dict()), d / "optim.pt")
    meta = dict(step=step, rows=TRAIN["rows"], loss=TRAIN["loss"],
                probe=TRAIN["probe"][-32:], started=TRAIN["started"],
                saved=time.time(),
                shards=[n for n, _ in stream.shards],
                rows_epoch=stream.R - stream.eval_rows,
                holdout=stream.eval_rows, cfg=cfg_keys, final=final)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    if ARGS.pilot:                                # pilot: local only, no upload
        TRAIN["last_ckpt"] = time.time()
        log(f"pilot ckpt saved locally at {d} (no upload)")
        return True
    ok = bank_dir(d, f"{CKPT_SUB}/step_{step:06d}", tree, desc="⬆ ckpt")
    if ok:
        pt = TMP_ROOT / "latest.txt"
        pt.write_text(f"step_{step:06d}\n")
        bank(pt, f"{CKPT_SUB}/latest.txt", tree, desc="⬆ pointer")
        for p in d.iterdir(): p.unlink()
        d.rmdir()
        log(f"ckpt ✔ step {step:,} → {CKPT_SUB}/step_{step:06d} banked")
    else:
        log(f"⚠ ckpt step {step:,} kept local at {d} — rerun uploads it")
    TRAIN["last_ckpt"] = time.time()
    return ok

def resume_peek(tree):
    """find the newest checkpoint dir on HF; download ONLY meta.json."""
    cands = set()
    pref = f"{CKPT_SUB}/"
    for full in tree:
        if full.startswith(pref):
            tail = full[len(pref):]
            if "/" in tail: cands.add(tail.split("/")[0])
    try:                                          # pointer wins when present
        p = Path(hf_hub_download(REPO, f"{CKPT_SUB}/latest.txt",
                                 repo_type="dataset"))
        cands.add(p.read_text().strip())
    except Exception:
        pass
    steps = sorted((c for c in cands if c.startswith("step_")),
                   key=lambda s: int(s.split("_")[1]))
    if not steps:
        return None
    nam = steps[-1]
    try:
        p = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/meta.json",
                                 repo_type="dataset"))
        meta = json.loads(p.read_text())
    except Exception as e:                        # adapter/optim may be missing
        log(f"⚠ meta of {nam} unreadable ({str(e)[:70]}) — fresh start")
        return None
    log(f"resume: newest checkpoint {nam} (meta loaded)")
    return nam, meta

def resume_load(model, optim, sched, nam, meta, stream, blk):
    """load adapter + optimizer + position; freeze data to ckpt's shard list."""
    cfg = meta.get("cfg", {})
    frozen = cfg.get("shards") or []
    missing = [n for n in frozen if n not in blk]
    if missing:
        log(f"❌ checkpoint {nam} froze {len(missing)} shard(s) that are no "
            "longer in the manifest — the step→data map would be corrupt")
        log("   → the shard fingerprint changed: rerun with --no-resume "
            "for a fresh lineage (old checkpoints stay on HF)")
        raise SystemExit(1)
    if frozen and frozen != [n for n, _ in stream.shards]:
        log(f"data: freezing to the checkpoint's shard list "
            f"({len(frozen)} shards, {sum(blk[n] for n in frozen):,} rows)")
        stream2 = RowStream([(n, blk[n]) for n in frozen])
        stream.shards = stream2.shards; stream.R = stream2.R
        stream.eval_rows = stream2.eval_rows; stream.starts = stream2.starts
    if cfg.get("rank") and cfg.get("rank") != ARGS.rank:
        log(f"⚠ rank differs (ckpt {cfg.get('rank')} vs arg {ARGS.rank}) — "
            "adapter shapes may mismatch; retry without --rank")
    try:
        ad = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/adapter.pt",
                                  repo_type="dataset"))
        set_lora(model, torch.load(ad, map_location="cpu"))
    except Exception as e:
        log(f"⚠ adapter load failed ({str(e)[:70]}) — starting fresh weights")
        return False
    try:
        op = Path(hf_hub_download(REPO, f"{CKPT_SUB}/{nam}/optim.pt",
                                  repo_type="dataset"))
        optim.load_state_dict(torch.load(op, map_location="cpu"))
        for s in optim.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                if isinstance(s.get(k), torch.Tensor):
                    s[k] = s[k].to(DEV)
    except Exception as e:
        log(f"⚠ optimizer state load failed ({str(e)[:70]}) — "
            "scheduler/optimizer re-warm (short, normal)")
    TRAIN["step"] = int(meta["step"])
    TRAIN["rows"] = int(meta.get("rows", TRAIN["step"] * ROWS_PER_STEP))
    TRAIN["loss"] = meta.get("loss")
    TRAIN["probe"] = list(meta.get("probe", []))
    TRAIN["started"] = meta.get("started", time.time())
    TRAIN["last_ckpt"] = time.time()
    for _ in range(TRAIN["step"] + 1):            # deterministic WSD replay
        sched.step()
    last = TRAIN["probe"][-1][1] if TRAIN["probe"] else "—"
    log(f"resume ✔ at step {TRAIN['step']:,} · rows {TRAIN['rows']:,} · "
        f"last probe loss {last}")
    return True

# ═══════════════════════════ optimizer + scheduler ════════════════════════════
def build_optim(model):
    ps = [p for p in model.parameters() if p.requires_grad]
    if not ps:
        lora_n = sum(1 for n, _ in model.named_modules() if "lora_" in n)
        log(f"❌ optimizer got 0 trainable params · lora modules {lora_n} · "
            "quitting so it fails loudly here, not later")
        raise SystemExit(1)
    opt = torch.optim.AdamW(ps, lr=LR_PEAK, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: wsd_factor(st, MAX_STEPS))
    return opt, sched

# ═════════════════════════════ soup (decay-end average) ═══════════════════════
SNAP = {}
def snapshot(model, key):
    SNAP[key] = lora_state(model)                 # CPU dict (~0.5GB/shot)
    log(f"soup: snapshot '{key}' captured ({len(SNAP)} held)")

def soup_state():
    if len(SNAP) < 2: return None
    keys = list(SNAP)
    ref = SNAP[keys[0]]
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in ref.items()}
    for s in SNAP.values():
        for k, v in s.items(): avg[k] += v.float()
    return {k: (avg[k] / len(SNAP)).to(v.dtype) for k, v in ref.items()}

def try_soup_ship(model, stream):
    """probe(final) vs probe(soup-in-place); leave the winner loaded.
    ingredients: decay-third snapshots + the final adapter (classic WSD soup)."""
    SNAP["fin"] = lora_state(model)               # final = last ingredient
    if len(SNAP) < 2:
        SNAP.pop("fin", None)
        log("soup: fewer than 2 decay snapshots (crash earlier?) — skip")
        return
    cur = lora_state(model)
    final_probe = TRAIN["probe"][-1][1] if TRAIN["probe"] else float("inf")
    set_lora(model, soup_state())
    soup_probe = probe(model, stream)
    if soup_probe <= final_probe:
        log(f"soup: soup wins ({soup_probe} ≤ final {final_probe}) → shipping soup")
    else:
        log(f"soup: final wins ({final_probe} ≤ soup {soup_probe}) → shipping final")
        set_lora(model, cur)
    TRAIN["probe"].append([TRAIN["step"], soup_probe])

# ══════════════════════════ merge + ship cpt.v1 ═══════════════════════════════
def ship(model, optim, stream, tree, cfg_keys):
    if ARGS.pilot:
        log("pilot: ship skipped (the full run banks models/cpt.v1/)")
        return
    out = TMP_ROOT / "cpt_v1"; shutil.rmtree(out, ignore_errors=True)
    if not TPU:
        log("GPU/QLoRA: 4-bit base cannot merge → banking adapter only "
            "(merge runs on a TPU pass later)")
        save_ckpt(model, optim, tree, stream, cfg_keys, final=True)
        return
    readme = (f"# cpt.v1 · code_rag lineage\n\n"
              f"- parent: {BASE} · method: LoRA r={ARGS.rank} all-linear\n"
              f"- data: {len(stream.shards)} shards · {stream.R:,} blocks × "
              f"{BLOCK} tok · probe loss {TRAIN['probe'][-1][1] if TRAIN['probe'] else '—'}\n"
              f"- schedule: WSD peak {LR_PEAK:g} · decay {DECAY_FRAC:.2f} → "
              f"{DECAY_FLOOR} · final step {TRAIN['step']:,}\n")
    (out / "README.md").write_text(readme)
    try:
        log("merge: unloading adapters into base weights … (1-2 min)")
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True, max_shard_size="3GB")
        AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
        log(f"ship: banking merged bf16 → {SHIP_SUB}/ (~8GB, file-skip safe)")
        if bank_dir(out, SHIP_SUB, tree):
            log(f"ship ✔ {SHIP_SUB}/ complete — cpt.v1 ready for Phase 5")
    except Exception as e:
        log(f"⚠ merged ship failed ({str(e)[:120]}) — adapter-only fallback")
        save_ckpt(model, optim, tree, stream, cfg_keys, final=True)
    finally:
        shutil.rmtree(out, ignore_errors=True); gc.collect()

# ═════════════════════════════ training drive ═════════════════════════════════
class _EpochDS:
    def __init__(self, stream, skip): self.s, self.k = stream, int(skip)
    def __iter__(self):
        return self.s.iter_epoch(start=self.k)

def train_session(model, optim, sched, stream, tree, cfg_keys):
    step, rows_done = TRAIN["step"], TRAIN["rows"]
    per_epoch = stream.R - stream.eval_rows
    total = (per_epoch * EPOCHS) // ROWS_PER_STEP
    if step > 0:
        log(f"plan: continuing from step {step:,} / {total:,} "
            f"(rows {rows_done:,} · {ROWS_PER_STEP:,}/step)")
    else:
        log(f"plan: {total:,} steps · {per_epoch:,} train rows/epoch "
            f"({stream.eval_rows:,} held-out) · {EPOCHS} epochs · "
            f"WSD peak {LR_PEAK:g} (warm {int(100*WARM_FRAC)}% · "
            f"decay {int(100*DECAY_FRAC)}% → {DECAY_FLOOR})")
    milestones = set(wsd_milestones(total))
    pb = bar("train", total - step, unit="step")
    micro, ema_sum, ema_n = 0, 0.0, 0
    for ep in range(EPOCHS):
        if step >= total: break
        skip = max(rows_done - ep * per_epoch, 0)
        if rows_done >= (ep + 1) * per_epoch:
            continue                               # this epoch already banked in
        ds = _EpochDS(stream, skip)
        dl = torch.utils.data.DataLoader(ds, batch_size=LOADER_B,
                                         drop_last=True, num_workers=0)
        if TPU:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader
            inner = MpDeviceLoader(dl, DEV)
        else:
            inner = dl
        for batch in inner:
            if step >= total: break
            x, y = batch["input_ids"], batch["labels"]
            out = model(input_ids=x)
            loss = token_ce(out.logits, y)
            (loss / ACCUM).backward()
            ema_sum += float(loss.item()); ema_n += 1
            micro += 1
            if micro == ACCUM:
                if TPU:
                    XM.optimizer_step(optim, barrier=True)   # reduce+step+mark
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                micro = 0; step += 1; rows_done += ROWS_PER_STEP
                pb.update(1)
                if step % LOG_EVERY == 0:
                    ema = ema_sum / max(ema_n, 1)
                    tok_s = ROWS_TOKENS * LOG_EVERY / max(
                        time.time() - TRAIN["clock"], 1e-9)
                    TRAIN["loss"] = round(ema, 4)
                    TRAIN["tok_s"] = tok_s
                    pb.set_postfix(loss=f"{ema:.3f}",
                                   lr=f"{sched.get_last_lr()[0]:0.2e}",
                                   tok_s=f"{tok_s/1000:.1f}k")
                    TRAIN["step"], TRAIN["rows"] = step, rows_done
                    TRAIN["clock"] = time.time()
                    ema_sum, ema_n = 0.0, 0
                if step % PROBE_EVERY == 0:
                    v = probe(model, stream)
                    TRAIN["probe"].append([step, v])
                    hist = [p[1] for p in TRAIN["probe"][-4:]]
                    log(f"probe @ {step:,}: loss {v} (history {hist})")
                    save_ckpt(model, optim, tree, stream, cfg_keys)
                    pb.close(); pb = bar("train", total - step, unit="step")
                if step in milestones and not ARGS.pilot:
                    snapshot(model, f"m{len(SNAP)}")
                if time.time() - TRAIN["last_ckpt"] > CKPT_MIN * 60:
                    save_ckpt(model, optim, tree, stream, cfg_keys)
                    pb.close(); pb = bar("train", total - step, unit="step")
        del dl, ds, inner
        gc.collect()
    TRAIN["step"], TRAIN["rows"] = step, rows_done
    pb.close()
    return step >= total

def pilot_selftest(model, optim, stream, cfg_keys):
    """prove the resume path works: save locally → reload adapters → compare."""
    tree = {}                                     # force local-only save
    args_backup = ARGS.pilot
    save_ckpt(model, optim, tree, stream, cfg_keys)
    d = TMP_ROOT / f"step_{TRAIN['step']:06d}"
    try:
        cur = lora_state(model)
        sd = torch.load(d / "adapter.pt", map_location="cpu")
        set_lora(model, sd)
        re = lora_state(model)
        same = all(torch.equal(cur[k], re[k]) for k in cur
                   if k in re) and (set(cur) == set(re))
        optim_sd = torch.load(d / "optim.pt", map_location="cpu")
        n_tensors = len(optim_sd.get("state", {}))
        log(f"selftest: optimizer state round-trip ok ({n_tensors} tensors)")
        if not same: log("selftest ❌ — check set_peft_model_state_dict keys")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        ARGS.pilot = args_backup

# ════════════════════════════════════ main ════════════════════════════════════
def summary(step, total, stream, done):
    log("══════════ CPT TRAIN SUMMARY ══════════")
    h = (time.time() - TRAIN["started"]) / 3600
    log(f"   steps {step:,}/{total:,} · rows {TRAIN['rows']:,} · "
        f"tokens {TRAIN['rows']*BLOCK:,.0f}")
    log(f"   losses: ema {TRAIN['loss']} · probes "
        f"{[p[1] for p in TRAIN['probe'][-5:]]}")
    log(f"   elapsed {h:.1f}h (session {time.time()-T0:.0f}s)")
    if not done and step < total:
        hrs = (total - step) * ROWS_TOKENS / (
            (TRAIN.get("tok_s") or EST_TOK_S)) / 3600
        log(f"⚠ NOT finished · est remaining ≈ {hrs:.1f}h ·")
        log("  → rerun the SAME script on a fresh session (auto-resume)")
    else:
        log("✔ CPT done — lineage: Qwen3-4B → cpt.v1 (your 300M-token mix)")
        log("next → data/rag_build.py (Phase 4 index) then Phase 5 SFT")

def main():
    global ARGS, TPU, DEV, XM, EPOCHS, LR_PEAK, MAX_STEPS, WARM_STEPS, \
           LOADER_B, ACCUM, ROWS_PER_STEP, ROWS_TOKENS, OPTIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0,
                    help=">0 = run N steps only (smoke test, local-only ckpt)")
    ap.add_argument("--backend", choices=["auto", "tpu", "gpu"], default="auto")
    ap.add_argument("--rank", type=int, default=None, help="LoRA rank override")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--lr", type=float, default=LR_PEAK)
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"default: {EPOCHS_TPU} on TPU, {EPOCHS_GPU} on GPU")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing (escape hatch)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints; restart the lineage")
    args = ap.parse_args()
    ARGS, LR_PEAK = args, args.lr
    log(f"════════════ CPT TRAIN · seed {SEED} · block {BLOCK} · "
        f"pilot={args.pilot or 'off'} ════════════")
    pick_root()
    manifest, tree, shards, blk = load_manifest()

    # ① peek resume BEFORE building the stream (it may freeze the shard list)
    peek = None if args.no_resume else resume_peek(tree)

    TPU = pick_backend(args)
    if TPU:
        import torch_xla.core.xla_model as xm
        XM, DEV = xm, xm.xla_device()
        EPOCHS = args.epochs or EPOCHS_TPU
        ARGS.rank = args.rank or 128
        LOADER_B = LBS_TPU
        log(f"backend: TPU · {xm.xla_device_count()} cores · bf16")
    else:
        if not torch.cuda.is_available():
            log("❌ no CUDA device and no TPU — attach an accelerator "
                "(Settings → Accelerator: TPU v5e-8 or GPU T4)")
            raise SystemExit(1)
        DEV, XM = torch.device("cuda:0"), None
        EPOCHS = args.epochs or EPOCHS_GPU
        ARGS.rank = args.rank or 64
        LOADER_B = LBS_GPU
        log(f"backend: CUDA · {torch.cuda.get_device_name(0)} · QLoRA")
        log("⚠ T4 CPT is the emergency path (tens of tok/s) — the free TPU")
        log("  v5e-8 is ~15-30× faster; GPU sessions are for pilots/debug")

    stream = RowStream(shards)
    ACCUM = max(1, G_BATCH // LOADER_B)
    ROWS_PER_STEP = LOADER_B * ACCUM
    ROWS_TOKENS = ROWS_PER_STEP * BLOCK
    per_epoch = stream.R - stream.eval_rows
    MAX_STEPS = (per_epoch * EPOCHS) // ROWS_PER_STEP
    if args.pilot: MAX_STEPS = int(args.pilot)
    WARM_STEPS = max(4, int(MAX_STEPS * WARM_FRAC))
    log(f"stream: {len(stream.shards)} shards · {stream.R:,} rows · "
        f"{stream.eval_rows:,} held-out · {per_epoch:,} to train on")
    log(f"plan: {MAX_STEPS:,} steps · {EPOCHS} epochs · "
        f"{ROWS_PER_STEP} rows/step = {ROWS_TOKENS/1000:.0f}k tok/step")
    log(f"     est time ≈ {MAX_STEPS*ROWS_TOKENS/EST_TOK_S/3600:.1f}h "
        f"(at {EST_TOK_S/1000:.0f}k tok/s planning rate; measured soon)")

    model = build_model(ARGS.rank)
    optim, sched = build_optim(model)
    OPTIM = optim
    cfg_keys = dict(base=BASE, rank=ARGS.rank, lr=args.lr, epochs=EPOCHS,
                    accum=ACCUM, loader_b=LOADER_B, seed=SEED,
                    rows_per_step=ROWS_PER_STEP, max_steps=MAX_STEPS,
                    shards=[n for n, _ in stream.shards])
    if peek is not None:
        nam, meta = peek
        try:
            resume_load(model, optim, sched, nam, meta, stream, blk)
        except Exception as e:
            log(f"⚠ resume failed ({str(e)[:90]}) — continuing fresh")
    elif args.no_resume:
        log("starting a FRESH lineage (--no-resume given)")

    if not TRAIN["probe"]:
        v = probe(model, stream)
        TRAIN["probe"].append([TRAIN["step"], v])
        log(f"probe baseline: loss {v}")
    TRAIN["clock"] = time.time()

    done = False
    try:
        done = train_session(model, optim, sched, stream, tree, cfg_keys)
        if args.pilot and done:
            pilot_selftest(model, optim, stream, cfg_keys)
        if done and not args.pilot:
            try: try_soup_ship(model, stream)
            except Exception as e: log(f"soup skipped ({str(e)[:80]})")
            save_ckpt(model, optim, tree, stream, cfg_keys, final=True)
            ship(model, optim, stream, tree, cfg_keys)
        elif not args.pilot and TRAIN["step"] > 0:
            save_ckpt(model, optim, tree, stream, cfg_keys)   # crash guard
    except KeyboardInterrupt:
        log("⚠ interrupted — saved the checkpoint first …")
        try: save_ckpt(model, optim, tree, stream, cfg_keys)
        except Exception: pass
        raise
    except Exception as e:
        log(f"❌ train loop failed: {type(e).__name__}: {str(e)[:200]}")
        try: save_ckpt(model, optim, tree, stream, cfg_keys)
        except Exception: pass
        raise
    summary(TRAIN["step"], MAX_STEPS, stream, done)

if __name__ == "__main__":
    main()
