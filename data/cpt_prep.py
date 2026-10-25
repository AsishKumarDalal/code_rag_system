#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3 · STEP 1 — CPT data preparation   (docs/phase3.md · Part II)

  six filtered Common Pile repos (+ fineweb-edu replay)
     │
     ├─ ① AUDIT   per-repo census: text field, avg row size, dup-rate probe
     ├─ ② SELECT  seed-pinned stratified slice → quotas:
     │              stackv2_edu 90M · github_archive 40M · stackexchange 30M
     │              wikimedia 20M · arxiv_papers 20M · replay 100M  (= 300M)
     │            13-gram decontamination vs our evals (+ canary test)
     │            exact dedup (sha256) · long-doc windows · tiny-doc skip
     └─ ③ PACK    seeded global shuffle → fixed 4096-token int32 blocks
                  → data/cpt/block_NNNN.bin → banked to HF as each shard completes

Resumability (same ground-truth pattern as stages 0/1):
  • every counter, quota progress and shard ledger lives in cpt_manifest.json,
    mirrored to the HF repo; a killed session resumes at the next row/block.
  • selection streams per source with `rows_scanned` skip-counted on resume.
  • pack recomputes deterministically from span files (seeded shuffle), so a
    crash mid-pack can never corrupt already-banked blocks: rerun produces the
    byte-identical shard and skips upload when the repo already has that file.

Run (Kaggle cell):
    !pip -q install -U "huggingface_hub>=0.23" datasets transformers
    %run data/cpt_prep.py            # full run (~1-2 h CPU, no GPU needed)
    %run data/cpt_prep.py --pilot 250000   # tiny test slice (~5 min)

Wall-clock cost is CPU/network only — the GPUs of Phase 3 are untouched.

Disk roots — quota-aware (fix requested):
  • pick_root() auto-selects /tmp (ephemeral big disk) or /kaggle/working
    (~20GB persistent quota), whichever holds more free space; CPT_ROOT env
    var overrides. Peak need ≈ 2GB either way.
  • a disk guard stops data creation softly below MIN_FREE_GB — resumable.
================================================================================
"""
import os, sys, io, gc, json, time, shutil, hashlib, struct, random, argparse
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

try:
    from tqdm.auto import tqdm
except Exception:                                        # tqdm ships with hub dep
    class _MiniBar:
        def __init__(self, total=None, desc="", **k):
            self.total, self.n = total or 0, 0
        def update(self, n=1): self.n += n
        def set_postfix(self, **k): pass
        def close(self): pass
    tqdm = _MiniBar

import numpy as np

# ── Kaggle convenience: auto-install anything missing (needs Internet ON) ─────
def _ensure(pkg):
    try:
        __import__(pkg)
        return
    except Exception:
        pass
    print(f"(auto-installing {pkg} …)", flush=True)
    import subprocess
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg]).returncode
    if rc != 0:
        print(f"\nneed: pip install {pkg}"); raise SystemExit(1)

try:
    from tqdm.auto import tqdm
except Exception:
    _ensure("tqdm"); _ensure("huggingface_hub>=0.23")
    from tqdm.auto import tqdm
try:
    from huggingface_hub import HfApi, hf_hub_download
except Exception:
    _ensure("huggingface_hub>=0.23")
    from huggingface_hub import HfApi, hf_hub_download

# ══════════════════════════════════ CONFIG ════════════════════════════════════
HF_TOKEN     = os.environ.get("HF_TOKEN", "")         # or Kaggle secret HF_TOKEN
REPO         = "sddfhy546y/code_rag_dataset"          # dataset repo (banking)
SEED         = 3407                                   # pinned → identical slice
TOKENIZER    = "Qwen/Qwen3-4B"                        # Phase 3 base tokenizer
BLOCK        = 4096                                    # tokens / packed block
SHARD_MB     = 200
SHARD_BLOCKS = SHARD_MB * 1024 * 1024 // (BLOCK * 4)   # ≈ 12800 blocks/shard
SEP_TEXT     = "\n\n"                                  # separator between docs
MIN_SPAN_TOK = 80                                      # tiny-doc skip (tokens)
MIN_ROW_CHR  = 400                                     # cheap pre-tokenize filter
MAX_ROWS     = 8_000_000                               # hard cap per source scan
AUDIT_ROWS   = 8_000                                   # rows probed per audit
DUP_SAMPLE   = 2_000                                   # rows hashed for dup-rate
GRAM_N       = 13                                      # decon n-gram size
DEDUP_KEEP   = 8000                                    # hashes persisted for resume
GRACE_ROWS   = 100                                     # consecutive errors → abort

# Quotas: source → unique tokens in OUR slice (docs/phase3.md §17)
SOURCES = [
    ("common-pile/stackv2_edu_filtered",     "stackv2_edu",     90_000_000, 4095),
    ("common-pile/github_archive_filtered",  "github_archive",  40_000_000, 2000),
    ("common-pile/stackexchange_filtered",   "stackexchange",   30_000_000, 4095),
    ("common-pile/wikimedia_filtered",       "wikimedia",       20_000_000, 4095),
    ("common-pile/arxiv_papers_filtered",    "arxiv_papers",    20_000_000, 4095),
    ("HuggingFaceFW/fineweb-edu",            "replay",         100_000_000, 4095),
]
REPLAY_CONFIG = "sample-10BT"
PUBLISHED_TOKENS = {                    # source taxonomy sizes for audit sanity
    "stackv2_edu": 67.8e9, "github_archive": 11.0e9, "stackexchange": 23.9e9,
    "wikimedia": 15.8e9, "arxiv_papers": 6.0e9, "replay": 15e12,
}
EVAL_SETS = [                            # decontamination surfaces (fail-soft)
    dict(ids=["openai/openai_humaneval"],                split="test"),
    dict(ids=["google-research-datasets/mbpp"],          split="test"),
    dict(ids=["openai/gsm8k"],                           split="test", cfg="main"),
]
TEXT_FIELD_HINTS = ["text", "content", "body", "code", "prompt", "passage"]
SKIP_DECON = os.environ.get("SKIP_DECON", "0") == "1"

# quota-aware roots: /tmp = ephemeral big disk, /kaggle/working = 20GB quota.
# The working dataset itself never lands in working/ anyway (<2GB peak), only
# the small manifest mirror does — pick_root() chooses whichever has more free.
ROOT_CANDIDATES = [
    (Path("/tmp/code_rag_cpt"),           "ephemeral big disk"),
    (Path("/kaggle/working/code_rag_cpt"), "kaggle working disk (20GB quota)"),
]
MIN_FREE_GB = 6.0                                   # stop creating data below this
CPT_ROOT = DATA_DIR = TMP_DIR = MANIFEST = VISIBLE = None    # pick_root() assigns

def disk_free(p):
    try:
        return shutil.disk_usage(str(p)).free
    except Exception:
        return 0
def free_gb(p): return disk_free(p) / 1024**3

def pick_root():
    """env override wins; else the candidate with the most free space."""
    global CPT_ROOT, DATA_DIR, TMP_DIR, MANIFEST, VISIBLE
    env = os.environ.get("CPT_ROOT")
    if env:
        CPT_ROOT = Path(env); note = "(env override)"
    else:
        fro = [(free_gb(c), c, nl) for c, nl in ROOT_CANDIDATES]
        _free, CPT_ROOT, note = max(fro, key=lambda t: t[0])
        note = f"(auto: {note}, most free space)"
    DATA_DIR  = CPT_ROOT / "cpt"
    TMP_DIR   = CPT_ROOT / "tmp"
    MANIFEST  = CPT_ROOT / "cpt_manifest.json"
    VISIBLE   = Path("/kaggle/working/code_rag_cpt")
    os.environ.setdefault("HF_DATASETS_CACHE", str(TMP_DIR / "hf_cache"))
    log(f"disk: root {CPT_ROOT} {note} · free "
        + " · ".join(f"{c}={free_gb(c):.0f}GB" for c, _ in ROOT_CANDIDATES[:2]))
# ══════════════════════════════════════════════════════════════════════════════

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
    except TypeError:                                # _MiniBar fallback
        return tqdm(total=total, desc=desc)

# ── HF plumbing ───────────────────────────────────────────────────────────────
_API = None
def get_token():
    """HF token, env-first: os env → Kaggle secret → Colab secret → stored login."""
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
        t = userdata.get("HF_TOKEN")          # Kaggle-grey code path: wraps fine
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
        tok = HF_TOKEN or get_token() or os.environ.get("KAGGLE_SECRET_HF_TOKEN", "")
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
    """path → size for the dataset repo ({} on failure — then uploads retry)."""
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
    """real BufferedIOBase + progress bar (the hub rejects any other wrapper)."""
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

def bank(local: Path, repo_path: str, tree: dict, desc="⬆ shard"):
    """Upload once; skip when the repo already holds a same-size file."""
    size = local.stat().st_size
    if tree.get(repo_path) == size:
        return True
    rd = None
    try:
        rd = ProgressReader(local, bar(f"{desc} {local.name}", size, "B", True))
        api().upload_file(path_or_fileobj=rd, path_in_repo=repo_path,
                          repo_id=REPO, repo_type="dataset",
                          commit_message="cpt-prep")
        tree[repo_path] = size
        return True
    except Exception as e:
        log(f"   (upload deferred: {str(e)[:90]})")
        return False
    finally:
        if rd is not None: rd.close()

def save_manifest(silent=False):
    try:
        MANIFEST.write_text(json.dumps(M, indent=1))
        try:
            VISIBLE.mkdir(parents=True, exist_ok=True)
            (VISIBLE / "cpt_manifest.json").write_text(json.dumps(M, indent=1))
        except Exception:
            pass                                   # not on Kaggle → fine
        if not silent:
            api().upload_file(path_or_fileobj=str(MANIFEST),
                              path_in_repo="cpt_manifest.json",
                              repo_id=REPO, repo_type="dataset",
                              commit_message="cpt manifest")
    except Exception as e:
        log(f"   (manifest save deferred: {str(e)[:80]})")

# ═══════════════════════ global state (resumable) ═════════════════════════════
M = dict(kind="cpt_prep", version=1, created=now(),
         seed=SEED, base_tokenizer=TOKENIZER, block_size=BLOCK,
         shard_blocks=SHARD_BLOCKS, sep_text=SEP_TEXT, gram_n=GRAM_N,
         quotas={}, audit={}, sources={},
         evals=dict(loaded=[], failed=[], grams=0, canary_ok=None),
         dedup_prefix=[],
         pack=dict(blocks_total=0, est_blocks=0, spans=0, shards={},
                   uploaded_blocks=0),
         timings={})

def fresh_src(key, **kw):
    d = M["sources"].setdefault(key, dict(done=False, pilot=False, rows_scanned=0, tokens_got=0,
        spans_kept=0, dedup_skipped=0, decon_skipped=0, short_skipped=0,
        parse_skipped=0, error=None,
        timing=dict(select_h=0.0)))
    d.update(kw); return d

def mload():
    """bootstrap manifest: local file first, then HF copy, else fresh."""
    global M
    if MANIFEST.exists():
        try:
            M = json.loads(MANIFEST.read_text())
            log(f"manifest: loaded from disk (seed {M.get('seed')})")
            return
        except Exception as e:
            log(f"manifest unreadable ({str(e)[:60]}) → trying HF copy")
    try:
        p = hf_hub_download(REPO, "cpt_manifest.json", repo_type="dataset")
        M = json.loads(Path(p).read_text())
        log("manifest: recovered from HF repo")
        return
    except Exception:
        log("manifest: fresh start (no local / no HF copy)")

# ═══════════════════════════ tokenizer ════════════════════════════════════════
_TOK = None
def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        log(f"tokenizer: loading {TOKENIZER} … (needs internet)")
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
        log("tokenizer: ready")
    return _TOK

def encode(text):
    return tok()(text, add_special_tokens=False, truncation=False)["input_ids"]

# ═════════════════════ ① AUDIT — per-source census ═════════════════════════════
def textfield_from_row(row):
    """pick the likely main text column of a raw dataset row."""
    if not isinstance(row, dict):
        return None, []
    cands = [k for k, v in row.items() if isinstance(v, str) and len(v) > MIN_ROW_CHR]
    for hint in TEXT_FIELD_HINTS:                   # domain-known names win
        if hint in cands: return hint, list(row.keys())
    if cands: return max(cands, key=lambda k: len(row[k])), list(row.keys())
    return None, list(row.keys())

def _stream(key):
    """streaming row iterator with deterministic seed and row-skip for resume."""
    from datasets import load_dataset
    repo = SRC_BY_KEY[key]["repo"]
    cfg = REPLAY_CONFIG if key == "replay" else None
    s = M["sources"].get(key, {})
    ds = load_dataset(repo, cfg, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)   # deterministic per seed
    it = iter(ds)
    skip = int(s.get("rows_scanned", 0))
    for _ in range(skip):                            # resume mid-stream
        try: next(it)
        except StopIteration: return iter(())
    return it

def audit_source(key):
    a = M["audit"].setdefault(key, dict(ok=False, text_field=None, columns=None,
        rows_probed=0, avg_row_bytes=0.0, dup_rate=None, error=None))
    if a["ok"]:
        log(f"audit:{key} skipped (ok already)"); return True
    repo = SRC_BY_KEY[key]["repo"]
    log(f"→ audit {repo}")
    try:
        it = iter(load_raw(key))                    # unshuffled fast probe
        hashes = set()
        with bar(f"audit {key}", AUDIT_ROWS) as pb:
            for row in it:
                field, cols = textfield_from_row(row)
                a["columns"], a["rows_probed"] = cols, a["rows_probed"] + 1
                if field:
                    a["text_field"] = field
                    s = row[field]
                    a["avg_row_bytes"] += len(s)
                    if a["rows_probed"] <= DUP_SAMPLE:
                        hashes.add(hashlib.sha256(s[:8192].encode("utf-8")).hexdigest()[:16])
                if a["rows_probed"] >= AUDIT_ROWS: break
                pb.update(1)
        n = max(a["rows_probed"], 1)
        a["avg_row_bytes"] = round(a["avg_row_bytes"] / n, 1)
        a["dup_rate"] = round(1 - len(hashes) / max(min(n, DUP_SAMPLE), 1), 3)
        a["ok"] = bool(a["text_field"])
        if not a["ok"]:
            log(f"⚠ audit {key}: no usable text column ({a['columns']})")
        else:
            log(f"   field='{a['text_field']}' avg_row={a['avg_row_bytes']:.0f}B "
                f"dup={a['dup_rate']:.1%} rows_probed={a['rows_probed']:,}")
        save_manifest(silent=True)
        return a["ok"]
    except Exception as e:
        a["error"] = str(e)[:200]
        log(f"❌ audit {key} failed → {a['error']}")
        save_manifest(silent=True)
        return False

def load_raw(key):    # audit-only unshuffled loader
    from datasets import load_dataset
    repo = SRC_BY_KEY[key]["repo"]
    cfg = REPLAY_CONFIG if key == "replay" else None
    return iter(load_dataset(repo, cfg, split="train", streaming=True))

# ═══════════ 13-gram DECONTAMINATION (eval grams + canary) ════════════════════
_GRAMS = None
def build_grams():
    global _GRAMS
    if _GRAMS is not None: return _GRAMS
    if SKIP_DECON:
        log("decon: SKIPPED via SKIP_DECON=1 (not recommended)")
        return None
    M["evals"]["loaded"], M["evals"]["failed"] = [], []   # fresh lists — never inherit
    from datasets import load_dataset
    grams, hasher = set(), hashlib.blake2b
    add = grams.add
    def grams_of(words):
        n, L = GRAM_N, len(words)
        for i in range(max(L - n + 1, 0)):
            add(int.from_bytes(
                hasher(" ".join(words[i:i+n]).encode("utf-8"),
                       digest_size=8).digest(), "big"))
    log("→ loading eval texts for decontamination …")
    pb = bar("decon grams")
    for ev in EVAL_SETS:
        got = False
        for name in ev["ids"]:
            try:
                d = (load_dataset(name, ev["cfg"], split=ev["split"])
                     if ev.get("cfg") else load_dataset(name, split=ev["split"]))
                for row in d:
                    grams_of(" ".join(str(v) for v in row.values() if isinstance(v, str)).split())
                    pb.update(1)
                    if len(grams) > 4_000_000: break
                if name not in M["evals"]["loaded"]: M["evals"]["loaded"].append(name)
                got = True; break
            except Exception as e:
                log(f"   eval load failed {name} ({str(e)[:70]}) → trying next")
        if not got:
            M["evals"]["failed"].append(ev["ids"][0])
            M["evals"]["failed"] = list(dict.fromkeys(M["evals"]["failed"]))
    if grams:
        # canary: matcher must rediscover gram windows of a known eval doc
        try:
            ev0 = EVAL_SETS[0]["ids"][0]
            d0 = load_dataset(ev0, split=EVAL_SETS[0]["split"])
            words = (" ".join(str(v) for v in next(iter(d0)).values()
                              if isinstance(v, str))).split()
            M["evals"]["canary_ok"] = bool(grams_from_probe(words[10:150], grams)
                                           and grams_from_probe(words[60:200], grams))
            if not M["evals"]["canary_ok"]:
                log("⚠ canary NOT caught → matcher broken, decon disabled!")
            else:
                log("decon: canary caught ✓")
        except Exception as e:
            M["evals"]["canary_ok"] = False
            log(f"⚠ canary test error ({str(e)[:70]}) → decon disabled!")
    M["evals"]["grams"] = len(grams)
    _GRAMS = grams
    pb.close()
    log(f"decon: {len(grams):,} eval 13-grams "
        f"(loaded sets: {M['evals']['loaded'] or 'none'})")
    return grams

def grams_from_probe(words, grams):
    h = hashlib.blake2b
    for i in range(max(len(words) - GRAM_N + 1, 0)):
        k = int.from_bytes(h(" ".join(words[i:i+GRAM_N]).encode("utf-8"),
                             digest_size=8).digest(), "big")
        if k in grams: return True
    return False

def decon_hit(words):
    if _GRAMS is None or not M["evals"].get("canary_ok"): return False
    return grams_from_probe(words, _GRAMS)

# ═════════ ② SELECT — stratified seed-pinned slice (per source) ═══════════════
def do_windows(ids, cap):
    """long docs → ≤3 windows (head/mid/tail); bool marks doc-start span."""
    if len(ids) <= cap: return [(ids, True)]
    head, tail = ids[:cap], ids[-cap:]
    if len(ids) > 3 * cap:
        m0 = (len(ids) - cap) // 2
        return [(head, True), (ids[m0:m0+cap], False), (tail, False)]
    return [(head, True), (tail, False)]

class Spans:
    """per-source span/index files (regenerable, kept in TMP_DIR only)."""
    def __init__(self, key):
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.spans = open(TMP_DIR / f"spans_{key}.bin", "ab")
        self.index = open(TMP_DIR / f"idx_{key}.bin", "ab")
    def add(self, ids, first):
        off = self.spans.tell()
        self.spans.write(struct.pack("<I", len(ids)))
        self.spans.write(np.asarray(ids, dtype=np.int32).tobytes())
        self.index.write(struct.pack("<QB", off, 1 if first else 0))
    def close(self):
        self.spans.flush(); self.index.flush()
        self.spans.close(); self.index.close()

_SHA = set()
def sha16(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]

def select_source(key, quota, cap):
    """stream → audit gate → tokenize → decon → dedup → windows → span files."""
    s = fresh_src(key)
    if s["done"]:
        if s.get("pilot") and not PILOT:          # bug-fix: pilot 'done' must not block the full run
            s["done"] = False; s["pilot"] = False
            log(f"select:{key} re-opening — its done-flag came from a pilot pass "
                f"({s['tokens_got']:,} tok); continuing to the full quota")
        else:
            log(f"select:{key} skipped (done, {s['tokens_got']:,} tok)"); return True
    if PILOT: quota = min(quota, PILOT)
    if not audit_source(key):
        log(f"❌ {key} blocked by audit — source skipped this run"); return False
    field = M["audit"][key]["text_field"]
    grams_on = not SKIP_DECON
    t0 = time.time()
    log(f"→ select {SRC_BY_KEY[key]['repo']} · quota {quota/1e6:.0f}M tok · "
        f"cap {cap} · field '{field}' · resume row {s['rows_scanned']:,}")
    sp = Spans(key)
    pb = bar(f"sel {key}", unit="tok", scale=True)
    try:
        it = _stream(key)
        errs = 0
        while s["tokens_got"] < quota:
            try:
                row = next(it)
                s["rows_scanned"] += 1
                text = row.get(field) if isinstance(row, dict) else None
                if not isinstance(text, str) or len(text) < MIN_ROW_CHR:
                    s["parse_skipped"] += 1; continue
                h = sha16(text)
                if h in _SHA:
                    s["dedup_skipped"] += 1; continue
                words = text.lower().split()
                if grams_on and decon_hit(words):
                    s["decon_skipped"] += 1; continue
                ids = encode(text)
                n = len(ids)
                if n < MIN_SPAN_TOK:
                    s["short_skipped"] += 1; continue
                for span, first in do_windows(ids, cap):
                    sp.add(span, first); s["spans_kept"] += 1
                _SHA.add(h)
                s["tokens_got"] += n
                pb.update(n)
                pb.set_postfix(rows=f"{s['rows_scanned']:,}",
                               dedup=s["dedup_skipped"], decon=s["decon_skipped"],
                               kept=f"{s['spans_kept']:,}")
                if s["rows_scanned"] % 500 == 0:
                    if free_gb(CPT_ROOT) < MIN_FREE_GB:
                        s["error"] = f"low disk (<{MIN_FREE_GB}GB free)"
                        log("⚠ low disk — stopping this source cleanly; "
                            "rerun resumes at the saved row offset")
                        break
                    M["dedup_prefix"] = list(_SHA)[-DEDUP_KEEP:]
                    save_manifest(silent=True)
            except StopIteration:
                log(f"   stream exhausted at {s['tokens_got']:,}/{quota:,} tokens")
                break
            except Exception as e:
                errs += 1
                if errs >= GRACE_ROWS:
                    s["error"] = f">{GRACE_ROWS} row errors at row {s['rows_scanned']} ({str(e)[:80]})"
                    log(f"❌ {key} aborted — rerun resumes from row {s['rows_scanned']:,}")
                    break
        else:
            s["done"] = True; s["pilot"] = bool(PILOT)   # full-run 'done' is the only real one
        s["timing"]["select_h"] = round((time.time() - t0) / 3600, 3)
        log(f"   {key}: {s['tokens_got']:,} tok · spans {s['spans_kept']:,} · "
            f"dedup {s['dedup_skipped']:,} · decon {s['decon_skipped']:,} · "
            f"short {s['short_skipped']:,} · rows {s['rows_scanned']:,}"
            f" ({s['timing']['select_h']}h)")
    finally:
        sp.close()
        pb.close()
        M["dedup_prefix"] = list(_SHA)[-DEDUP_KEEP:]
        save_manifest()
    return bool(s["done"])

# ══════════════ ③ PACK — seeded shuffle → fixed 4096-token blocks ═════════════
def read_index(key):
    out, ix = [], TMP_DIR / f"idx_{key}.bin"
    with open(ix, "rb") as f:
        while True:
            b = f.read(9)
            if not b: break
            off, first = struct.unpack("<QB", b)
            out.append((off, first))
    return out

def pack_all(tree):
    PK = M["pack"]
    srcs = [s[1] for s in SOURCES if M["sources"].get(s[1], {}).get("tokens_got", 0) > 0]
    incomplete = [s[1] for s in SOURCES if not M["sources"].get(s[1], {}).get("done")]
    if not srcs:
        log("pack: no spans yet — selection needed first"); return False
    if incomplete and not PILOT:
        log("pack: SKIPPED — selection incomplete → rerun when all quotas fill")
        return False
    missing = [k for k in srcs if not (TMP_DIR / f"idx_{k}.bin").exists()]
    if missing:
        if M["pack"].get("finalized"):
            log("pack: already finalized earlier (span files cleaned) — nothing to do")
            return True
        log(f"pack: span files missing for {missing} → re-select those sources first")
        return False
    # content fingerprint: identical spans ⇒ identical shard names, so the
    # size-based upload skip is always safe; a changed slice gets new names.
    fp_raw = json.dumps([SEED, {k: M["sources"][k]["tokens_got"] for k in srcs}],
                        sort_keys=True).encode("utf-8")
    PREFIX = "c" + hashlib.blake2b(fp_raw, digest_size=4).hexdigest()
    PK["fp"] = PREFIX
    PK["shards"] = {}                                  # fresh ledger each pack
    PK["blocks_total"] = 0                             # fresh recompute — never inherit
    PK["uploaded_blocks"] = 0
    records, per_src = [], {}
    for si, key in enumerate(srcs):
        per_src[si] = open(TMP_DIR / f"spans_{key}.bin", "rb")
        records.extend((si, off, first) for off, first in read_index(key))
    rng = random.Random(SEED + 11)
    rng.shuffle(records)
    tot = sum(M["sources"][k]["tokens_got"] for k in srcs)
    est_blocks = (tot + BLOCK - 1) // BLOCK
    PK["est_blocks"] = est_blocks; PK["spans"] = len(records)
    log(f"pack: {len(records):,} spans · {tot:,} tokens → ~{est_blocks:,} × 4096-blocks"
        f" · fingerprint {PREFIX}"
        f"{' · PILOT slice' if incomplete else ''}")
    pb = bar("pack→blocks", est_blocks, unit="blk")
    sep = np.asarray(encode(SEP_TEXT), dtype=np.int32)
    carry = np.empty(0, dtype=np.int32)
    n4 = BLOCK * 4
    state = dict(f=None, name=None, blocks=0)
    def shard_name(idx): return f"{PREFIX}_block_{idx:04d}.bin"
    def open_shard(idx):
        if free_gb(DATA_DIR) < 1.0:
            raise RuntimeError(f"disk nearly full before shard #{idx} — "
                               "blocks so far are banked; rerun after freeing space")
        f = open(DATA_DIR / shard_name(idx), "wb")
        state.update(f=f, name=shard_name(idx), blocks=0)
        return f
    def settle(idx):
        name, p = shard_name(idx), DATA_DIR / shard_name(idx)
        PK["shards"][name] = dict(blocks=state["blocks"], bytes=p.stat().st_size,
                                  uploaded=False)
        ok = bank(p, f"data/cpt/{name}", tree)
        PK["shards"][name]["uploaded"] = ok
        if ok:
            p.unlink()
            PK["uploaded_blocks"] += state["blocks"]
        save_manifest(silent=True)
    if not DATA_DIR.exists(): DATA_DIR.mkdir(parents=True)
    try:
        f = open_shard(0)
        for si, off, first in records:
            h = per_src[si]; h.seek(off)
            (n,) = struct.unpack("<I", h.read(4))
            ids = np.frombuffer(h.read(n * 4), dtype=np.int32)
            chunk = np.concatenate([sep, ids]) if first else ids
            if carry.size: chunk = np.concatenate([carry, chunk])
            full = len(chunk) // BLOCK
            for b in range(full):
                f.write(chunk[b*BLOCK:(b+1)*BLOCK].astype(np.int32).tobytes())
                state["blocks"] += 1; PK["blocks_total"] += 1
                pb.update(1)
                if state["blocks"] >= SHARD_BLOCKS:    # shard complete → bank it
                    idx = (PK["blocks_total"] - 1) // SHARD_BLOCKS
                    f.close(); settle(idx)
                    f = open_shard(idx + 1)
                elif PK["blocks_total"] % 500 == 0:
                    save_manifest(silent=True)
            carry = chunk[full*BLOCK:]
    finally:
        for fh in per_src.values(): fh.close()
        try:
            if state.get("f") and not state["f"].closed: state["f"].close()
        except Exception:
            pass
    # final partial block → pad with EOS so every block is exactly 4096 tokens
    if carry.size:
        last_idx = PK["blocks_total"] // SHARD_BLOCKS   # shard of the final block
        p = DATA_DIR / shard_name(last_idx)
        with open(p, "r+b") as f2:
            f2.seek(state["blocks"] * n4)
            pad = np.full(BLOCK - carry.size,
                          int(tok().eos_token_id) % (2**31), dtype=np.int32)
            f2.write(np.concatenate([carry, pad]).astype(np.int32).tobytes())
        state["blocks"] += 1; PK["blocks_total"] += 1; pb.update(1)
    final_idx = max((PK["blocks_total"] - 1) // SHARD_BLOCKS, 0)
    if (DATA_DIR / shard_name(final_idx)).exists():
        settle(final_idx)                              # bank last (partial) shard
    for p in DATA_DIR.glob(f"{PREFIX}_block_*.bin"):   # drop empty leftovers
        if p.stat().st_size == 0: p.unlink()
    pb.close()
    un_up = [n for n, v in PK["shards"].items() if not v["uploaded"]]
    # disk hygiene: only clean up when EVERYTHING is banked on HF — otherwise
    # the span files stay so a rerun can retry the missing uploads
    if not incomplete and not PILOT and PK["shards"] and not un_up:
        freed = 0
        for p in TMP_DIR.glob("*"):
            try:
                if p.is_file():
                    freed += p.stat().st_size; p.unlink()
            except Exception:
                pass
        PK["finalized"] = True
        log(f"cleanup: span intermediates freed {gfmt(freed)} (pack finalized)")
    elif not PILOT:
        log(f"⚠ kept span files ({len(un_up)} shard(s) not yet uploaded) — "
            f"rerun retries those uploads (and selection if needed)")
    M["timings"]["pack_s"] = round(time.time() - T0, 1)
    save_manifest()
    log(f"pack: {PK['blocks_total']:,} blocks · "
        f"{gfmt(PK['blocks_total'] * n4)} tokenized · "
        f"uploaded {gfmt(PK['uploaded_blocks'] * n4)}")
    return True

# ══════════════════════════════════ main ══════════════════════════════════════
PILOT = 0
SRC_BY_KEY = {k: dict(repo=r) for (r, k, _q, _c) in SOURCES}
TREE: dict = {}

def summary(failed):
    log("══════════════ CPT PREP SUMMARY ══════════════")
    tot = 0
    for (repo, key, quota, _cap) in SOURCES:
        s = M["sources"].get(key, fresh_src(key))
        flag = ("✔ done" if s["done"]
                else ("❌ failed" if s["error"] or key in failed else "… pending"))
        log(f"   {key:<15} {s['tokens_got']/1e6:8.1f}M/{quota/1e6:.0f}M tok · "
            f"spans {s['spans_kept']:>8,} · dedup {s['dedup_skipped']:>7,} · "
            f"decon {s['decon_skipped']:>5,} · {flag}")
        tot += s["tokens_got"]
    PK = M["pack"]
    log(f"   TOTAL tokens {tot:,} · blocks {PK['blocks_total']:,} / "
        f"est {PK['est_blocks']:,} · shards {len(PK['shards'])} "
        f"(uploaded {sum(1 for v in PK['shards'].values() if v['uploaded'])})")
    log(f"   decon chips grams {M['evals']['grams']:,} · canary "
        f"{M['evals']['canary_ok']} · eval-load failures "
        f"{M['evals']['failed'] or 'none'}")
    log(f"   elapsed {time.time()-T0:,.0f}s · manifest {MANIFEST}")
    log("next → data/cpt_train.py (Phase 3 · CPT training)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int,
                    default=int(os.environ.get("CPT_PILOT", "0")),
                    help=">0 caps each source's tokens (quick test run); "
                         "or set env CPT_PILOT=200000")
    ap.add_argument("--reselect", action="store_true",
                    help="clear pilot-poisoned done-flags & pack state "
                         "(row offsets + dedup prefix are kept)")
    a = ap.parse_args()
    global PILOT
    PILOT = a.pilot
    log(f"════════════ CPT PREP · seed {SEED} · block {BLOCK} tok · "
        f"shard {SHARD_MB}MB · pilot={PILOT or 'off'} ════════════")
    pick_root()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if free_gb(CPT_ROOT) < MIN_FREE_GB:
        log(f"⚠ root has {free_gb(CPT_ROOT):.1f}GB free (<{MIN_FREE_GB}) — "
            f"set CPT_ROOT (e.g. /kaggle/working/code_rag) and rerun")
    mload()
    tok()                                       # fail fast with a clear error
    for (r, k, q, _c) in SOURCES:
        M["quotas"][k] = q
        fresh_src(k)
    global TREE
    TREE = repo_tree()
    log("plan: %dM tokens from %d source(s)" % (
        sum(M["quotas"].values()) / 1e6, len(SOURCES)))
    if a.reselect or os.environ.get("CPT_RESELECT") == "1":
        for (r, k, q, c) in SOURCES:
            fresh_src(k, done=False, pilot=False, error=None)
        M["pack"]["finalized"] = False
        save_manifest()
        log("reselect: done-flags cleared (row offsets kept) — selection will "
            "continue to full quotas")
    _SHA.update(M.get("dedup_prefix", []))
    # ① audit
    for (_r, k, *_rest) in SOURCES:
        audit_source(k)
    # ② select (decon built once inside)
    build_grams()
    for (_r, k, q, c) in SOURCES:
        try:
            select_source(k, q, c)
        except Exception as e:
            fresh_src(k, error=str(e)[:200])
            log(f"❌ {k} failed → recorded, continuing ({str(e)[:90]})")
        gc.collect()
    # ③ pack + bank
    try:
        pack_all(TREE)
    except Exception as e:
        log(f"⚠ pack failed ({str(e)[:120]}) — rerun recomputes blocks "
            f"identically and re-uploads only what is missing")
    failed = [k for (_r, k, _q, _c) in SOURCES
              if not M["sources"].get(k, {}).get("done")]
    if failed:
        log(f"⚠ incomplete sources: {failed} → rerun resumes at saved row offsets")
    save_manifest()
    summary(failed)

if __name__ == "__main__":
    main()
