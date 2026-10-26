#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 5 · STEP 1 — SFT DATA BUILD  (docs/phase_5_sft.md · the build contract)

    public teacher corpora  →  ONE loop grammar  →  banked episodes
        │
        ├─ ① STREAM   corag/multihopqa (2wiki/hotpot/musique · train) and
        │             nvidia/Nemotron-Cascade-SFT-Stage-2 (instrument/
        │             consequence/code · streamed jsonl, never whole-file)
        ├─ ② REFLOW   every item into the loop trace:
        │             <question>→<plan>→<search>→<obs>→…→<thinking>→<answer>
        │             <obs> blocks are CONTEXT (masked at train time): the
        │             model must learn to consume them, never invent them
        ├─ ③ GATES    format state-machine / length / obs-pairing / dedup /
        │             13-gram decontamination vs HumanEval·MBPP·GSM8K
        │             (same honest-eval EVAL_SETS rule as cpt_prep.py)
        ├─ ④ QUOTAS   ≈46k episodes · gate→instrument→consequence→code
        │             curriculum order preserved in file order
        ├─ ⑤ SPLIT    hash-fixed val_*.jsonl (~1.1% = ~512 of 46k) — val
        │             checkpoint selection in sft_train.py
        └─ ⑥ BANK     data/sft/episodes_<fp>_NNNN.jsonl + sft_manifest.json
                      → HF repo sddfhy546y/code_rag_dataset (ground truth);
                      shard → uploaded → deleted (local peak <1GB)

Corag obs is an EVIDENCE BUNDLE, not chunks: corag/multihopqa ships
subqueries/subanswers but only doc-ID lists (no chunk text), so the bundle
keeps the consume-and-cite semantics; when Phase 4's index exists we can
rebuild with real retrieved chunks without touching the grammar.

Storage: HF cache lives in /tmp (ephemeral); /kaggle/working is never used.

Trace contract shared with sft_train.py:
    trace_text = "\n".join(f"<{tag}>{text}</{tag}>" for tag,text,_ in parts)
    (parts order IS the curriculum; mask=1 parts contribute no loss)

Run (any Kaggle cell — no GPU needed):
    %run sft/sft_build.py --probe          # print raw schemas once, exit
    %run sft/sft_build.py --pilot 40       # tiny slice (~10 min)
    %run sft/sft_build.py                  # full build (~1.5-2.5h CPU)
    %run sft/sft_build.py --stat           # summarize banked data
    %run sft/sft_build.py --no-resume      # fresh lineage (default = resume)
================================================================================
"""
import os, sys, re, gc, json, time, shutil, hashlib, argparse
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# caches on the ephemeral disk — NOT the 20GB /kaggle/working quota
try:
    _H = "/tmp/code_rag_sft/hf"; Path(_H).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", _H)
except Exception:
    pass

# ════════════════════════════════ CONFIG ══════════════════════════════════════
TMP_ROOT = Path(os.environ.get("SFT_BUILD_TMP", "/tmp/code_rag_sft"))
ROOT_CANDIDATES = [
    (Path("/tmp/code_rag_sft"), "ephemeral big disk"),
    (Path("/kaggle/working/.sft_stage"), "kaggle working disk (20GB quota)"),
]

REPO = "sddfhy546y/code_rag_dataset"   # same bank as Phase 3 (HF ground truth)
DATA_SUB = "data/sft"                  # episodes bank here
MANIFEST = "sft_manifest.json"

CORAG_ID = "corag/multihopqa"
CORAG_CFGS = {            # config: desired episodes (train has subqueries)
    "2wikimultihopqa": 10000,
    "hotpotqa":        12000,
    "musique":          6000,
}
NEMO_ID = "nvidia/Nemotron-Cascade-SFT-Stage-2"
NEMO_FILES = {   # verified layout: root tool_calling + general/ + code/ folders
    "instrument":  ["tool_calling.jsonl"],
    "consequence": ["general/*.jsonl", "general.jsonl"],
    "code":        ["code/*.jsonl", "code.jsonl"],
}
GENRE_QUOTA = {"instrument": 8000, "consequence": 4000, "code": 6000}
GENRE_ORDER = ["gate", "instrument", "consequence", "code"]

SEED = 3407
MAX_TOK_EP = 6144          # per-episode token cap (tokenizer or chars/4 est)
MIN_TOK_EP = 48
MIN_QUESTION = 14
MIN_ANSWER = 8
SHARD_ROWS = 6000          # episodes per JSONL shard (~50-70MB)
SAMPLE_TOK = 20000         # rows tokenized for the token estimate
EST_CHAR_TOK = 3.7         # chars/token fallback when the tokenizer is absent

# any test surface we will eval on — SAME definition as Phase 3 (§26 honesty)
EVAL_SETS = [
    ("openai_humaneval", None, "test", "prompt"),
    ("google-research-datasets/mbpp", None, "full", ["text", "code"]),
    ("openai/gsm8k", "main", "test", "question"),
]
NGRAM = 13
DECON_LIMIT = 4000
CANARY_KEY = "the midwest is a region of the united states of america"

# ════════════════════════════ small plumbing ══════════════════════════════════
T0 = time.time()
def now(): return datetime.now(timezone.utc).strftime("%H:%M:%S")
def log(msg): print(f"[{now()}] {msg}", flush=True)
def gfmt(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"
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
    log(f"disk: root {TMP_ROOT} (auto: {why}) · free {free_gb(TMP_ROOT):.0f}GB")
def sha16(s):
    return hashlib.sha256(str(s or "").encode("utf-8", "ignore")).hexdigest()[:16]
def head(s, n=160):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return (s[:n] + "…") if len(s) > n else s

def ensure_libs(specs):
    from importlib.metadata import version
    import subprocess
    for spec in specs:
        name = spec.split(">=")[0]
        try: have = version(name)
        except Exception: have = None
        if have: continue
        log(f"installing missing lib: {spec} …")
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                             spec]).returncode
        if rc != 0:
            log(f"❌ pip install {spec} failed — install it once with:\n"
                f"    !pip install {spec}")
            raise SystemExit(1)

# ── HF token resolution (env → Kaggle secret → Colab secret) ─────────────────
_API = None
def get_token():
    t = os.environ.get("HF_TOKEN", "")
    if t: return t
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t:
            os.environ["HF_TOKEN"] = t
            log("token: from Kaggle Secrets")
            return t
    except Exception:
        pass
    try:
        from google.colab import userdata
        t = userdata.get("HF_TOKEN")
        if t:
            os.environ["HF_TOKEN"] = t
            log("token: from Colab userdata")
            return t
    except Exception:
        pass
    return ""

def api():
    global _API
    if _API is None:
        from huggingface_hub import HfApi
        tok = get_token()
        if tok:
            try:
                from huggingface_hub import login
                login(token=tok)
            except Exception as e:
                log(f"login warning ({str(e)[:80]})")
        else:
            log("⚠ no HF_TOKEN — pushes only work if already logged in")
        _API = HfApi()
    return _API

def repo_tree():
    try:
        items = api().list_repo_tree(REPO, repo_type="dataset", recursive=True)
        return {i.path: getattr(i, "size", None) for i in items}
    except Exception as e:
        log(f"⚠ repo tree unavailable ({str(e)[:70]}) — uploads will re-run")
        return {}

class UploadReader:
    """file-like + progress bar (hub rejects hand-rolled wrappers)."""
    def __init__(self, path, pb):
        self._f, self._pb = open(path, "rb"), pb
    def read(self, size=-1):
        d = self._f.read(size)
        if d and self._pb: self._pb.update(len(d))
        return d
    def close(self):
        try: self._f.close()
        except Exception: pass

def bank(local: Path, repo_path: str, tree: dict, desc="⬆", force=False):
    """upload-or-skip by size (idempotent-ish; crash → rerun re-ups)."""
    local = Path(local)
    need = local.stat().st_size
    if not force and tree.get(repo_path) == need:
        return True
    try:
        from tqdm.auto import tqdm as _tq
        pb = _tq(total=need, desc=f"{desc} {local.name}", unit="B",
                 unit_scale=True, dynamic_ncols=True)
    except Exception:
        pb = None
    try:
        if pb is None:
            api().upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                              repo_id=REPO, repo_type="dataset",
                              commit_message="sft-build")
        else:
            api().upload_file(path_or_fileobj=UploadReader(local, pb),
                              path_in_repo=repo_path, repo_id=REPO,
                              repo_type="dataset", commit_message="sft-build")
        tree[repo_path] = need
        return True
    except Exception as e:
        log(f"   (upload deferred: {str(e)[:100]})")
        return False
    finally:
        if pb is not None: pb.close()

def hf_pull(path):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, path, repo_type="dataset"))

# ═══════════════ decontamination (honest-eval rule, same as Phase 3) ══════════
def _clean_tokens(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()

def _ngrams(tokens, n=NGRAM):
    if len(tokens) < n: return set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

def load_eval_ngrams():
    from datasets import load_dataset
    sets = []
    for dsid, cfg, split, fields in EVAL_SETS:
        grams, n = set(), 0
        try:
            ds = load_dataset(dsid, cfg, split=split, streaming=True)
            for row in ds:
                if n >= DECON_LIMIT: break
                flds = [fields] if isinstance(fields, str) else fields
                for f in flds:
                    t = str(row.get(f) or "")
                    if t: grams |= _ngrams(_clean_tokens(t))
                n += 1
        except Exception as e:
            log(f"decon: {dsid} unavailable ({str(e)[:60]}) — fail-soft skip")
        sets.append((dsid, grams))
        log(f"decon: {dsid} → {len(grams):,} {NGRAM}-grams")
    probe = _ngrams(_clean_tokens(CANARY_KEY))
    if any(probe & g for _, g in sets if g):
        log("decon: canary ✓")
    else:
        log("decon: ⚠ canary missed (eval sets came back empty?) — decon "
            "will under-block; RED FLAG, check the network/configs")
    return sets

def decon_hit(text, sets):
    toks = _clean_tokens(text or "")
    if len(toks) < NGRAM: return None
    grams = _ngrams(toks)
    if not grams: return None
    for dsid, gs in sets:
        if grams & gs: return dsid
    return None

# ═════════════════════════ episode contract + tokenizer meter ═════════════════
TAG_RE = re.compile(r"<(question|plan|search|obs|thinking|answer)>")
HOPS_MAX = 5

_TOK = None
def tokenizer():
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
            log("meter: Qwen tokenizer loaded (exact episode sizes)")
        except Exception as e:
            log(f"meter: tokenizer unavailable ({str(e)[:70]}) — chars/"
                f"{EST_CHAR_TOK} fallback estimates (stricter caps)")
    return _TOK

def est_tokens(ep, tok=None):
    """loss-relevant size meter (tags add a few tokens — cap has margin)."""
    n = 0
    if tok is not None:
        try:
            for _tag, t, _m in ep["parts"]:
                n += len(tok(t or " ")["input_ids"])
            return n
        except Exception:
            pass
    return int(sum(len(p[1]) for p in ep["parts"]) / EST_CHAR_TOK)

def episode_gate(ep):
    """the trace contract as a state machine → '' ok | reason-string."""
    parts = ep.get("parts") or []
    if not parts: return "no parts"
    i, n = 0, len(parts)
    def tag(k): return parts[k][0]
    def txt(k): return parts[k][1]
    def mask(k): return parts[k][2]
    if tag(0) != "question": return "trace must start <question>"
    i = 1
    if i < n and tag(i) == "plan": i += 1
    hops = 0
    while i < n and tag(i) == "search":
        hops += 1
        i += 1
        if i >= n or tag(i) != "obs": return "search without obs"
        if mask(i) != 1: return "obs must be masked"
        i += 1
    if hops > HOPS_MAX: return f"too many hops ({hops})"
    if i >= n or tag(i) != "thinking": return "missing <thinking> block"
    i += 1
    if i >= n or tag(i) != "answer": return "missing <answer> block"
    i += 1
    if i != n: return f"stray tail blocks ({n - i})"
    for k in range(n):
        if mask(k) != 1 and not txt(k): return f"empty {tag(k)}"
        if TAG_RE.search(txt(k)): return "pre-tagged text"
    q = txt(0)
    a = txt(n - 1)
    if len(q) < MIN_QUESTION: return "question too short"
    if len(a) < MIN_ANSWER: return "answer too short"
    rag_expected = 1 if hops else 0
    if int((ep.get("meta") or {}).get("rag", -1)) != rag_expected:
        return f"meta.rag wrong ({(ep.get('meta') or {}).get('rag')}≠{rag_expected})"
    if int((ep.get("meta") or {}).get("hops", -1)) != hops:
        return f"meta.hops wrong ({(ep.get('meta') or {}).get('hops')}≠{hops})"
    if hops and "Sources: [E" not in a:
        return "answer without evidence tags (rag episode)"
    return ""

# ════════════════════════ reflow helpers (trace assembly) ═════════════════════
def part(tag, text, masked=0):
    return (tag, re.sub(r"[ \t]+", " ", str(text or "")).strip(), masked)

def question_of(ep):
    for tag, t, _m in ep["parts"]:
        if tag == "question": return t
    return ""
def trace_text(ep):
    return "\n".join(f"<{t}>{x}</{t}>" for t, x, _m in ep["parts"])

# ══════════════════════ builder 1: corag → gate episodes ═════════════════════
def plan_text(subqueries):
    lines = [f"{i+1}) {sq}" for i, sq in enumerate(subqueries)]
    return ("Decompose the question into ordered sub-problems and resolve "
            "each with evidence:\n" + "\n".join(lines) +
            "\nThen combine the evidence to answer the original question.")

def gate_ep(row, src_note):
    """corag row → gate episode. Returns (episode|None, drop_reason)."""
    sq = row.get("subqueries") or []
    sa = row.get("subanswers") or []
    q = str(row.get("query") or "").strip()
    ans = [str(a).strip() for a in (row.get("answers") or []) if str(a or "").strip()]
    if not sq or not sa: return None, "no subqueries/subanswers"
    if len(sq) != len(sa): return None, f"hop pair mismatch ({len(sq)} vs {len(sa)})"
    if not q: return None, "empty question"
    if not ans or not ans[0]: return None, "empty answer"
    if len(sq) > HOPS_MAX: return None, f"hops {len(sq)} > {HOPS_MAX}"
    if TAG_RE.search(q) or any(TAG_RE.search(str(x or "")) for x in list(sq) + list(sa)):
        return None, "pre-tagged text"
    ids = row.get("context_doc_ids") or []
    parts = [part("question", q), part("plan", plan_text(sq))]
    evid = []
    for i, (sqi, sai) in enumerate(zip(sq, sa)):
        sqi, sai = str(sqi or "").strip(), str(sai or "").strip()
        if not sqi: return None, f"empty subquery hop {i+1}"
        if not sai: return None, f"empty subanswer hop {i+1}"
        two = ids[i * 8: i * 8 + 2] if len(ids) > i * 8 else []
        docs = ", ".join(str(x) for x in two) if two else "source doc set"
        obs = ("[E" + str(i + 1) + "] Evidence bundle for sub-problem " +
               str(i + 1) + ' "' + head(sqi, 90) + '" (docs: ' + docs + '):\n' +
               head(sai, 1200).replace("\n", " "))
        parts.append(part("search", sqi))
        parts.append(part("obs", obs, masked=1))
        evid.append((i + 1, sqi, sai))
    ev = "\n".join("[E" + str(i) + "] " + head(s, 110) + " -> " + head(a, 160)
                   for i, s, a in evid)
    n_miss = sum(1 for _i, _s, a in evid
                 if a.lower().startswith("no relevant information"))
    if n_miss:
        think = ("Sub-problem evidence, in order:\n" + ev + "\n" +
                 str(n_miss) + " of " + str(len(evid)) +
                 " evidence blocks report nothing relevant. Do not invent "
                 "the missing part; state what the evidence supports and "
                 "flag the gap.")
    else:
        think = ("Sub-problem evidence, in order:\n" + ev + "\nAll evidence "
                 "blocks are usable. Compose the final answer from them; "
                 "cite every evidence tag used.")
    final = "; ".join(ans) if len(ans) > 1 else ans[0]
    cite = " ".join("[E" + str(i) + "]" for i, _s, _a in evid)
    parts.append(part("thinking", think))
    parts.append(part("answer", final + "\nSources: " + cite))
    return (dict(parts=parts, meta=dict(source=src_note, genre="gate",
                                        hops=len(sq), rag=1)), "")

# ═════════════builder 2: nemotron → instrument/consequence/code ══════════════
def to_turns(row):
    if isinstance(row.get("conversations"), list):
        out = []
        for t in row["conversations"]:
            if not isinstance(t, dict): continue
            f = str(t.get("from") or t.get("role") or "").lower()
            role = {"human": "user", "gpt": "assistant", "system": "system",
                    "tool": "tool", "function": "tool",
                    "user": "user", "assistant": "assistant"}.get(f, f)
            v = t.get("value", t.get("content"))
            if v is None and isinstance(t.get("tool_calls"), list):
                v = json.dumps(t["tool_calls"])
            out.append((role, str(v or "")))
        return out
    if isinstance(row.get("messages"), list):
        return [(str(t.get("role") or ""), str(t.get("content") or ""))
                for t in row["messages"] if isinstance(t, dict)]
    if isinstance(row.get("input"), (str, list)) and \
       isinstance(row.get("output"), (str, list)):
        ins = row["input"] if isinstance(row["input"], str) else \
              "\n".join(str(x) for x in row["input"])
        outs = row["output"] if isinstance(row["output"], str) else \
               "\n".join(str(x) for x in row["output"])
        return [("user", ins), ("assistant", outs)]
    if isinstance(row.get("question"), (str, list)) and \
       isinstance(row.get("response"), (str, list)):
        qv = row["question"] if isinstance(row["question"], str) else \
            " ".join(str(x) for x in row["question"])
        rv = row["response"] if isinstance(row["response"], str) else \
            " ".join(str(x) for x in row["response"])
        return [("user", qv), ("assistant", rv)]
    return None

TOOL_CALL_RE = re.compile(r"<(?:tool_call|tools)>(.*?)</(?:tool_call|tools)>", re.S)
TOOL_FENCE_RE = re.compile(r"```(?:tool_call|json)\s*(\{.*?\})\s*```", re.S)

def calls_in(text):
    t = text or ""
    return TOOL_CALL_RE.findall(t) + TOOL_FENCE_RE.findall(t)

def calls_list(row, turns):
    out = []
    if isinstance(row.get("tool_calls"), list):
        out.append(json.dumps(row["tool_calls"]))
    for r, c in turns:
        if r == "assistant":
            out.extend(calls_in(c))
    return out

def strip_reasoning(text):
    t = str(text or "")
    ot, ct = "<" + "think" + ">", "</" + "think" + ">"
    t = re.sub(re.escape(ot) + ".*?" + re.escape(ct), "", t, flags=re.S)
    t = re.sub(r"^\s*think(?:ing)?:\s*", "", t, flags=re.I | re.M)
    return t.strip()

NO_SEARCH_THINK = {
    "consequence":
        "No retrieval needed - this is a conversational reply; answer from "
        "what I already know.",
    "code":
        "No retrieval needed - this is a code problem; state the "
        "constraints, build the solution, walk through it, then hand over "
        "the code.",
}

def nemtek_ep(row, genre):
    """nemotron row → instrument (tool loop) or consequence/code episode."""
    turns = to_turns(row)
    if not turns: return None, "unrecognized schema"
    turns = [(r, c) for r, c in turns if c and str(c).strip()]
    if not turns: return None, "all-empty turns"
    for _r, c in turns:
        if TAG_RE.search(str(c)): return None, "already-tagged text"
    users = [c for r, c in turns if r == "user" and len(str(c).strip()) > 8]
    if not users: return None, "no user turn"
    q = str(users[0]).strip()[:6000]    # cap masked run: 6k chars < seq tokens
    if len(q) < MIN_QUESTION: return None, "question too short"
    if genre == "instrument":
        calls = calls_list(row, turns)
        obs_turns = [c for r, c in turns if r == "tool" and str(c).strip()]
        if not calls and not obs_turns:
            return None, "instrument without tool evidence"
        parts = [part("question", q),
                 part("plan", "Collect the missing facts with explicit tool "
                              "calls; read every observation; then compose "
                              "the final reply.")]
        obs_body = "\n".join(head(c, 1200) for c in obs_turns[:3]).strip()
        if not obs_body:
            obs_body = "\n".join("call -> " + head(c, 600) for c in calls[:3])
        if not obs_body: return None, "no usable observation"
        sq = ("tool calls: " + head(" | ".join(head(c, 150) for c in calls[:2]),
                                    300) if calls else "lookup: " + head(q, 140))
        parts.append(part("search", sq))
        parts.append(part("obs", "[E1] Tool observations:\n" + obs_body,
                          masked=1))
        final = [c for r, c in turns if r == "assistant" and
                 not calls_in(c) and len(str(c).strip()) > MIN_ANSWER]
        if not final: return None, "no natural-language reply"
        think = ("Check the tool observation [E1] against the plan; use its "
                 "fields for the reply; flag anything absent instead of "
                 "inventing values.")
        ans = strip_reasoning(final[-1]) + "\nSources: [E1]"
        parts.append(part("thinking", think))
        parts.append(part("answer", ans))
        return (dict(parts=parts,
                     meta=dict(source=NEMO_ID, genre="instrument", hops=1,
                               rag=1)), "")
    asst = [c for r, c in turns if r == "assistant"]
    if not asst: return None, "no assistant turn"
    a = strip_reasoning(asst[-1])
    if len(a) < MIN_ANSWER: return None, "answer too short"
    if len(a) > 40000: return None, "answer too long"
    return (dict(parts=[part("question", q),
                        part("thinking", NO_SEARCH_THINK[genre]),
                        part("answer", a)],
                 meta=dict(source=NEMO_ID, genre=genre, hops=0, rag=0)), "")

# ══════════════════════════════ source streams ════════════════════════════════
def skip_n(it, n):
    """fast-forward a stream to a resume offset (cheap, no host bloat)."""
    for _ in range(n):
        try: next(it)
        except StopIteration: return True
    return False

def stream_corag(cfg):
    from datasets import load_dataset
    return iter(load_dataset(CORAG_ID, cfg, split="train", streaming=True))

def stream_nemo(genre):
    from datasets import load_dataset
    cands = NEMO_FILES[genre]
    last = None
    for rel in cands:
        try:
            ds = load_dataset("json", data_files=f"{NEMO_ID}/{rel}",
                              split="train", streaming=True)
            return iter(ds)
        except Exception as e:
            last = e
    log(f"⚠ {NEMO_ID} files for {genre} unreachable ({str(last)[:70]})")
    return iter(())

# ═══════════════════════════════ producer ════════════════════════════════════
def effective_quotas():
    q = dict(GENRE_QUOTA)
    q["gate"] = sum(CORAG_CFGS.values())
    return q

def produce(genre, want, sets, state, tok):
    """stream source(s) until `want` fresh gated episodes (resume-aware).

    state["scan"] = absolute rows consumed per stream (saved only at flush;
    a crash rewinds to the last flush, so nothing produced is ever lost).
    """
    built_total = 0
    if genre == "gate":
        tw = sum(CORAG_CFGS.values())
        alloc, acc = {}, 0
        names = list(CORAG_CFGS)
        for c in names[:-1]:
            alloc[c] = want * CORAG_CFGS[c] // tw
            acc += alloc[c]
        alloc[names[-1]] = want - acc
    else:
        alloc = {NEMO_ID: want}
    try:
        from tqdm.auto import tqdm
        pb = tqdm(total=want, desc=f"build {genre}", unit="ep",
                  dynamic_ncols=True)
    except Exception:
        pb = None
    for src, want_c in alloc.items():
        if want_c <= 0 or built_total >= want: continue
        key = f"{genre}:{src}"
        offset = int(state["scan"].get(key, 0))
        it = stream_corag(src) if genre == "gate" else stream_nemo(genre)
        if skip_n(it, offset):
            log(f"⚠ {key}: stream ended before resume offset {offset:,}")
        scanned, got_c = 0, 0
        while got_c < want_c:
            try:
                row = next(it)
            except StopIteration:
                break
            scanned += 1
            state["scan"][key] = offset + scanned
            ep, reason = (gate_ep(row, f"{CORAG_ID}/{src}") if genre == "gate"
                          else nemtek_ep(row, genre))
            if ep is None:
                state["drop"].setdefault(reason, [0, genre])[0] += 1
                continue
            r = episode_gate(ep)
            if r:
                state["drop"].setdefault("gate: " + r, [0, genre])[0] += 1
                continue
            if decon_hit(trace_text(ep), sets):
                state["drop"].setdefault("decontamination", [0, genre])[0] += 1
                continue
            h = sha16(question_of(ep))
            if h in state["seen"]:
                state["drop"].setdefault("duplicate question", [0, genre])[0] += 1
                continue
            state["seen"].add(h)
            state["eps"].setdefault(genre, []).append(ep)
            built_total += 1
            got_c += 1
            if pb: pb.update(1)
            if built_total % 4000 == 0:
                if pb: pb.set_postfix_str(f"scanned {scanned:,}")
                gc.collect()
        label = (f"{CORAG_ID}/{src}" if genre == "gate"
                 else f"{NEMO_ID}/{genre}")
        log(f"produce {label}: +{got_c:,} (want {want_c:,} · "
            f"scanned {scanned:,} this pass)")
    if pb: pb.close()
    return built_total

# ═══════════════════ manifest + deterministic val + banking ═══════════════════
VAL_PERMILLE = 11   # ~1.1% of episodes -> ~512 val out of ~46k; hash-fixed

def is_val(ep):
    """val membership is a property of the episode, not of the session —
    so partial flushes and resumed flushes always agree."""
    h = int(sha16(question_of(ep))[:4], 16)
    return h < (65536 * VAL_PERMILLE) // 1000

def load_banked_manifest():
    for rel in (MANIFEST, f"{DATA_SUB}/{MANIFEST}"):
        try:
            return json.loads(hf_pull(rel).read_text(encoding="utf-8"))
        except Exception:
            continue
    return None

def write_jsonl(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for ep in rows:
            f.write(json.dumps(ep, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

def save_manifest(m, tree):
    p = TMP_ROOT / MANIFEST
    p.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    bank(p, MANIFEST, tree, desc="⬆ manifest", force=True)

def flush(state, m, tree, tok, pilot, complete):
    """merge banked rows + this session -> rewrite shards with a
    deterministic val split -> bank -> update the manifest."""
    if not m.get("fp"):
        m["nonce"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        q = effective_quotas()
        m["fp"] = sha16("|".join(f"{g}:{q[g]}" for g in GENRE_ORDER)
                        + f"|{SEED}|{m['nonce']}|{int(pilot)}")[:10]
        log(f"lineage: fp {m['fp']} (nonce {m['nonce']})")
    fp = m["fp"]
    rows = []
    if m.get("files") and not pilot:
        for name in sorted(m["files"]):
            try:
                p = hf_pull(f"{DATA_SUB}/{name}")
            except Exception as e:
                log(f"❌ flush: cannot pull banked {name} ({str(e)[:70]}) — "
                    "aborting BEFORE overwriting anything; rerun to retry")
                return False
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line: rows.append(json.loads(line))
            log(f"flush: pulled {name} ({m['files'][name]:,} eps)")
    n_old = len(rows)
    for g in GENRE_ORDER:
        rows.extend(state["eps"].get(g, []))
    if not rows:
        log("flush: no episodes at all — nothing to bank")
        return False
    if len(rows) < n_old:
        log("❌ flush: merged row count shrank — refusing (bug guard)")
        return False
    val_rows = [ep for ep in rows if is_val(ep)]
    train = [ep for ep in rows if not is_val(ep)]
    # token estimate (sampled over train rows; honest +-10%)
    ntok = 0
    sample = train[:SAMPLE_TOK]
    for ep in sample:
        ntok += est_tokens(ep, tok)
    if sample and len(train) > len(sample):
        ntok = int(ntok * len(train) / len(sample))
    log(f"flush: {len(train):,} train / {len(val_rows):,} val episodes · "
        f"approx {ntok/1e6:.0f}M tokens")
    kept = {}
    for i in range(0, len(train), SHARD_ROWS):
        chunk = train[i:i + SHARD_ROWS]
        name = f"episodes_{fp}_{i // SHARD_ROWS:04d}.jsonl"
        p = TMP_ROOT / name
        write_jsonl(chunk, p)
        if bank(p, f"{DATA_SUB}/{name}", tree, desc="⬆ shard"):
            kept[name] = len(chunk)
        else:
            log(f"⚠ {name} not banked — kept locally for the retry")
        p.unlink(missing_ok=True)
    val_name = ""
    if val_rows:
        val_name = f"val_{fp}.jsonl"
        p = TMP_ROOT / val_name
        write_jsonl(val_rows, p)
        if bank(p, f"{DATA_SUB}/{val_name}", tree, desc="⬆ val"):
            pass
        p.unlink(missing_ok=True)
    m["files"] = kept
    m["episodes"] = len(train)
    m["val"] = len(val_rows)
    m["val_file"] = val_name
    counts = {}
    for ep in train:
        g = (ep.get("meta") or {}).get("genre", "?")
        counts[g] = counts.get(g, 0) + 1
    m["genre_counts"] = counts
    m["token_est"] = int(ntok)
    m["complete"] = bool(complete)
    m["built"] = datetime.now(timezone.utc).isoformat()
    m["scan"] = dict(state["scan"])
    m["drop"] = dict(state["drop"])
    save_manifest(m, tree)
    return True

# ═════════════════════════════ modes: probe / stat ════════════════════════════
def probe():
    """print first-row schemas of every source, then exit (run me first)."""
    for cfg in CORAG_CFGS:
        try:
            row = next(stream_corag(cfg))
            log(f"probe {CORAG_ID}/{cfg}: keys={list(row.keys())}")
            for k, v in row.items():
                log(f"    {k}: {head(v, 110)}")
        except Exception as e:
            log(f"probe {CORAG_ID}/{cfg} FAILED: {str(e)[:110]}")
    for g in ("instrument", "consequence", "code"):
        try:
            row = next(stream_nemo(g))
            log(f"probe {NEMO_ID}/{g}: keys={list(row.keys())}")
            for k, v in row.items():
                log(f"    {k}: {head(v, 110)}")
        except Exception as e:
            log(f"probe {NEMO_ID}/{g} FAILED: {str(e)[:110]}")

def stat():
    m = load_banked_manifest()
    if not m:
        log("no sft_manifest.json on the repo — nothing banked yet")
        return
    log(f"built {m.get('built')} · fp {m.get('fp')} · complete={m.get('complete')}")
    log(f"episodes {m.get('episodes', 0):,} · val {m.get('val', 0):,} · "
        f"tokens approx {m.get('token_est', 0)/1e6:.0f}M")
    for g, c in (m.get("genre_counts") or {}).items():
        log(f"    {g:12s} {c:8,}")
    for f, n in (m.get("files") or {}).items():
        log(f"    shard {f} ({n:,} eps)")
    if m.get("val_file"):
        log(f"    val   {m['val_file']} ({m.get('val', 0):,} eps)")
    drops = m.get("drop") or {}
    if drops:
        log(f"drops total {sum(v[0] for v in drops.values()):,}:")
        for k, (n, g) in sorted(drops.items(), key=lambda kv: -kv[1][0])[:24]:
            log(f"    {n:8,}  {k}  ({g})")
    log("next -> sft/sft_train.py (Phase 5 training, GPU-first)")

# ═══════════════════════════════════ MAIN ═════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="print schemas, exit")
    ap.add_argument("--stat", action="store_true", help="summarize banked data")
    ap.add_argument("--pilot", type=int, default=0,
                    help=">0 = small slice of N episodes, no val")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the banked manifest: fresh lineage")
    ap.add_argument("--obs-source", choices=["bundle", "corpus"],
                    default="bundle",
                    help="bundle = subanswer evidence (now) · corpus = real "
                         "Phase-4 chunks (later)")
    args = ap.parse_args()
    log(f"════════════ SFT BUILD · seed {SEED} · pilot={args.pilot or 'off'} "
        f"════════════")
    got_tok = get_token()
    if got_tok: os.environ["HF_TOKEN"] = got_tok
    if args.probe:
        ensure_libs(["huggingface_hub>=0.23", "datasets>=2.20"])
        probe(); return
    if args.stat:
        ensure_libs(["huggingface_hub>=0.23"])
        stat(); return
    ensure_libs(["huggingface_hub>=0.23", "datasets>=2.20",
                 "transformers>=4.44", "tqdm"])
    pick_root()
    if args.obs_source == "corpus":
        log("⚠ --obs-source corpus needs the Phase-4 index (not built yet) — "
            "building evidence bundles instead")
    tree = repo_tree()
    m = None
    if not args.pilot and not args.no_resume:
        m = load_banked_manifest()
    if m:
        log(f"resume: {m.get('episodes', 0):,} train + {m.get('val', 0):,} val "
            f"episodes banked · fp {m.get('fp')}")
    else:
        m = dict(files={}, genre_counts={}, scan={}, drop={}, fp="",
                 nonce="", val_file="", episodes=0, val=0)
        log("lineage: FRESH manifest" + (" (pilot)" if args.pilot else ""))
    quotas = effective_quotas()
    total_q = sum(quotas.values())
    if args.pilot:
        for g in GENRE_ORDER:
            quotas[g] = max(4, quotas[g] * args.pilot // total_q)
        log("pilot quotas: " + " · ".join(f"{g}:{quotas[g]}"
                                          for g in GENRE_ORDER))
    remaining = {}
    for g in GENRE_ORDER:
        done = int((m.get("genre_counts") or {}).get(g, 0))
        q = quotas[g]
        # val removal takes ~1.1%; treat >=97% as quota-satisfied (no endless
        # top-up on every rerun)
        remaining[g] = 0 if done >= int(q * 0.97) else max(0, q - done)
    log("plan: " + " · ".join(f"{g}:{remaining[g]}" for g in GENRE_ORDER) +
        f" · total {sum(remaining.values()):,} to build")
    state = dict(eps={}, seen=set(), scan=dict(m.get("scan") or {}),
                 drop=dict(m.get("drop") or {}))
    # rehydrate the dedup set from banked shards (resume honesty)
    if m.get("files") and not args.pilot:
        for name in sorted(m["files"]):
            try:
                p = hf_pull(f"{DATA_SUB}/{name}")
                n0 = len(state["seen"])
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        try:
                            state["seen"].add(sha16(question_of(json.loads(line))))
                        except Exception:
                            pass
                log(f"resume: dedup set +{len(state['seen']) - n0:,} hashes "
                    f"from {name}")
            except Exception as e:
                log(f"⚠ resume: {name} unreachable ({str(e)[:60]}) — "
                    "duplicate-questions across sessions possible")
    if sum(remaining.values()) <= 0 and m.get("episodes"):
        stat()
        log("already at quota — nothing to build (use --no-resume to rebuild)")
        return
    tok = tokenizer()
    sets = load_eval_ngrams()
    exhausted = []
    for g in GENRE_ORDER:
        if remaining[g] <= 0: continue
        got = produce(g, remaining[g], sets, state, tok)
        if got < remaining[g]:
            exhausted.append(g)
            log(f"⚠ {g}: stream ended before quota ({got:,}/{remaining[g]:,}) "
                "— continuing, rerun later to top up")
    complete = (not exhausted) and (not args.pilot)
    ok = flush(state, m, tree, tok, bool(args.pilot), complete) \
        if (state["eps"] or m.get("files")) else False
    h = (time.time() - T0) / 3600
    log("══════════ SFT BUILD SUMMARY ══════════")
    log("    genres · " + " · ".join(
        f"{g}:{m.get('genre_counts', {}).get(g, 0):,}" for g in GENRE_ORDER))
    log(f"    episodes {m.get('episodes', 0):,} · val {m.get('val', 0):,} · "
        f"tokens approx {m.get('token_est', 0)/1e6:.0f}M · fp {m.get('fp')}")
    n_drop = sum(v[0] for v in (m.get("drop") or {}).values())
    log(f"    dropped {n_drop:,} (reason map in the manifest) · "
        f"complete={m.get('complete')}")
    log(f"    elapsed {h:.1f}h · flush={'ok' if ok else 'FAILED/partial'}")
    if m.get("episodes"):
        log("next -> sft/sft_train.py (Phase 5 training on the GPUs)")
    else:
        log("⚠ nothing banked — check the drop reasons in the manifest")

if __name__ == "__main__":
    main()




