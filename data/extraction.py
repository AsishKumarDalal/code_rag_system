#!/usr/bin/env python3
"""
================================================================================
 data/extraction.py  —  Stage 1: banked archives  →  canonical docs.jsonl shards
================================================================================
 Input : the archives you banked in HF dataset repo (data/oss_repos/*.tar.gz,
         data/md_docs/*.tar.gz, data/stack/*.parquet, data/fineweb/*.parquet,
         data/wiki/*.bz2 …)  — or a local folder (SOURCE_MODE='local')
 Output: docs/shard_00001.jsonl.gz …  (one JSON object per line, canonical schema)
         uploaded back to the same HF dataset repo, local copies deleted → disk stays flat

 Run (Kaggle / local):
     %run data/extraction.py          # or: python3 data/extraction.py
     PILOT_LIMIT=3 %run data/extraction.py     # test on 3 archives first

 Schema of one output record (this is the contract for all later stages):
   id, content_sha256, source_path, origin, kind, lang, code_lang, parse_ok,
   symbols, license, title, headings, text, bytes, created, extra
================================================================================
"""
import os, re, io, sys, json, gzip, time, shutil, tarfile, hashlib, threading
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise SystemExit("Run first:  pip install -U 'huggingface_hub>=0.23'")
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# ============================================================== CONFIG ======
HF_REPO   = "sddfhy546y/code_rag_dataset"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")   # Kaggle: Add-ons ▸ Secrets ▸ 'HF_TOKEN' — never hardcode
OUT_PREFIX = "docs"                 # uploaded as  docs/shard_00001.jsonl.gz
SHARD_MB   = 200                    # uncompressed size of one output shard
MAX_DOC_BYTES = 4 * 1024 * 1024     # skip files bigger than this (logs, dumps, blobs)
MIN_DOC_CHARS = 40                  # skip near-empty documents
DEDUP_EXACT  = True                 # drop byte-identical documents (content hash)
CHUNK_CHARS   = 6000                # max chars per emitted doc (splits huge pages)

SOURCE_MODE = "hf"                  # "hf" (read from HF repo) | "local"
LOCAL_ROOT  = "./corpus"            # used when SOURCE_MODE='local'
INCLUDE_PREFIXES = ["data/oss_repos/", "data/md_docs/", "data/local/",
                    "data/stack/", "data/fineweb/"]
SKIP_PREFIXES   = []                # e.g. ["data/wiki/"] if you skip the dump
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".bmp",
            ".mp4", ".mp3", ".wav", ".mov", ".mkv", ".avi", ".flac", ".ogg",
            ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".whl", ".jar",
            ".class", ".pyc", ".o", ".a", ".lib", ".obj", ".pdb", ".zip", ".ico"}
WIKI_BZ2 = True                     # parse the Wikipedia dump too
PILOT_LIMIT = 0                    # 0 = everything; e.g. 3 for a quick test

if Path("/kaggle").exists():
    ROOT = "/tmp/code_rag_extract"; VISIBLE = Path("/kaggle/working/code_rag_extract")
else:
    ROOT = "./_stage1_extract";      VISIBLE = Path(ROOT)
ROOT = os.environ.get("EXTRACT_ROOT", ROOT)
TMP  = Path(ROOT) / "tmp"; TMP.mkdir(parents=True, exist_ok=True)
LOGD = Path(ROOT) / "logs"; LOGD.mkdir(parents=True, exist_ok=True)
MAN_PATH = Path(ROOT) / "docs_manifest.json"
VISIBLE.mkdir(parents=True, exist_ok=True)

# ============================================================== UTILS ======
def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def fmt(n):
    x, u = float(n), ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while x >= 1024 and i < 4:
        x /= 1024; i += 1
    return f"{x:,.1f} {u[i]}"
def get_token():
    t = os.environ.get("HF_TOKEN")
    if t: return t.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
    except Exception: pass
    return HF_TOKEN.strip()

# ============================================================== MANIFEST ===
MAN = {"created": now(), "done": {}, "inflight": None, "shard_idx": 0,
       "totals": {"docs": 0, "bytes": 0, "archives": 0}}
_LOCK = threading.Lock()

def man_save(api=None):
    tmp = MAN_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(MAN, indent=1))
    tmp.replace(MAN_PATH)
    try:
        (VISIBLE / "docs_manifest.json").write_text(json.dumps(MAN, indent=1))
    except Exception:
        pass
    if api:
        try:
            api.upload_file(path_or_fileobj=str(MAN_PATH), path_in_repo="docs_manifest.json",
                            repo_id=HF_REPO, repo_type="dataset", commit_message=f"extract manifest {now()}")
        except Exception as e:
            log(f"(manifest push failed: {str(e)[:70]})")

def man_load(api):
    global MAN
    if not MAN_PATH.exists():
        try:
            hf_hub_download(repo_id=HF_REPO, filename="docs_manifest.json",
                            repo_type="dataset", local_dir=str(MAN_PATH.parent))
            MAN = json.loads(MAN_PATH.read_text())
            log(f"manifest recovered from HF: {len(MAN.get('done',{}))} archive(s) done")
        except Exception:
            log("fresh start (no previous manifest)")
    else:
        MAN = json.loads(MAN_PATH.read_text())
        log(f"manifest loaded: {len(MAN.get('done',{}))} archive(s) done")

# ============================================================== HF =========
def api_client():
    a = HfApi(token=get_token())
    a.create_repo(HF_REPO, repo_type="dataset", exist_ok=True, private=True)
    return a

def repo_files(api):
    """path → size for every file in the repo (not just data/)."""
    out = {}
    try:
        for t in api.list_repo_tree(repo_id=HF_REPO, repo_type="dataset", recursive=True):
            p = t.path if hasattr(t, "path") else t["path"]
            s = getattr(t, "size", None) or (t.get("size") if isinstance(t, dict) else None)
            out[p] = int(s or 0)
    except AttributeError:
        for p in api.list_repo_files(repo_id=HF_REPO, repo_type="dataset"):
            out[p] = 0
    return out

def fetch_to_local(repo_path, api):
    """Download one repo file into TMP, return local path."""
    dest_dir = TMP / "dl"
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / repo_path
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0:
        return local
    hf_hub_download(repo_id=HF_REPO, filename=repo_path, repo_type="dataset",
                    local_dir=str(dest_dir))
    return local

def upload_file(local_path, repo_path, api):
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=repo_path,
                    repo_id=HF_REPO, repo_type="dataset", commit_message=f"{repo_path} {now()}")

# ============================================================== SHARDS =====
class ShardWriter:
    """Buffers JSONL records, rotates at SHARD_MB, uploads + deletes each shard."""

    def __init__(self, api, start_idx):
        self.api, self.idx = api, start_idx
        self.buf, self.raw = [], 0
        self.limit = SHARD_MB * 1024 * 1024
        self.shards = []

    def add(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        self.buf.append(line); self.raw += len(line)
        if self.raw >= self.limit:
            self.rotate()

    def rotate(self):
        if not self.buf:
            return None
        name = f"{OUT_PREFIX}/shard_{self.idx:05d}.jsonl.gz"
        path = TMP / f"shard_{self.idx:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
            f.writelines(self.buf)
        upload_file(path, name, self.api)
        self.shards.append(name)
        self.idx += 1
        self.buf, self.raw = [], 0
        path.unlink(missing_ok=True)
        log(f"   ⬆ shard uploaded: {name} ({fmt(path.stat().st_size if path.exists() else 0)} on disk, deleted)")
        return name

# ============================================================== TEXT =======
CODE_LANGS = {
    ".py": "python", ".ipynb": "jupyter", ".js": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".go": "go",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".c": "c", ".h": "c", ".cc": "cpp",
    ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".swift": "swift", ".m": "objc",
    ".mm": "objc", ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
    ".sql": "sql", ".pl": "perl", ".lua": "lua", ".dart": "dart", ".r": "r",
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang", ".clj": "clojure",
    ".hs": "haskell", ".jl": "julia", ".vue": "vue", ".svelte": "svelte",
    ".sol": "solidity", ".zig": "zig", ".nim": "nim", ".proto": "protobuf",
}
DOC_EXT = {".md", ".markdown", ".mdx", ".rst", ".txt", ".adoc", ".tex", ".org"}
HTML_EXT = {".html", ".htm", ".xhtml"}

SYMBOL_PATTERNS = {
    "python":     [r"^\s*(?:async\s+)?def\s+(\w+)", r"^\s*class\s+(\w+)"],
    "javascript": [r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)",
                   r"^\s*(?:export\s+)?class\s+(\w+)", r"^\s*const\s+(\w+)\s*=\s*(?:async\s*)?\("],
    "typescript": [r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)",
                   r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
                   r"^\s*(?:export\s+)?interface\s+(\w+)", r"^\s*(?:export\s+)?type\s+(\w+)"],
    "java":       [r"^\s*(?:public|private|protected)?\s*(?:final\s+)?class\s+(\w+)",
                   r"^\s*(?:public|private|protected)\s+\w[\w<>\[\]]*\s+(\w+)\s*\("],
    "c":          [r"^\s*(?:static\s+)?\w[\w\s\*]*\s+\**(\w+)\s*\([^;]*\)\s*\{",
                   r"^\s*struct\s+(\w+)", r"^\s*#define\s+(\w+)"],
    "cpp":        [r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+(\w+)",
                   r"^\s*\w[\w\s\*:<>,]*\s+\**(\w+)\s*\([^;]*\)\s*\{"],
    "go":         [r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)", r"^\s*type\s+(\w+)\s+(?:struct|interface)"],
    "rust":       [r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)"],
    "ruby":       [r"^\s*def\s+(\w+)", r"^\s*class\s+(\w+)", r"^\s*module\s+(\w+)"],
    "php":        [r"^\s*(?:public|private|protected)?\s*(?:static\s+)?function\s+(\w+)",
                   r"^\s*(?:abstract\s+)?class\s+(\w+)"],
    "csharp":     [r"^\s*(?:public|private|protected)?\s*(?:sealed\s+)?class\s+(\w+)",
                   r"^\s*(?:public|private|protected)\s+\w[\w<>\[\]]*\s+(\w+)\s*\("],
    "kotlin":     [r"^\s*(?:fun\s+(\w+))", r"^\s*class\s+(\w+)", r"^\s*object\s+(\w+)"],
    "swift":      [r"^\s*(?:public\s+|private\s+)?func\s+(\w+)", r"^\s*class\s+(\w+)"],
    "sql":        [r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|INDEX|FUNCTION)\s+(\w+)"],
}
_COMPILED = {lang: [re.compile(p, re.M | re.I) for p in pats]
             for lang, pats in SYMBOL_PATTERNS.items()}

def extract_symbols(text, lang):
    pats = _COMPILED.get(lang or "")
    if not pats: return []
    out = []
    for p in pats:
        for m in p.finditer(text):
            n = m.group(1)
            if n and n not in out and len(out) < 80:
                out.append(n)
    return out

def md_structure(text):
    """Title + heading trail of a markdown/rst document."""
    heads, title = [], ""
    for line in text.splitlines()[:4000]:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        if m:
            h = m.group(2).strip()
            heads.append(h)
            if not title and len(m.group(1)) == 1:
                title = h
    if not title and heads:
        title = heads[0]
    return title, heads[:40]

def strip_html(text):
    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", text)
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'"))
    return re.sub(r"[ \t]+", " ", text), title

STOP = {"en": {"the","and","of","to","in","is","that","for","with","this","are","on","as","it"},
        "es": {"el","la","de","que","y","en","los","del","se","las","por","un","para","con"},
        "fr": {"le","la","les","de","et","des","en","un","une","du","que","pour","dans","avec"},
        "de": {"der","die","das","und","ist","von","zu","mit","den","für","auf","ein","eine"},
        "hi": {"और","का","की","के","में","से","है","यह","वह","को","पर","लिए","एक"},
        "zh": {"的","了","和","是","在","不","有","这","那","与","及","对"}}

def guess_lang(text):
    sample = text[:4000].lower()
    best, best_score = "en", 0
    for lang, words in STOP.items():
        score = sum(sample.count(" " + w + " ") for w in words)
        if score > best_score:
            best, best_score = lang, score
    return best, round(min(1.0, best_score / 40.0), 3)

def decode_text(data):
    """bytes → text, or None if it isn't text we care about."""
    if b"\x00" in data[:2000]:
        return None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            t = data.decode(enc)
            if "\x00" in t[:2000]:
                return None
            return t
        except UnicodeDecodeError:
            continue
    return None

def normalize(text):
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()

def sha(text):
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()

# ============================================================== RECORDS ====
SEEN = set()          # content hashes for exact dedup (session-local)

def make_record(path, text, origin, kind=None, extra=None):
    """Canonical record for one document/file."""
    text = normalize(text)
    if len(text) < MIN_DOC_CHARS:
        return None
    ext = Path(path).suffix.lower()
    code_lang = CODE_LANGS.get(ext)
    title, headings = "", []
    if ext in HTML_EXT:
        text, title = strip_html(text)
    elif ext in DOC_EXT or ext == "":
        title, headings = md_structure(text)
    if not title:
        if code_lang:                       # code: the file name is the best title
            title = Path(path).name
        else:
            first = text.splitlines()[0].strip() if text.splitlines() else ""
            title = first[:120] if len(first) > 3 else ""
    kind = kind or ("code" if code_lang else ("doc" if ext in DOC_EXT or ext in HTML_EXT else "text"))
    lang, lang_score = guess_lang(text)
    h = sha(text)
    if DEDUP_EXACT:
        if h in SEEN:
            return None
        SEEN.add(h)
    rec = {
        "id": "sha256:" + h,
        "content_sha256": h,
        "source_path": path,
        "origin": origin,
        "kind": kind,
        "lang": lang,
        "lang_score": lang_score,
        "code_lang": code_lang,
        "parse_ok": True,
        "symbols": extract_symbols(text, code_lang) if code_lang else [],
        "license": (extra or {}).get("license", ""),
        "title": title,
        "headings": headings,
        "text": text,
        "bytes": len(text.encode("utf-8", "ignore")),
        "created": now(),
        "extra": {k: v for k, v in (extra or {}).items() if k != "license"},
    }
    return rec

def split_long(text, limit=CHUNK_CHARS):
    """Split very long pages into paragraph-aligned chunks."""
    if len(text) <= limit:
        return [text]
    out, cur = [], []
    for para in text.split("\n\n"):
        if cur and sum(len(x) for x in cur) + len(para) > limit:
            out.append("\n\n".join(cur)); cur = []
        cur.append(para)
    if cur:
        out.append("\n\n".join(cur))
    return out

# ============================================================== READERS ====
def read_tar(path, origin):
    """Yield records from a .tar/.tar.gz of a repo."""
    try:
        tf = tarfile.open(path, "r|gz")
    except tarfile.TarError:
        tf = tarfile.open(path, "r:*")
    with tf:
        for m in tf:
            if not m.isfile() or m.size > MAX_DOC_BYTES:
                continue
            name = m.name
            if Path(name).suffix.lower() in SKIP_EXT:
                continue
            try:
                data = tf.extractfile(m).read()
            except Exception:
                continue
            text = decode_text(data)
            if not text:
                continue
            parts = split_long(text)
            for i, part in enumerate(parts):
                p = name + (f"#part{i}" if len(parts) > 1 else "")
                r = make_record(p, part, origin)
                if r:
                    yield r

def read_parquet(path, origin):
    """Yield records from an HF parquet shard (The Stack / FineWeb style)."""
    if pq is None:
        log("   (pyarrow not installed → skipping parquet)")
        return
    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    text_col = next((c for c in ("content", "text", "code", "document", "page_content")
                     if c in cols), None)
    if not text_col:
        log(f"   (no text column in {Path(path).name}; cols={list(cols)[:8]})")
        return
    path_col   = next((c for c in ("path", "file_path", "filename", "url", "source") if c in cols), None)
    lic_col    = next((c for c in ("license", "licence") if c in cols), None)
    lang_col   = next((c for c in ("language", "lang") if c in cols), None)
    for batch in pf.iter_batches(batch_size=1000):
        d = batch.to_pydict()
        for i in range(len(d[text_col])):
            text = d[text_col][i]
            if not isinstance(text, str) or len(text) < MIN_DOC_CHARS:
                continue
            src = str(d[path_col][i]) if path_col and d.get(path_col) else Path(path).name
            extra = {}
            if lic_col and d.get(lic_col):  extra["license"] = str(d[lic_col][i])
            if lang_col and d.get(lang_col): extra["src_lang"] = str(d[lang_col][i])
            parts = split_long(text)
            for k, part in enumerate(parts):
                p = src + (f"#part{k}" if len(parts) > 1 else "")
                r = make_record(p, part, origin, extra=extra)
                if r:
                    yield r

def read_wiki_bz2(path, origin):
    """Stream the Wikipedia XML dump → article records."""
    import bz2
    title, buf, in_text = "", [], False
    n = 0
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("<title>"):
                title = line[7:line.find("</title>")].strip()
            elif "<text" in line and ">" in line:
                in_text = True
                buf = [line[line.find(">") + 1:]]
            elif in_text:
                if "</text>" in line:
                    buf.append(line[:line.find("</text>")])
                    body = "".join(buf)
                    body = re.sub(r"\{\{[^{}]*\}\}", " ", body)      # templates
                    body = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", body)
                    body = re.sub(r"<[^>]+>", " ", body)
                    parts = split_long(body)
                    for k, part in enumerate(parts):
                        p = f"wikipedia/{title}" + (f"#part{k}" if len(parts) > 1 else "")
                        r = make_record(p, part, origin, kind="doc",
                                        extra={"article": title})
                        if r:
                            yield r; n += 1
                    in_text, buf = False, []
                else:
                    buf.append(line)
    log(f"   (wikipedia articles: {n})")

def read_plain(path, origin):
    """Any single text file (e.g. an .md dumped directly)."""
    data = Path(path).read_bytes()
    text = decode_text(data)
    if not text:
        return
    parts = split_long(text)
    for k, part in enumerate(parts):
        p = Path(path).name + (f"#part{k}" if len(parts) > 1 else "")
        r = make_record(p, part, origin)
        if r:
            yield r

READERS = {
    ".tar.gz": read_tar, ".tgz": read_tar, ".tar": read_tar, ".tar.bz2": read_tar,
    ".parquet": read_parquet, ".jsonl": read_plain, ".json": read_plain,
    ".md": read_plain, ".txt": read_plain, ".bz2": read_wiki_bz2,
}

# ============================================================== MAIN =======
def main():
    api = api_client()
    man_load(api)
    files = repo_files(api)
    if SOURCE_MODE == "local":
        targets = []
        base = Path(LOCAL_ROOT)
        for p in base.rglob("*"):
            if p.is_file():
                targets.append(str(p))
    else:
        targets = [p for p in files
                   if any(p.startswith(pre) for pre in INCLUDE_PREFIXES)
                   and not any(p.startswith(pre) for pre in SKIP_PREFIXES)]
        targets.sort()
    done = set(MAN.get("done", {}).keys())
    todo = [p for p in targets if p not in done]
    if PILOT_LIMIT:
        todo = todo[:PILOT_LIMIT]
    total_bytes = sum(files.get(p, 0) for p in todo) if SOURCE_MODE == "hf" else 0
    log(f"plan: {len(todo)} archive/file(s) to process, ~{fmt(total_bytes)} "
        f"({len(done)} already done)")
    if MAN.get("inflight"):
        log(f"⚠ previous run was interrupted on: {MAN['inflight']}")

    writer = ShardWriter(api, MAN.get("shard_idx", 0))
    t0 = time.time(); done_bytes = 0; n_docs = 0
    for src in todo:
        name = Path(src).name
        key  = str(src)
        ext  = "".join(Path(src).suffixes[-2:]).lower()
        reader = READERS.get(ext)
        if not reader and WIKI_BZ2 and ext == ".bz2":
            reader = read_wiki_bz2
        if not reader:
            log(f"→ skip (no reader): {name}")
            continue
        try:
            local = fetch_to_local(src, api) if SOURCE_MODE == "hf" else Path(src)
            log(f"→ processing {name} ({fmt(local.stat().st_size)})")
            MAN["inflight"] = {"src": key, "shards": [], "idx_start": writer.idx}
            man_save(api)
            n = 0
            for rec in reader(local, key):
                writer.add(rec); n += 1
            shards = writer.rotate()                     # close the archive's shard
            if shards:
                MAN["inflight"]["shards"].append(shards)
            MAN["done"][key] = {"records": n, "shards": MAN["inflight"]["shards"]}
            MAN["shard_idx"] = writer.idx
            MAN["totals"]["docs"] += n
            MAN["totals"]["bytes"] += local.stat().st_size
            MAN["totals"]["archives"] += 1
            MAN["inflight"] = None
            man_save(api)
            n_docs += n
            done_bytes += local.stat().st_size
            if SOURCE_MODE == "hf":
                local.unlink(missing_ok=True)
            el = time.time() - t0
            log(f"   ✓ {n} docs | {fmt(done_bytes)} processed "
                f"| {fmt(done_bytes/el)}/s | docs {MAN['totals']['docs']:,}")
        except Exception as e:
            log(f"❌ {name} failed: {str(e)[:120]}")
            MAN.setdefault("failed", {})[key] = str(e)[:200]
            man_save(api)
    writer.rotate()
    man_save(api)
    log(f"DONE — {MAN['totals']['docs']:,} docs from {MAN['totals']['archives']} archive(s) "
        f"→ {OUT_PREFIX}/shard_*.jsonl.gz")
    log("next → data/chunking.py (RAG chunks) then data/cpt_prep.py (CPT slice)")

if __name__ == "__main__":
    main()
