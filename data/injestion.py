#!/usr/bin/env python3
"""
================================================================================
 data/injestion.py  —  Stage 0 corpus acquisition ("relay fetcher")
================================================================================
 What it does
   • Downloads code + docs sources (Wikipedia, StackExchange, GitHub repos,
     FineWeb-Edu replay slice, The Stack v2 code slices).
   • Keeps only ~10 GB on local disk: reaches FLUSH_GB → parallel-uploads
     everything staged to your HF dataset repo → deletes local copies.
   • Fully resumable across sessions: manifest.json is mirrored to the HF repo,
     and on startup every item is re-verified against the actual repo contents
     (HF tree = ground truth, not local disk).
   • Fail-soft: one bad source logs 'failed' and the run continues.
     Rerun the same command later — failed units auto-retry, finished units skip.
   • Live progress: banked-on-HF vs planned GB, session speed, ETA.

 Run (local machine, python 3.9+):
     python data/injestion.py
 Run (Kaggle):
     %run data/injestion.py        # or paste whole file into one cell
     # ROOT auto-switches to /kaggle/working/code_rag_fetch

 Before first run, edit if needed:
   • CONFIG block: HF_TOKEN (rotate regularly / use Kaggle Secret),
     UA contact email, FLUSH_GB, source lists.
================================================================================
"""
import os, re, sys, io, json, time, shutil, tarfile, threading, subprocess
from datetime import datetime, timezone
from pathlib import Path
from fnmatch import fnmatch
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise SystemExit("Run first:  pip install -U 'huggingface_hub>=0.23'")
try:                                   # tqdm ships with huggingface_hub, but be safe
    from tqdm.auto import tqdm as tqdm_auto
except ImportError:
    tqdm_auto = None

# ============================================================ CONFIG ========
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # Kaggle: Add-ons ▸ Secrets ▸ 'HF_TOKEN' — never hardcode
HF_REPO         = "sddfhy546y/code_rag_dataset"              # dataset repo (type=dataset)
REMOTE_MANIFEST = "manifest.json"                            # mirrored manifest path on HF
FLUSH_GB        = 10.0          # flush to HF when staging reaches this
SPLIT_GB        = 5.0           # single files above this get sharded .partNNNN
DL_WORKERS      = 6             # threads inside big downloads/untar (adiaftool: light use)
UP_WORKERS      = 3             # parallel uploads in a flush batch
SHA256_VERIFY   = False         # True → sha256 verified manifest entries (slower)
SYNC_EVERY_SEC  = 150           # manifest→HF how often while work runs

# A descriptive User-Agent carrying a real contact is *required* by
# dumps.wikimedia.org (without it → HTTP 403) and appreciated by archive.org.
UA = "code-rag-corpus-fetcher/0.3 (LLM corpus assembly; contact: add-your-email@example.com)"

# ---- source lists (edit freely) ---------------------------------------------
WIKI_URL = ("https://dumps.wikimedia.org/enwiki/latest/"
            "enwiki-latest-pages-articles-multistream.xml.bz2")
SE_BASE  = "https://archive.org/download/stackexchange/"
SE_INCLUDE = ["stackoverflow.com-Posts", "softwareengineering", "codereview"]
SE_EXCLUDE = ["Users", "Badges", "Comments", "Votes", "PostHistory", "PostLinks"]

OSS_REPOS = ["python/cpython", "rust-lang/rust", "golang/website", "django/django",
             "pallets/flask", "numpy/numpy", "pandas-dev/pandas", "pytorch/pytorch",
             "tensorflow/tensorflow", "keras-team/keras", "kubernetes/website",
             "postgres/postgres", "nginx/nginx", "nodejs/node", "openssl/openssl",
             "llvm/llvm-project", "apache/spark", "scikit-learn/scikit-learn",
             "sympy/sympy", "docker/compose", "redis/redis", "godotengine/godot"]

# ---- MARKDOWN sources: docs-heavy repos (mostly .md files → ideal for RAG) ----
MD_DOC_REPOS = ["rust-lang/book",                         # The Rust Book
                "MicrosoftDocs/azure-docs",               # huge Azure docs (.md)
                "microsoft/TypeScript-Website",
                "microsoft/vscode-docs",
                "microsoft/Web-Dev-For-Beginners",
                "vuejs/docs", "vitejs/vite", "sveltejs/svelte",
                "angular/angular", "vercel/next.js", "facebook/react",
                "getify/Functional-Light-JS"]

# ---- YOUR OWN local code + markdown folders (optional) -----------------------
#  [{"path": "/home/me/my_corpus", "name": "my_code", "exts": [".py", ".md"]}, ...]
LOCAL_SOURCES = []

# ---- ANY extra HF datasets (code or markdown) -------------------------------
#  {"repo": "org/dataset", "allow": ["**/*.md"], "exts": [".md"], "key": "md_docs"}
EXTRA_HF_SOURCES = [
    # {"repo": "HuggingFaceFW/fineweb-edu", "allow": ["sample/10BT/*"], "key": "more_web"},
    # {"repo": "bigcode/the-stack-smol", "allow": ["data/python/*"], "key": "stack_smol"},
]

# ---- giant-file handling -----------------------------------------------------
BIG_FILE_GB      = 6.0    # files bigger than this are streamed part-by-part
BIG_FILE_PART_GB = 2.0    # each streamed part is uploaded + deleted before the next

FINEWEB_REPO  = "HuggingFaceFW/fineweb-edu"      # replay anchor (anti-forgetting)
FINEWEB_ALLOW = ["sample/10BT/*"]
STACK_REPO    = "bigcode/the-stack-v2-dedup"     # accept license once on HF w/ your token
STACK_ALLOW   = ["data/python/*", "data/typescript/*"]   # add "data/c++/*", "data/go/*", ...

# ---- paths -------------------------------------------------------------------
if Path("/kaggle").exists():
    # Kaggle: /kaggle/working has a ~20GB quota → use the big ephemeral disk instead.
    ROOT = "/tmp/code_rag_fetch"
    VISIBLE_DIR = Path("/kaggle/working/code_rag_fetch")   # manifest copy you can see
else:
    ROOT = "./_stage0_fetch"
    VISIBLE_DIR = Path(ROOT)
ROOT = os.environ.get("STAGE0_ROOT", ROOT)                 # manual override
STAGING       = Path(ROOT) / "staging"; STAGING.mkdir(parents=True, exist_ok=True)
LOGDIR        = Path(ROOT) / "logs";    LOGDIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = Path(ROOT) / "manifest.json"
VISIBLE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================ UTILS =========
def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def fmt(n):
    x, u = float(n), ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while x >= 1024 and i < 4:
        x /= 1024; i += 1
    return f"{x:,.1f} {u[i]}"
def disk_free_gb(p="."): return shutil.disk_usage(p).free / 2**30

SESSION = {"t0": time.time(), "bytes": 0}
_SLOCK = threading.Lock()
def add_bytes(n):
    with _SLOCK: SESSION["bytes"] += n
def session_speed():
    dt = time.time() - SESSION["t0"]
    return (SESSION["bytes"] / dt) if dt > 8 else 0.0
def eta_str(sec):
    if not sec or sec <= 0: return "?"
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s"
def unpart(name): return re.sub(r"\.part\d{4}$", "", name)

# ------------------------------------------------------------ PROGRESS -----
class _MiniBar:
    """Fallback progress bar (used only if tqdm is missing). Terminal-friendly."""

    def __init__(self, total=None, desc="", initial=0, **kw):
        self.total = total or 0
        self.n = initial
        self.desc = desc
        self.post = ""
        self.last = time.time()
        self.start = time.time()

    def update(self, k=1):
        self.n += k
        now = time.time()
        if now - self.last > 1.0:
            self.last = now
            self._show()

    def _show(self):
        pct = (100 * self.n / self.total) if self.total else 0.0
        rate = self.n / max(1e-9, time.time() - self.start)
        print(f"\r{self.desc}: {pct:5.1f}%  {fmt(self.n)}  @ {fmt(rate)}/s  {self.post}",
              end="", flush=True)

    def set_postfix_str(self, s): self.post = s
    def set_description(self, s): self.desc = s
    def refresh(self): self._show()
    def close(self):
        self._show(); print()

def make_bar(total=None, desc="", initial=0, **kw):
    """Works in terminal (plain bars) and in notebooks (auto widget)."""
    if tqdm_auto is not None:
        return tqdm_auto(total=total, desc=desc, initial=initial, unit="B",
                         unit_scale=True, unit_divisor=1024, dynamic_ncols=True,
                         mininterval=0.3, **kw)
    return _MiniBar(total=total, desc=desc, initial=initial, **kw)

class ProgressReader(io.BufferedReader):
    """A real buffered file object (the Hub rejects anything else) that also
    feeds a progress bar while the bytes stream out."""

    def __init__(self, path, bar):
        self._bar = bar
        super().__init__(open(path, "rb"), buffer_size=1024 * 1024)

    def read(self, size=-1):
        b = super().read(size)
        if b: self._bar.update(len(b))
        return b

def add_banked(n):
    """Bytes successfully uploaded to HF (drives the overall corpus bar)."""
    SESSION["banked"] = SESSION.get("banked", 0) + n
    b = SESSION.get("bar")
    if b is not None:
        b.update(n)

# ============================================================ MANIFEST ======
MANIFEST  = {"created": now(), "items": {}}
_MLOCK    = threading.Lock()
_CHANGES  = [0]

def get_token():
    t = os.environ.get("HF_TOKEN")
    if t: return t.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
    except Exception:
        pass
    return HF_TOKEN.strip()

def msave():
    with _MLOCK:
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(MANIFEST, indent=1))
        tmp.replace(MANIFEST_PATH)
    try:                                   # extra copy in a place you can browse
        v = VISIBLE_DIR / "manifest.json"
        if v.resolve() != MANIFEST_PATH.resolve():
            v.write_text(json.dumps(MANIFEST, indent=1))
    except Exception:
        pass

def mset(iid, **kw):
    with _MLOCK:
        it = MANIFEST["items"].setdefault(iid, {})
        it.update(kw); it["id"] = iid; it["ts"] = now()
        _CHANGES[0] += 1

def msave_to_repo(api=None):
    try:
        api = api or HfApi(token=get_token())
        api.upload_file(path_or_fileobj=str(MANIFEST_PATH), path_in_repo=REMOTE_MANIFEST,
                        repo_id=HF_REPO, repo_type="dataset", commit_message=f"manifest {now()}")
    except Exception as e:
        log(f"(manifest→HF push failed, will retry: {str(e)[:90]})")

def sync_worker(interval=SYNC_EVERY_SEC):
    while True:
        time.sleep(interval)
        if _CHANGES[0]:
            msave_to_repo(); _CHANGES[0] = 0

# ============================================================ HF TREE ======
def repo_tree(api):
    """path→size of every file under data/ in the HF repo (banked corpus truth)."""
    out = {}
    try:
        for t in api.list_repo_tree(repo_id=HF_REPO, repo_type="dataset", recursive=True):
            p = t.path if hasattr(t, "path") else t["path"]
            s = getattr(t, "size", None) or (t.get("size") if isinstance(t, dict) else None)
            if p.startswith("data/"): out[p] = int(s or 0)
    except AttributeError:
        for p in api.list_repo_files(repo_id=HF_REPO, repo_type="dataset"):
            if p.startswith("data/"): out[p] = 0
    return out

def sync_from_repo(api=None):
    """Session-restart safety:
    1) no local manifest → pull it from HF repo
    2) verify every item's up_path really exists in the repo tree
    3) if repo has files but we lost the manifest → rebuild minimal items"""
    api = api or HfApi(token=get_token())
    if not MANIFEST_PATH.exists():
        try:
            hf_hub_download(repo_id=HF_REPO, filename=REMOTE_MANIFEST, repo_type="dataset",
                            local_dir=str(MANIFEST_PATH.parent))
            MANIFEST["items"] = json.loads(MANIFEST_PATH.read_text()).get("items", {})
            log(f"manifest recovered from HF ({len(MANIFEST['items'])} items)")
        except Exception:
            log("no manifest here or on HF — starting fresh")
    tree = repo_tree(api)
    fixed, demoted = 0, 0
    for iid, it in list(MANIFEST["items"].items()):
        ups = it.get("up_paths") or []
        if not ups: continue
        hits = [p for p in ups if p in tree]
        if hits and len(hits) == len(ups):
            if it.get("state") != "flushed":
                it["state"] = "flushed"; fixed += 1
            it["verified"] = now()
        elif not hits and it.get("state") == "flushed":
            if Path(it.get("dest", "zzz")).exists():
                it["state"] = "staged"; demoted += 1
            else:
                it["state"] = "limbo"; demoted += 1
    if fixed or demoted:
        log(f"repo-sync: {fixed} confirmed flushed, {demoted} demoted→re-fetch")
        msave()
    # manifest lost entirely? rebuild 'flushed' items straight from repo contents
    known = set()
    for it in MANIFEST["items"].values():
        known |= set(it.get("up_paths") or [])
    orphans = [p for p in tree if p not in known]
    if orphans and len(MANIFEST["items"]) <= 1:
        groups = {}
        for p in orphans:
            groups.setdefault(unpart(p), []).append(p)
        for base, files in groups.items():
            contiguous = files == [f"{base}.part{i:04d}" for i in range(len(files))]
            state = "flushed" if (len(files) == 1 or contiguous) else "needs-manual-check"
            mset(f"legacy::{base}", kind="legacy", key=base.split("/")[1],
                 up_paths=sorted(files) if len(files) > 1 else [ files[0] ] if files else [],
                 state=state)
            if state != "flushed":
                log(f"⚠ orphan with missing parts, review later: {base}")
        msave()
        log(f"bootstrap: rebuilt {len(groups)} item(s) from repo contents")
    return tree, api

def report(api=None, tag=""):
    api = api or HfApi(token=get_token())
    tree = repo_tree(api)
    per, tot = {}, 0
    for p, s in tree.items():
        k = p.split("/")[1]; per[k] = per.get(k, 0) + s; tot += s
    log("=== BANKED ON HF (ground truth, not local) ===")
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        log(f"   {k:<14} {fmt(v)}")
    log(f"   {'TOTAL':<14} {fmt(tot)} {tag}")
    return tot

# ============================================================ FETCHERS =====
def http_size(url):
    try:
        r = urlopen(Request(url, method="HEAD", headers={"User-Agent": UA}), timeout=30)
        n = int(r.headers.get("Content-Length", 0) or 0)
        return n or None
    except Exception:
        return None

def wget_resume(url, dest, tries=6, name=""):
    dest, exp = Path(dest), http_size(url)
    dest.parent.mkdir(parents=True, exist_ok=True)     # FIX: folder must exist first
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and (exp is None or dest.stat().st_size == exp):
        return "already-downloaded"
    t_last, t0 = time.time(), time.time()
    for attempt in range(1, tries + 1):
        pos = part.stat().st_size if part.exists() else 0
        got = 0
        bar = make_bar(total=exp, desc=f"↓ {name}", initial=pos)
        try:
            hdr = {"User-Agent": UA, **({"Range": f"bytes={pos}-"} if pos else {})}
            with urlopen(Request(url, headers=hdr), timeout=90) as r, open(part, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); add_bytes(len(b))
                    got += len(b); bar.update(len(b))
            if exp and pos + got != exp:
                raise IOError(f"size mismatch: {pos+got} != {exp}")
            os.replace(part, dest)
            bar.close()
            return "downloaded"
        except Exception as e:
            bar.close()
            log(f"   retry {attempt}/{tries} ({str(e)[:90]})")
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"gave up on: {url}")

def git_clone(url, dest, depth=1):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists(): return "cloned"
    r = subprocess.run(["git", "clone", "--depth", str(depth), "--single-branch",
                        url, str(dest)], capture_output=True, text=True)
    if r.returncode != 0:
        (LOGDIR / "clone_errors.log").open("a").write(f"{url}\n{r.stderr[-300:]}\n")
        raise RuntimeError(f"git clone failed: {url}")
    return "cloned"

def tar_local_dir(src, out_tar, exts=None, max_files=300000):
    """Pack a LOCAL folder (optionally only .py/.md files) into one tar.gz."""
    src, out_tar = Path(src), Path(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    if exts:
        exts = {e.lower() for e in exts}
    n = 0
    with tarfile.open(out_tar, "w:gz") as tf:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs
                       if d not in (".git", "__pycache__", "node_modules", ".venv",
                                    "venv", "dist", "build", ".mypy_cache")]
            for fn in files:
                p = Path(root) / fn
                if exts and p.suffix.lower() not in exts:
                    continue
                try:
                    tf.add(str(p), arcname=str(p.relative_to(src)))
                    n += 1
                except OSError:
                    pass
                if n >= max_files:
                    break
    return n

def upload_with_progress(api, local_path, repo_path, bar=None):
    """Upload a file to HF, showing a progress bar (falls back to plain upload)."""
    msg = f"{repo_path} {now()}"
    if bar is not None:
        try:
            with ProgressReader(local_path, bar) as rf:
                api.upload_file(path_or_fileobj=rf, path_in_repo=repo_path,
                                repo_id=HF_REPO, repo_type="dataset", commit_message=msg)
            return
        except Exception as e:
            log(f"   (progress-upload fallback: {str(e)[:70]})")
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=repo_path,
                    repo_id=HF_REPO, repo_type="dataset", commit_message=msg)

def wget_stream_upload(url, dest, exp, api, key, up_prefix, part_bytes=None):
    """Giant-file path: download in 2GB parts, upload each part to HF, delete it
    locally, continue. Peak disk usage = 1 part (2GB) instead of the whole file."""
    part_bytes = part_bytes or int(BIG_FILE_PART_GB * 2**30)
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    n_parts = (exp + part_bytes - 1) // part_bytes
    ups_done = []
    tree = set(repo_tree(api))                # SET (was a dict → .add() crashed)
    main_bar = make_bar(total=exp, desc=f"↓ {dest.name[:34]}")
    for i in range(n_parts):
        start = i * part_bytes
        end   = min(exp, start + part_bytes) - 1
        want  = end - start + 1
        p     = dest.with_name(dest.name + f".part{i:04d}")
        repo_p = f"{up_prefix}/{p.name}"
        if repo_p in tree:                       # already banked on HF → skip download
            ups_done.append(repo_p)
            main_bar.set_postfix_str(f"part {i+1}/{n_parts} already banked")
            main_bar.update(want)
            continue
        if p.exists() and p.stat().st_size == want:
            log(f"   part {i+1}/{n_parts} already on disk")
        else:
            ok = False
            for a in range(4):
                try:
                    hdr = {"User-Agent": UA, "Range": f"bytes={start}-{end}"}
                    with urlopen(Request(url, headers=hdr), timeout=120) as r, \
                         open(p, "wb") as f:
                        while True:
                            b = r.read(1 << 20)
                            if not b: break
                            f.write(b); add_bytes(len(b)); main_bar.update(len(b))
                    if p.stat().st_size == want:
                        ok = True; break
                    raise IOError(f"part size {p.stat().st_size} != {want}")
                except Exception as e:
                    log(f"   part {i+1} retry {a+1}: {str(e)[:80]}")
                    time.sleep(3)
            if not ok:
                main_bar.close()
                raise RuntimeError(f"part {i} of {url} failed")
        # upload this single part, then free it
        if repo_p not in tree:
            ubar = make_bar(total=want, desc=f"⬆ part {i+1}/{n_parts}")
            upload_with_progress(api, p, repo_p, ubar)
            ubar.close()
            tree.add(repo_p)
            add_banked(want)
        ups_done.append(repo_p)
        p.unlink(missing_ok=True)
        main_bar.set_postfix_str(f"part {i+1}/{n_parts} uploaded")
    main_bar.close()
    return ups_done

def tar_repo(repo_dir, tar_path):
    Path(tar_path).parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(str(repo_dir), arcname=Path(repo_dir).name)
    shutil.rmtree(repo_dir, ignore_errors=True)

def hf_fetch(repo_id, path_in_repo, to_dir):
    return hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                           repo_type="dataset", local_dir=to_dir)

def hf_list(repo_id, allow=None, deny=None, exts=None):
    """List files of an HF dataset. allow/deny = path patterns, exts = keep only
    these file extensions (e.g. ['.md'] for markdown-only, ['.py'] for code-only)."""
    api = HfApi(token=get_token())
    try:
        tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)
        it = [(t.path, getattr(t, "size", None)) for t in tree]
    except AttributeError:
        it = [(p, None) for p in api.list_repo_files(repo_id=repo_id, repo_type="dataset")]
    exts_l = {e.lower() for e in exts} if exts else None
    out = []
    for p, s in it:
        if not any(fnmatch(p, a) for a in (allow or ["*"])):  continue
        if     any(fnmatch(p, d) for d in (deny or [])):      continue
        if exts_l and Path(p).suffix.lower() not in exts_l:   continue
        out.append((p, int(s or 0)))
    return out

def split_file(src, n_bytes):
    src, size, parts = Path(src), Path(src).stat().st_size, []
    if size <= n_bytes: return parts
    with open(src, "rb") as f:
        for i, off in enumerate(range(0, size, n_bytes)):
            f.seek(off)
            with open(f"{src}.part{i:04d}", "wb") as w:
                shutil.copyfileobj(f, w, 1 << 20)
            parts.append(f"{src}.part{i:04d}")
    src.unlink()
    return parts

# ============================================================ PLAN ==========
def plan_units():
    U = []
    U.append(dict(kind="wget", key="wiki", url=WIKI_URL,
                  dest=str(STAGING / "wiki" / Path(WIKI_URL).name),
                  up=[f"data/wiki/{Path(WIKI_URL).name}"], exp=http_size(WIKI_URL)))
    try:
        md = json.load(urlopen(Request("https://archive.org/metadata/stackexchange",
                                       headers={"User-Agent": UA}), timeout=60))
        for t in md.get("files", []):
            n = t["name"]
            if n.lower().endswith(".7z") \
               and not any(d in n for d in SE_EXCLUDE) \
               and any(i in n for i in SE_INCLUDE):
                U.append(dict(kind="wget", key="stackexchange", url=SE_BASE + n,
                              dest=str(STAGING / "stackexchange" / n),
                              up=[f"data/stackexchange/{n}"],
                              exp=int(t.get("size") or 0) or None))
    except Exception as e:
        log(f"(stackexchange listing unavailable: {str(e)[:80]})")
    U.append(dict(kind="git", key="oss_repos", url="https://github.com/mdn/content.git",
                  name="mdn_content", dest=str(STAGING / "oss_repos" / "mdn_content.tar.gz"),
                  up=["data/oss_repos/mdn_content.tar.gz"], exp=None))
    for r in OSS_REPOS:
        name = r.split("/")[-1]
        U.append(dict(kind="git", key="oss_repos", url=f"https://github.com/{r}.git",
                      name=name, dest=str(STAGING / "oss_repos" / f"{name}.tar.gz"),
                      up=[f"data/oss_repos/{name}.tar.gz"], exp=None))
    # markdown-heavy documentation repos
    for r in MD_DOC_REPOS:
        name = r.split("/")[-1]
        U.append(dict(kind="git", key="md_docs", url=f"https://github.com/{r}.git",
                      name=name, dest=str(STAGING / "md_docs" / f"{name}.tar.gz"),
                      up=[f"data/md_docs/{name}.tar.gz"], exp=None))
    # your own local code + markdown folders
    for s in LOCAL_SOURCES:
        U.append(dict(kind="local", key="local", url=s["path"], name=s["name"],
                      exts=s.get("exts", []),
                      dest=str(STAGING / "local" / f"{s['name']}.tar.gz"),
                      up=[f"data/local/{s['name']}.tar.gz"], exp=None))
    # any extra HF datasets (code or markdown) you plug in
    for s in EXTRA_HF_SOURCES:
        for p, sz in hf_list(s["repo"], allow=s.get("allow"), deny=s.get("deny"),
                             exts=s.get("exts")):
            U.append(dict(kind="hf", key=s["key"], repo=s["repo"], path=p,
                          dest=str(STAGING / s["key"] / p),
                          up=[f"data/{s['key']}/{p}"], exp=sz or None))
    for p, s in hf_list(FINEWEB_REPO, allow=FINEWEB_ALLOW):
        U.append(dict(kind="hf", key="fineweb", repo=FINEWEB_REPO, path=p,
                      dest=str(STAGING / "fineweb" / p),
                      up=[f"data/fineweb/{p}"], exp=s or None))
    for p, s in hf_list(STACK_REPO, allow=STACK_ALLOW):
        U.append(dict(kind="hf", key="stack", repo=STACK_REPO, path=p,
                      dest=str(STAGING / "stack" / p),
                      up=[f"data/stack/{p}"], exp=s or None))
    return U

# ============================================================ FLUSH =========
def _reassign_parts(base_rel, part_rel_paths):
    """After splitting a big file, the unit that owned the single file now owns
    the .partNNNN files → rewrite its up_paths (or create a legacy unit)."""
    base_up = f"data/{base_rel}"
    part_ups = [f"data/{r}" for r in part_rel_paths]
    for iid, it in list(MANIFEST["items"].items()):
        if base_up in (it.get("up_paths") or []):
            it["up_paths"] = part_ups
            return
    mset(f"legacy::{base_up}", kind="legacy", key=base_up.split("/")[1],
         up_paths=part_ups, state="staged")

def flush_batch(api, force=False):
    dirs_to_clean = set()
    staged = [q for q in STAGING.rglob("*")
              if q.is_file() and not q.name.endswith(".part")]
    staging_sz = sum(q.stat().st_size for q in staged)
    if staging_sz == 0:
        return 0
    if not force and staging_sz < FLUSH_GB * 2**30:
        return 0
    log(f"FLUSH: {fmt(staging_sz)} staged → uploading to HF")
    tree_before = repo_tree(api)

    # shard oversized files; fix their owning unit's part list
    for q in staged:
        if q.stat().st_size > SPLIT_GB * 2**30:
            base_rel = Path(q.relative_to(STAGING)).with_name(unpart(q.name))
            parts = split_file(q, int(SPLIT_GB * 2**30))
            part_rel = [str(Path(base_rel).with_name(Path(p).name)) for p in parts]
            _reassign_parts(str(base_rel), part_rel)
            log(f"   split {q.name} → {len(parts)} parts of ≤{fmt(SPLIT_GB*2**30)}")

    files = {str(q.relative_to(STAGING)): q for q in STAGING.rglob("*")
             if q.is_file() and not q.name.endswith(".part")}
    uploaded = {}
    def up(rel):
        repo_path = f"data/{rel}"
        if repo_path in tree_before:            # verified already on HF: skip
            uploaded[repo_path] = True
            return rel
        size = files[rel].stat().st_size
        bar = make_bar(total=size, desc=f"⬆ {rel[-40:]}")
        try:
            upload_with_progress(api, files[rel], repo_path, bar)
        finally:
            bar.close()
        uploaded[repo_path] = True
        add_banked(size)
        return rel

    errs = []
    with ThreadPoolExecutor(UP_WORKERS) as ex:
        futs = {ex.submit(up, rel): rel for rel in files}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                errs.append((futs[fut], str(e)[:90]))

    tree_after = set(tree_before) | set(uploaded)
    freed = 0
    # 1) delete every local file whose content is confirmed on HF (frees disk)
    for rel, q in files.items():
        if f"data/{rel}" in tree_after and q.exists():
            freed += q.stat().st_size
            q.unlink()
            dirs_to_clean.add(q.parent)
    # 2) mark manifest units as flushed once all of their parts are banked
    for iid, it in list(MANIFEST["items"].items()):
        ups = it.get("up_paths") or []
        if ups and all(p in tree_after for p in ups) \
           and it.get("state") in ("staging", "staged"):
            it["state"] = "flushed"; it["verified"] = now()
    for d in sorted(dirs_to_clean, key=lambda x: -len(x.parts)):
        try:
            d.rmdir()
        except OSError:
            pass
    msave(); msave_to_repo(api); _CHANGES[0] = 0
    if errs:
        log(f"FLUSH partial: {len(uploaded)} ok, {len(errs)} failed (auto-retry next run)")
        for rel, e in errs[:4]:
            log(f"   ✗ {rel[:60]} → {e}")
    else:
        log(f"FLUSH done: {len(uploaded)} file(s) ok, freed {fmt(freed)} "
            f"(disk free {disk_free_gb(ROOT):.1f} GB)")
    return len(uploaded)

# ============================================================ ETA ===========
def eta_line(units):
    plan        = sum(u["exp"] or 0 for u in units)
    unknown     = sum(1 for u in units if not u["exp"])
    flushed_idx = {p for it in MANIFEST["items"].values()
                   if it.get("state") == "flushed"
                   for p in (it.get("up_paths") or [])}
    staged      = sum(q.stat().st_size for q in STAGING.rglob("*") if q.is_file())
    sped        = session_speed()
    left        = max(0, plan - staged)
    eta         = eta_str(left / sped if sped > 1e6 else None)
    extra       = f", +{unknown} unknown-size items" if unknown else ""
    return (f"progress: plan ~{fmt(plan)}{extra} | staged {fmt(staged)} | "
            f"speed {fmt(sped)}/s | left ~{fmt(left)} → ~{eta}")

# ============================================================ MAIN ==========
def main():
    api0 = HfApi(token=get_token())
    api0.create_repo(HF_REPO, repo_type="dataset", exist_ok=True, private=True)
    tree, api0 = sync_from_repo(api0)
    msave()
    threading.Thread(target=sync_worker, daemon=True).start()
    units = plan_units()
    up_set = {p for it in MANIFEST["items"].values()
              if it.get("state") == "flushed"
              for p in (it.get("up_paths") or [])}
    log(f"plan: {len(units)} unit(s), est {fmt(sum(u['exp'] or 0 for u in units))}"
        f" (git sizes unknown, excluded)")
    report(api0, "  ← session start")

    plan_bytes = sum(u["exp"] or 0 for u in units)
    sbar = make_bar(total=plan_bytes, desc="corpus banked")
    SESSION["bar"] = sbar

    failed = []
    done_units = 0
    try:
        for u in units:
            iid = u.get("url") if u["kind"] != "hf" else f"{u['repo']}::{u['path']}"
            if all(p in up_set for p in u["up"]):
                done_units += 1
                continue
            if MANIFEST["items"].get(iid, {}).get("state") == "failed":
                log(f"retrying previously failed unit: {u['key']}")
            try:
                log(f"→ {u['kind']}:{u['key']} {Path(u.get('dest', u.get('name', '?'))).name}")
                if u["kind"] == "wget":
                    exp = u["exp"]
                    if exp and exp > BIG_FILE_GB * 2**30:
                        # giant file: stream 2GB parts → upload each → delete locally
                        log(f"   (giant file {fmt(exp)} → streaming in {BIG_FILE_PART_GB}GB parts)")
                        mset(iid, kind="wget", key=u["key"], dest=u["dest"],
                             up_paths=[], size=exp, state="staged")
                        ups = wget_stream_upload(u["url"], u["dest"], exp, api0,
                                                  u["key"], f"data/{u['key']}")
                        mset(iid, up_paths=ups, state="flushed", verified=now())
                    else:
                        mset(iid, kind="wget", key=u["key"], dest=u["dest"],
                             up_paths=u["up"], size=exp, state="staged")
                        wget_resume(u["url"], u["dest"], name=u["key"])
                elif u["kind"] == "git":
                    mset(iid, kind="git", key=u["key"], dest=u["up"][0],
                         up_paths=u["up"], state="staged")
                    tmp = STAGING / "_git" / u["name"]
                    git_clone(u["url"], tmp)
                    tar_path = STAGING / u["key"] / f"{u['name']}.tar.gz"
                    tar_repo(tmp, tar_path)
                    # git sizes were unknown up front → grow the overall bar
                    sbar.total = (sbar.total or 0) + tar_path.stat().st_size
                    sbar.refresh()
                elif u["kind"] == "local":
                    mset(iid, kind="local", key=u["key"], dest=u["up"][0],
                         up_paths=u["up"], state="staged")
                    src = Path(u["url"])
                    if not src.exists():
                        raise FileNotFoundError(f"local source missing: {src}")
                    out = STAGING / "local" / f"{u['name']}.tar.gz"
                    n = tar_local_dir(src, out, u.get("exts"))
                    log(f"   packed {n} file(s) from {src} → {fmt(out.stat().st_size)}")
                elif u["kind"] == "hf":
                    mset(iid, kind="hf", key=u["key"], dest=u["dest"],
                         up_paths=u["up"], size=u["exp"], state="staged")
                    hf_fetch(u["repo"], u["path"], to_dir=str(STAGING / u["key"]))
                mset(iid, state="staged", size=u.get("exp"))
                done_units += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                mset(iid, state="failed", error=str(e)[:180], key=u["key"],
                     dest=u.get("dest", ""))
                failed.append(f"{u['key']}")
                log(f"❌ {u['key']} failed → recorded, continuing ({str(e)[:90]})")
            if disk_free_gb(ROOT) < FLUSH_GB + 2:
                flush_batch(api0)
            sp = session_speed()
            left = max(0, (sbar.total or 0) - SESSION.get("banked", 0))
            sbar.set_postfix_str(
                f"{done_units}/{len(units)} units | {fmt(sp)}/s | "
                f"left ~{fmt(left)} → ~{eta_str(left/sp if sp > 1e6 else None)}")
            log(eta_line(units))
    finally:
        flush_batch(api0, force=True)          # push whatever is left
        msave(); msave_to_repo(api0)
        sbar.set_postfix_str(f"{done_units}/{len(units)} units | banked {fmt(SESSION.get('banked',0))}")
        sbar.close()
        log(eta_line(units))
        if failed:
            log(f"⚠ failed this session ({len(set(failed))}): {sorted(set(failed))}")
            log("  → rerun the same command; failed units auto-retry")
        report(api0, "  ← session end")
        log(f"manifest: {MANIFEST_PATH} (+ mirrored to HF repo)")
        log("next stage → data/extraction.py (turn these archives into docs.jsonl)")

if __name__ == "__main__":
    main()
