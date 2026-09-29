"""Bake Laya checkpoints into a Hugging Face cache from ModelScope, during an image build.

The runtime resolves a repo id through `huggingface_hub.snapshot_download`, which reads its
local cache. With `HF_HUB_OFFLINE=1` -- or a network that cannot reach huggingface.co -- that
resolution is purely local: `refs/<revision>` names a commit and an existing
`snapshots/<sha>` directory is returned as it is, with no repo listing at all
(`huggingface_hub/_snapshot_download.py`). A mirror therefore only has to lay the files out the
way the Hub would have, and every code path that names a checkpoint keeps its ids and its
revision handling: `Agent(...)`, the `Router` laya-serve builds, `laya.cli`, the integrations.

Only the standard library is used, because an image build has no reason to install the
ModelScope SDK for a handful of HTTP calls. Every file is streamed with resume and its size and SHA-256
are checked against the values the repository API reports, so a truncated or substituted
download fails the build instead of shipping inside a layer that looks healthy.

Usage inside a Dockerfile RUN:

    python prefetch_modelscope.py --models convaiinnovations/laya:multilingual \
        --revision master --cache-dir "$HF_HOME/hub"

Each spec is `repo[:subfolder]`; the subfolder selects one checkpoint from the bundled
repository, the same way `Agent("convaiinnovations/laya", subfolder="multilingual")` does.
"""
import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

FILES_API = "https://modelscope.cn/api/v1/models/{repo}/repo/files"
FILE_API = "https://modelscope.cn/api/v1/models/{repo}/repo"
# What a Laya checkpoint needs: the same set laya/agent.py hands snapshot_download, so a baked
# snapshot carries exactly the files a load would have fetched and none of its siblings.
NEEDED = ("rl_agent_config.json", "model.safetensors")
FOLDERS = ("tokenizer/", "encoder/")
CHUNK = 1 << 20
RETRIES = 4
BACKOFF = 2.0
TIMEOUT = 60
# The revision the runtime asks for by default. The mirror publishes `master`, the Hub publishes
# `main`, and a baked snapshot answers both: this is the name `refs/` is written under.
DEFAULT_BRANCH = "main"

# A checkpoint type and the path it means inside the bundled repository. These are the paths the
# `Router` and the one-shot quickstart load by default (laya/router.py: DEFAULT_MODELS), which is
# why a type bakes them rather than the standalone repositories of the same checkpoints: a baked
# image then serves the default configuration with no change at all.
TYPES = {
    "english": ["convaiinnovations/laya:"],
    "multilingual": ["convaiinnovations/laya:multilingual"],
    "typed-decisions": ["convaiinnovations/laya:typed-decisions"],
}
# The spellings laya/router.py accepts for the same three checkpoints, so an operator who copies a
# name out of that mapping is not told it is unknown here.
ALIASES = {"en": "english", "laya": "english", "default": "english",
           "multi": "multilingual", "ml": "multilingual",
           "typed": "typed-decisions", "decisions": "typed-decisions"}


def expand_model(value: str):
    """A type name, or one or more `repo[:subfolder]` specs, into the specs to bake.

    `all` is the whole family. Anything holding a `/` is taken as a spec, which is how a standalone
    repository (`convaiinnovations/laya-multilingual`) or a fine-tuned checkpoint is baked; specs
    and types can be mixed in one comma- or space-separated value.
    """
    parts = [part for part in re.split(r"[,\s]+", (value or "").strip()) if part]
    if not parts:
        raise SystemExit("laya: --model is empty; expected a checkpoint type (%s) or a "
                         "repo[:subfolder] spec" % ", ".join(sorted(TYPES) + ["all"]))
    specs = []
    for part in parts:
        if "/" in part:
            specs.append(part)
            continue
        name = ALIASES.get(part.lower(), part.lower())
        if name == "all":
            specs += [spec for key in sorted(TYPES) for spec in TYPES[key]]
        elif name in TYPES:
            specs += TYPES[name]
        else:
            raise SystemExit("laya: unknown checkpoint type %r; choose one of %s, all, or a "
                             "repo[:subfolder] spec" % (part, sorted(TYPES)))
    return specs


def wanted(rel: str) -> bool:
    """Whether a path inside the subfolder is one of the files a checkpoint load fetches."""
    return rel in NEEDED or rel.startswith(FOLDERS)


def split_spec(spec: str):
    """`repo[:subfolder]` -> (repo, subfolder); `:` with nothing after it means the root."""
    repo, sep, sub = spec.partition(":")
    return repo.strip(), sub.strip() if sep else ""


def cache_root(override):
    """The hub cache directory: an explicit path, `HF_HUB_CACHE`, or `$HF_HOME/hub`."""
    if override:
        return Path(override)
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def repo_folder(repo: str) -> str:
    """The cache directory name the Hub derives from a repo id: `ns/name` -> `models--ns--name`."""
    return "models--" + repo.replace("/", "--")


def list_repo(repo: str, revision: str, subfolder: str):
    """(tip commit, entries) for the revision's own files inside `subfolder`.

    Each entry reports the path as the API names it, the path relative to the checkpoint -- what
    the Hub caches -- its size, and its SHA-256 when the API publishes one. The listing API returns
    one directory at a time and names subdirectories as `tree` entries, so the walk follows them:
    the bundled repository keeps each checkpoint's tokenizer and encoder in their own directories.
    """
    entries, tip, queue = [], "", [subfolder]
    prefix = subfolder + "/" if subfolder else ""
    while queue:
        query = urllib.parse.urlencode({"Revision": revision, "Root": queue.pop(0)})
        try:
            with urllib.request.urlopen(FILES_API.format(repo=repo) + "?" + query,
                                        timeout=TIMEOUT) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
            raise SystemExit("laya: cannot list %s at revision %r: %s" % (repo, revision, exc))
        if not payload.get("Success"):
            raise SystemExit("laya: ModelScope returned %r for %s -- check the repo id and revision"
                             % (payload.get("Message"), repo))
        data = payload.get("Data") or {}
        tip = tip or ((data.get("LatestCommitter") or {}).get("Revision") or "").strip()
        for item in data.get("Files") or []:
            path, kind = item.get("Path", ""), item.get("Type")
            if kind == "tree":
                if path.startswith(prefix):
                    queue.append(path)
                continue
            if kind != "blob" or not path.startswith(prefix):
                continue
            rel = path[len(prefix):]
            if not wanted(rel):
                continue
            entries.append({
                "path": path,
                "rel": rel,
                "size": int(item.get("Size") or 0),
                "sha256": (item.get("Sha256") or "").strip(),
                "revision": (item.get("Revision") or "").strip(),
            })
    return tip, entries


def snapshot_commit(tip: str, entries: list) -> str:
    """The commit to key the baked snapshot by: the weights' own, else the revision's tip.

    The weights' commit is the one a caller would pin to get *these* bytes back, so it is the
    more useful key. A mixed-commit fetch still lands as one snapshot directory, because the
    offline path never compares the file list against a commit -- but that is a warning, not a
    failure: a maintenance commit to a README must not break a build.
    """
    commits = {entry["revision"] for entry in entries if entry["revision"]}
    weights = [entry["revision"] for entry in entries
               if entry["rel"] == "model.safetensors" and entry["revision"]]
    if len(commits) > 1:
        print("    note: files span commits %s; keying the snapshot by the weights' commit"
              % sorted(commits), file=sys.stderr)
    return (weights[0] if weights else "") or tip


def download(repo: str, revision: str, entry: dict, dest: Path) -> None:
    """Stream one file into `dest`, resuming a partial copy and verifying what landed."""
    query = urllib.parse.urlencode({"FilePath": entry["path"], "Revision": revision})
    url = FILE_API.format(repo=repo) + "?" + query
    for attempt in range(1, RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": "bytes=%d-" % have} if have else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=TIMEOUT) as response:
                # A 206 answers the Range request; anything else starts the file over, because
                # appending a full body to a partial one would corrupt it.
                mode = "ab" if response.status == 206 and have else "wb"
                with open(dest, mode) as fh:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                http.client.HTTPException) as exc:
            print("    retry %d/%d after %s: %s" % (attempt, RETRIES, type(exc).__name__, exc),
                  file=sys.stderr)
        # Written outside the `try`, so a response that closed early still re-asks for the rest
        # instead of being treated as a complete file.
        on_disk = dest.stat().st_size if dest.exists() else 0
        if entry["size"] and on_disk == entry["size"]:
            break
        if attempt == RETRIES:
            raise SystemExit("laya: %s: %d of %d bytes after %d attempts -- check the connection "
                             "to modelscope.cn" % (entry["path"], on_disk, entry["size"], RETRIES))
        time.sleep(BACKOFF * attempt)

    digest = hashlib.sha256()
    with open(dest, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    if entry["sha256"] and digest.hexdigest() != entry["sha256"]:
        raise SystemExit("laya: %s: sha256 mismatch -- the repository reports %s, the download "
                         "has %s" % (entry["path"], entry["sha256"], digest.hexdigest()))


def seed(cache: Path, repo: str, commit: str, revision: str, staging: Path) -> None:
    """Publish a verified download as a hub snapshot, then name its commit in `refs/`."""
    repo_dir = cache / repo_folder(repo)
    snapshot = repo_dir / "snapshots" / commit
    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(snapshot, ignore_errors=True)
    # Renamed into place only once it holds every file it claims to, and the ref is written only
    # after the rename: an offline snapshot_download returns whatever `snapshots/<sha>` already
    # contains, so a half-published snapshot would otherwise be served as if complete.
    os.replace(str(staging), str(snapshot))
    for name in (DEFAULT_BRANCH, revision):
        if name and len(name) < 64:
            (refs / name).write_text(commit, encoding="utf-8")


def prefetch(spec: str, revision: str, cache: Path) -> None:
    repo, subfolder = split_spec(spec)
    tip, entries = list_repo(repo, revision, subfolder)
    if not entries:
        raise SystemExit("laya: %s%s holds no checkpoint files at revision %r"
                         % (repo, "/" + subfolder if subfolder else "", revision))
    commit = snapshot_commit(tip, entries)
    if not commit:
        raise SystemExit("laya: %s%s names no commit at revision %r, so there is nothing to key "
                         "the baked snapshot by" % (repo, "/" + subfolder if subfolder else "",
                                                    revision))
    repo_dir = cache / repo_folder(repo)
    staging = repo_dir / "snapshots" / (".staging-" + str(os.getpid()))
    try:
        for entry in entries:
            # The repository-relative path, not the checkpoint-relative one: `Agent` appends the
            # subfolder itself (`model_dir/<subfolder>`), so the snapshot has to mirror the repo.
            dest = staging / entry["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            print("  %s: %d bytes" % (entry["path"], entry["size"]))
            download(repo, revision, entry, dest)
        seed(cache, repo, commit, revision, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print("  baked as models--%s/snapshots/%s" % (repo.replace("/", "--"), commit))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake ModelScope checkpoints into a hub cache")
    parser.add_argument("--model", required=True, metavar="TYPE_OR_SPEC",
                        help="checkpoint type (multilingual, english, typed-decisions, all) or one "
                             "or more comma- or space-separated repo[:subfolder] specs")
    parser.add_argument("--revision", default="master",
                        help="ModelScope revision (branch, tag or commit); default master")
    parser.add_argument("--cache-dir", default=None,
                        help="hub cache directory; defaults to HF_HUB_CACHE or $HF_HOME/hub")
    args = parser.parse_args()
    specs = expand_model(args.model)
    cache = cache_root(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        print("modelscope/%s at revision %s" % (spec, args.revision))
        prefetch(spec, args.revision, cache)
    print("baked %d checkpoint(s) into %s" % (len(specs), cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
