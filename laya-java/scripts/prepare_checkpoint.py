#!/usr/bin/env python3
"""Download a pinned laya checkpoint and export the fused ONNX graph the Java tests need.

Separated from `gen_fixtures.py` on purpose: the download and the export are the slow, cacheable
part of a CI run (minutes), while regenerating the fixtures from the Python code is the part that
must happen on every commit (seconds). Splitting them lets the workflow cache the first and never
cache the second.

Writes
    <repo>/checkpoints-java/<name>/            the checkpoint, as `Agent.open` expects it
    <repo>/onnx-java/<name>/laya.onnx          the fused graph, as `LayaSession.open` expects it

Usage:
    python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The same pin the .NET parity lane uses (laya-dotnet/tools/regen_golden.py). Both lanes measure
# the same checkpoint, so they must not drift to different revisions: a port that matched one and
# not the other would look like a port bug.
HF_REPO = "convaiinnovations/laya"
HF_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

# What `Agent.open` reads, plus what the exporter needs to trace the model.
WANTED = ("rl_agent_config.json", "model.safetensors", "config.json",
          "tokenizer.json", "tokenizer/*", "encoder/*")


def snapshot(name, revision):
    """The pinned checkpoint directory in the Hugging Face cache, downloading on first use."""
    from huggingface_hub import snapshot_download

    sys.path.insert(0, REPO)
    from laya.router import DEFAULT_MODELS

    if name not in DEFAULT_MODELS:
        raise SystemExit("unknown checkpoint %r; laya.router knows %s"
                         % (name, sorted(DEFAULT_MODELS)))
    repo, subfolder = DEFAULT_MODELS[name]
    if repo != HF_REPO:
        raise SystemExit(
            "%s now lives in %r, but the pinned revision belongs to %r; update HF_REPO and "
            "HF_REVISION here and in laya-dotnet/tools/regen_golden.py together"
            % (name, repo, HF_REPO))
    prefix = (subfolder + "/") if subfolder else ""
    root = snapshot_download(repo, revision=revision,
                             allow_patterns=[prefix + pattern for pattern in WANTED])
    return os.path.join(root, subfolder) if subfolder else root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="english, multilingual, ...")
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--force", action="store_true",
                        help="re-export even when the graph is already present")
    args = parser.parse_args(argv)

    model_dir = os.path.join(REPO, "checkpoints-java", args.checkpoint)
    graph_dir = os.path.join(REPO, "onnx-java", args.checkpoint)
    graph = os.path.join(graph_dir, "laya.onnx")

    print("downloading %s at %s" % (args.checkpoint, args.revision[:12]), flush=True)
    cached = snapshot(args.checkpoint, args.revision)
    # Copied out of the cache rather than used in place: the cache path contains the revision and
    # the tests take a plain directory, and a copy keeps a cached export from pointing into a
    # directory a later `snapshot_download` may garbage-collect.
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.copytree(cached, model_dir, symlinks=False)
    print("  checkpoint at %s" % os.path.relpath(model_dir, REPO))

    if os.path.exists(graph) and os.path.getsize(graph) > 0 and not args.force:
        print("  graph already exported at %s (use --force to redo it)"
              % os.path.relpath(graph, REPO))
        return 0

    os.makedirs(graph_dir, exist_ok=True)
    started = time.perf_counter()
    print("exporting the fused graph (this is the slow part)", flush=True)
    # The repository's own exporter, so the graph the Java tests run is the graph laya ships.
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_onnx.py"),
                    "--model", model_dir, "--output", graph],
                   check=True, cwd=REPO)
    if not os.path.exists(graph) or os.path.getsize(graph) == 0:
        raise SystemExit("the exporter reported success but wrote no graph at %s" % graph)
    print("  %s (%.1f MB) in %.1f s"
          % (os.path.relpath(graph, REPO), os.path.getsize(graph) / 1e6,
             time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
