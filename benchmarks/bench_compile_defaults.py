"""Cold process measurements; use a fresh cache, then repeat the identical command.

Example: TORCHINDUCTOR_CACHE_DIR=/tmp/laya-probe python benchmarks/bench_compile_defaults.py --model /path/to/checkpoint
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench_compile import CALLS, WORDS, questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()
    import torch
    from torch._dynamo.utils import counters
    from laya import Agent

    def smi():
        print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)

    smi()
    print("torch", torch.__version__, "model", args.model, flush=True)
    kw = {}
    if args.no_warmup:
        kw["compile_warmup"] = False
    if args.cache:
        kw["compile_cache"] = True
    if args.mode:
        kw["compile_mode"] = args.mode
    t = time.perf_counter()
    agent = Agent(args.model, device="cuda", compile=True, **kw)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    print("load_s", load_s, "cache", os.environ.get("TORCHINDUCTOR_CACHE_DIR"), flush=True)
    times = []
    # Repeat the sequence to expose CUDA graph recording/replay and retained-output issues.
    answers = []
    for i, (n, k, words) in enumerate(CALLS * 3):
        state = " ".join(WORDS[j % len(WORDS)] for j in range(words))
        t = time.perf_counter()
        result = agent.predict(state, questions(n, k))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        times.append(elapsed)
        if i < len(CALLS):
            answers.append(result["answers"])
        else:
            assert result["answers"] == answers[i % len(CALLS)], i
        print("request", i + 1, "seconds", elapsed, "graphs", counters["stats"]["unique_graphs"], flush=True)
    print("RESULT", json.dumps({"load_s": load_s, "requests_s": times,
          "allocated_mib": torch.cuda.memory_allocated() / 2**20,
          "reserved_mib": torch.cuda.memory_reserved() / 2**20,
          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
          "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
          "counters": {k: dict(v) for k, v in counters.items() if k in ("stats", "inductor")}}), flush=True)
    smi()


if __name__ == "__main__":
    main()
