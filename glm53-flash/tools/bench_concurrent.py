#!/usr/bin/env python3
"""Concurrency benchmark of the served GLM-5.3-Flash endpoint (Anthropic streaming API).

  python3 tools/bench_concurrent.py URL --key KEY [--streams 4] [--max-tokens 200] [--rounds 1]
Runs N different prompts (1) one after the other and (2) all at once; prints per-request TTFT / tok/s and the
aggregate tok/s of the concurrent round, plus the server's /health deltas (steps, step_tokens = batched rows).
"""
import argparse, json, os, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_endpoint import anthropic_stream, health  # noqa: E402

PROMPTS = ["Explain how a hash map works and when you would not use one.",
           "Write a Python function that parses ISO-8601 dates without external libraries, with tests.",
           "Summarise the causes of the French Revolution in five bullet points, then one paragraph.",
           "Describe the trade-offs of speculative decoding for LLM inference.",
           "Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Compare TCP and QUIC for a video conferencing application.",
           "What is the Sinkhorn algorithm and where is it used in machine learning?",
           "Plan a three-day itinerary for Kyoto in autumn."]


def one(url, key, prompt, max_tokens, out, i):
    try:
        out[i] = anthropic_stream(url, key, {"model": "glm-5.3-flash", "max_tokens": max_tokens,
                                             "messages": [{"role": "user", "content": prompt}]})
    except Exception as e:  # noqa: BLE001
        out[i] = {"error": repr(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--key", default=os.environ.get("GLM_KEY", ""))
    ap.add_argument("--streams", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--skip-serial", action="store_true")
    a = ap.parse_args()
    prompts = (PROMPTS * 4)[:a.streams]
    if not a.skip_serial:
        print(f"--- serial: {a.streams} requests one after the other")
        t0 = time.time(); tot = 0
        for i, p in enumerate(prompts):
            r = anthropic_stream(a.url, a.key, {"model": "glm-5.3-flash", "max_tokens": a.max_tokens, "messages": [{"role": "user", "content": p}]})
            tot += r["out_tokens"]
            print(f"  [{i}] ttft {r['ttft_s']}s  {r['tok_s']} tok/s  {r['out_tokens']} tok  {r['stop']}  {r['text'][:50]!r}")
        print(f"  serial total: {tot} tokens in {time.time() - t0:.1f}s = {tot / (time.time() - t0):.1f} tok/s")
    for rnd in range(a.rounds):
        print(f"--- concurrent round {rnd + 1}: {a.streams} requests at once")
        h0 = health(a.url, a.key)
        out = {}
        ths = [threading.Thread(target=one, args=(a.url, a.key, p, a.max_tokens, out, i)) for i, p in enumerate(prompts)]
        t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; dt = time.time() - t0
        h1 = health(a.url, a.key)
        tot = 0
        for i in range(a.streams):
            r = out[i]
            if "error" in r:
                print(f"  [{i}] ERROR {r['error']}"); continue
            tot += r["out_tokens"]
            print(f"  [{i}] ttft {r['ttft_s']}s  {r['tok_s']} tok/s  {r['out_tokens']} tok  {r['stop']}  {r['text'][:50]!r}")
        steps = h1.get("steps", 0) - h0.get("steps", 0); rows = h1.get("step_tokens", 0) - h0.get("step_tokens", 0)
        print(f"  concurrent total: {tot} tokens in {dt:.1f}s = {tot / dt:.1f} tok/s aggregate | server steps {steps}, "
              f"rows {rows} (avg batch {rows / max(steps, 1):.2f})")
    print("health:", {k: v for k, v in health(a.url, a.key).items() if k in ("active", "pending", "live_contexts", "max_streams", "max_sets", "requests", "tokens")})


if __name__ == "__main__":
    main()
