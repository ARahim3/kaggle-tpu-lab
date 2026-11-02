#!/usr/bin/env python3
"""Measure the served GLM-5.3-Flash endpoint from this machine.

  python3 tools/bench_endpoint.py [URL] [--key KEY] [--long-tokens 30000]
Runs: OpenAI chat (non-stream), Anthropic streaming (TTFT,
tok/s), an Anthropic tool call + follow-up with the tool result (prefix cache), and a long document Q&A twice (the
second turn should reuse the prefix). Prints one line per request with server-side prefill/decode figures from /health.
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def post(url, path, body, key, stream=False):
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = f"Bearer {key}"; hdr["x-api-key"] = key
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers=hdr)
    return urllib.request.urlopen(req, timeout=1800)


def health(url, key):
    req = urllib.request.Request(url + "/health", headers={"Authorization": f"Bearer {key}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def anthropic_stream(url, key, body):
    t0 = time.time(); first = None; n = 0; text = []; think = []; tools = []; stop = None
    with post(url, "/v1/messages", {**body, "stream": True}, key) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "content_block_delta":
                d = ev["delta"]
                if first is None:
                    first = time.time() - t0
                if d["type"] == "text_delta":
                    text.append(d["text"])
                elif d["type"] == "thinking_delta":
                    think.append(d["thinking"])
                elif d["type"] == "input_json_delta":
                    tools.append(d["partial_json"])
            elif ev["type"] == "content_block_start" and ev["content_block"]["type"] == "tool_use":
                tools.append(ev["content_block"]["name"] + ":")
            elif ev["type"] == "message_delta":
                stop = ev["delta"]["stop_reason"]; n = ev["usage"]["output_tokens"]
    dt = time.time() - t0
    return {"ttft_s": round(first or dt, 2), "total_s": round(dt, 2), "out_tokens": n,
            "tok_s": round(n / max(dt - (first or 0), 1e-6), 1), "stop": stop, "thinking": "".join(think)[:80],
            "text": "".join(text)[:120], "tools": "".join(tools)[:120], "_thinking": "".join(think), "_text": "".join(text)}


def test_png(w=336, h=224):
    """A synthetic PNG (no PIL): white background, a red filled circle on the left, a blue square on the right."""
    import struct, zlib
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            if (x - w // 4) ** 2 + (y - h // 2) ** 2 <= (h // 4) ** 2:
                row += b"\xe0\x20\x20"
            elif abs(x - 3 * w // 4) <= h // 4 and abs(y - h // 2) <= h // 4:
                row += b"\x20\x40\xe0"
            else:
                row += b"\xff\xff\xff"
        rows.append(bytes(row))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def image_steps(url, key):
    """6. vision: an Anthropic base64 image (two turns: the second reuses the image's context), an OpenAI data URL."""
    import base64
    b64 = base64.b64encode(test_png()).decode()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "What shapes and colours are in this image? One sentence."},
                                         {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]}]
    body = {"model": "glm-5.3-flash", "max_tokens": 120, "temperature": 0, "messages": msgs}
    n_in = json.loads(post(url, "/v1/messages/count_tokens", body, key).read())["input_tokens"]
    show = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}
    r1 = anthropic_stream(url, key, body); print(f"image turn 1 ({n_in} input tokens):", show(r1), "|", r1["_text"][:160])
    msgs += [{"role": "assistant", "content": [{"type": "text", "text": r1["_text"]}]},
             {"role": "user", "content": "Which shape is on the left?"}]
    h0 = health(url, a.key)
    r2 = anthropic_stream(url, key, {**body, "messages": msgs})
    print("image turn 2:", show(r2), "| prefix reused:", health(url, a.key)["prefix_tokens_reused"] - h0["prefix_tokens_reused"], "|", r2["_text"][:120])
    t = time.time()
    r = json.loads(post(url, "/v1/chat/completions", {"max_tokens": 80, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the image briefly."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}, key).read())
    print(f"openai image: {time.time() - t:.2f}s |", r["choices"][0]["message"].get("content", "")[:160], r["usage"])
    h = health(url, a.key)
    print("vision counters:", {k: v for k, v in h.items() if "image" in k or "vision" in k})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--key", default=os.environ.get("GLM_API_KEY", ""))
    ap.add_argument("--long-tokens", type=int, default=30000); ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--image", action="store_true", help="only the vision steps"); ap.add_argument("--no-image", action="store_true")
    a = ap.parse_args()
    url = (a.url or None).rstrip("/")
    print("endpoint", url, "| health", {k: v for k, v in health(url, a.key).items() if k in ("max_len", "requests", "prefix_hits")})
    if a.image:
        return image_steps(url, a.key)
    # 1. OpenAI chat
    t = time.time()
    r = json.loads(post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi in five words."}],
                                                       "max_tokens": 64, "temperature": 0}, a.key).read())
    print(f"openai chat: {time.time() - t:.2f}s |", json.dumps(r["choices"][0]["message"])[:200], r["usage"])
    # 2. Anthropic streaming
    print("anthropic stream:", anthropic_stream(url, a.key, {"model": "glm-5.3-flash", "max_tokens": 200, "temperature": 0,
                                                              "messages": [{"role": "user", "content": "Write a limerick about TPUs."}]}))
    # 3. tool call + follow-up
    tools = [{"name": "read_file", "description": "Read a text file from disk.",
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                               "required": ["path"]}}]
    msgs = [{"role": "user", "content": "Use the read_file tool to read /etc/hostname (at most 5 lines), then tell me the hostname."}]
    r = json.loads(post(url, "/v1/messages", {"model": "glm-5.3-flash", "max_tokens": 300, "temperature": 0, "messages": msgs, "tools": tools}, a.key).read())
    uses = [b for b in r["content"] if b["type"] == "tool_use"]
    print("tool call:", r["stop_reason"], json.dumps(uses)[:200])
    if uses:
        msgs += [{"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"], "content": "kaggle-tpu-vm\n"}]}]
        h0 = health(url, a.key)
        print("follow-up:", {k: v for k, v in anthropic_stream(url, a.key, {"model": "glm-5.3-flash", "max_tokens": 120, "temperature": 0, "messages": msgs, "tools": tools}).items() if not k.startswith("_")},
              "| prefix reused:", health(url, a.key)["prefix_tokens_reused"] - h0["prefix_tokens_reused"])
    if a.skip_long:
        return
    # 4. long document, two turns
    words = "silicon river lantern harbor meadow compass ledger orchard glacier signal".split()
    import random
    random.seed(0)
    doc = " ".join(random.choice(words) for _ in range(int(a.long_tokens * 0.75)))
    at = len(doc) // 3
    doc = doc[:at] + " The secret passcode for the vault is 7391-XRAY. " + doc[at:]
    msgs = [{"role": "user", "content": f"Here is a document:\n\n{doc}\n\nWhat is the secret passcode for the vault? Answer with just the passcode."}]
    body = {"model": "glm-5.3-flash", "max_tokens": 40, "temperature": 0, "messages": msgs}
    n_in = json.loads(post(url, "/v1/messages/count_tokens", body, a.key).read())["input_tokens"]
    print(f"long doc: {n_in} input tokens")
    show = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}
    r1 = anthropic_stream(url, a.key, body); print("  turn 1:", show(r1))
    # turn 2 sends the thinking block back (as Claude Code does within a turn)
    msgs += [{"role": "assistant", "content": [{"type": "thinking", "thinking": r1["_thinking"], "signature": ""}, {"type": "text", "text": r1["_text"]}]},
             {"role": "user", "content": "Repeat it once more, then say done."}]
    h0 = health(url, a.key)
    r2 = anthropic_stream(url, a.key, {**body, "messages": msgs}); print("  turn 2 (thinking sent back):", show(r2), "| prefix reused:", health(url, a.key)["prefix_tokens_reused"] - h0["prefix_tokens_reused"])
    # turn 3 drops the thinking (as clients do for earlier turns) -> the server keeps its reasoning and reuses the context
    msgs += [{"role": "assistant", "content": [{"type": "text", "text": r2["_text"]}]}, {"role": "user", "content": "Now say goodbye in three words."}]
    h0 = health(url, a.key)
    r3 = anthropic_stream(url, a.key, {**body, "messages": msgs}); print("  turn 3 (thinking dropped):", show(r3), "| prefix reused:", health(url, a.key)["prefix_tokens_reused"] - h0["prefix_tokens_reused"])
    # 5. interleaved sessions: a background call with another system prompt evicts the long-doc context; the next turn of
    #    the long-doc conversation must resume from its parked snapshot (only the new turn is prefilled)
    rb = anthropic_stream(url, a.key, {"model": "glm-5.3-flash", "max_tokens": 30, "temperature": 0,
                                        "system": "You write short titles for chats. " * 40,
                                        "messages": [{"role": "user", "content": "Title for a chat about TPUs, five words."}]})
    print("  background call (other system prompt):", show(rb))
    msgs += [{"role": "assistant", "content": [{"type": "text", "text": r3["_text"]}]}, {"role": "user", "content": "One more: count to three."}]
    h1 = health(url, a.key)
    r4 = anthropic_stream(url, a.key, {**body, "messages": msgs}); h2 = health(url, a.key)
    print("  turn 4 after the interleave:", show(r4), "| prefix reused:", h2["prefix_tokens_reused"] - h1["prefix_tokens_reused"],
          "| snap hits", h2.get("snap_hits", 0) - h1.get("snap_hits", 0), "| store", h2.get("snap_entries"), "entries",
          f"{h2.get('snap_bytes', 0) / 1e6:.0f} MB, snapshot time total {h2.get('snap_s', 0):.2f}s")
    if not a.no_image:
        image_steps(url, a.key)


if __name__ == "__main__":
    main()
