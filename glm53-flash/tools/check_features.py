#!/usr/bin/env python3
"""Check the served endpoint's request features on the real model: stop_sequences (Anthropic, streaming and not),
`stop` (OpenAI), tool_choice (none / any / a named tool, both APIs), the thinking budget (`thinking.budget_tokens`:
the reasoning is closed with a wrap-up line and the answer follows), keep-alive pings while a long tool call is
generated (the tunnel drops a response silent for ~100 s) and the bounded queue (429 when more than
max_streams + max_queue requests are in flight).

  python3 tools/check_features.py URL --key KEY [--queue]
"""
import argparse, json, os, sys, threading, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_endpoint import post, health  # noqa: E402

TOOLS_A = [{"name": "read_file", "description": "Read a text file from disk.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"}},
                             "required": ["path"]}},
           {"name": "list_dir", "description": "List a directory.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
TOOLS_O = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
           for t in TOOLS_A]


def anthropic_stream_events(url, key, body):
    """-> (events, longest silence in seconds between two received lines)."""
    evs, gap, t = [], 0.0, time.time()
    with post(url, "/v1/messages", {**body, "stream": True}, key) as r:
        for line in r:
            gap, t = max(gap, time.time() - t), time.time()
            line = line.decode().strip()
            if line.startswith("data: "):
                evs.append(json.loads(line[6:]))
    return evs, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--key", default=os.environ.get("GLM_KEY", ""))
    ap.add_argument("--queue", action="store_true", help="also fire max_streams + max_queue + 1 concurrent requests (expect one 429)")
    a = ap.parse_args()
    url, key = a.url.rstrip("/"), a.key
    h = health(url, key)
    print("health:", {k: h.get(k) for k in ("max_streams", "max_sets", "max_queue", "piece", "buckets", "hbm_free_gb")})
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    # Health and model discovery expose runtime details, so both require the API key.
    for path in ("/health", "/v1/models"):
        try:
            urllib.request.urlopen(url + path, timeout=10)
            denied = False
        except urllib.error.HTTPError as exc:
            denied = exc.code == 401
        check(f"unauthenticated GET {path} is denied", denied)
    check("authenticated health request", health(url, key).get("status") == "ok")

    # 1. stop_sequences, non-streaming: a numbered list must stop before "3."
    body = {"model": "glm-5.3-flash", "max_tokens": 200, "temperature": 0,
            "messages": [{"role": "user", "content": "List the numbers one to five as a numbered list, one per line, like '1. one'. No other text."}],
            "stop_sequences": ["3."]}
    t = time.time(); r = json.loads(post(url, "/v1/messages", body, key).read()); dt = time.time() - t
    text = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
    print(f"1. stop_sequences (non-stream) in {dt:.1f}s: stop_reason={r['stop_reason']} stop_sequence={r['stop_sequence']!r} text={text!r}")
    check("stop_reason", r["stop_reason"] == "stop_sequence" and r["stop_sequence"] == "3.")
    check("text cut before the sequence", "3." not in text and "2." in text)
    # 2. stop_sequences, streaming: same request; the deltas must not contain the sequence
    t = time.time(); evs, _ = anthropic_stream_events(url, key, body); dt = time.time() - t
    deltas = "".join(e["delta"]["text"] for e in evs if e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta")
    md = [e for e in evs if e["type"] == "message_delta"][-1]
    print(f"2. stop_sequences (stream) in {dt:.1f}s: {md['delta']} usage={md['usage']} text={deltas!r}")
    check("streamed stop_reason", md["delta"]["stop_reason"] == "stop_sequence" and md["delta"]["stop_sequence"] == "3.")
    check("streamed text cut", "3." not in deltas and "2." in deltas)
    # 3. OpenAI `stop`
    r = json.loads(post(url, "/v1/chat/completions", {"max_tokens": 200, "temperature": 0, "stop": ["3."],
                                                      "messages": body["messages"]}, key).read())
    c = r["choices"][0]
    print(f"3. openai stop: finish={c['finish_reason']} text={c['message']['content']!r} usage={r['usage']}")
    check("openai stop", c["finish_reason"] == "stop" and "3." not in c["message"]["content"] and "2." in c["message"]["content"])
    # 4. tool_choice: a named tool on a prompt that would not call it by itself
    msgs = [{"role": "user", "content": "Say hello."}]
    r = json.loads(post(url, "/v1/messages", {"model": "glm-5.3-flash", "max_tokens": 200, "temperature": 0, "messages": msgs,
                                              "tools": TOOLS_A, "tool_choice": {"type": "tool", "name": "list_dir"}}, key).read())
    uses = [b for b in r["content"] if b["type"] == "tool_use"]
    print(f"4. tool_choice tool=list_dir: stop_reason={r['stop_reason']} content={json.dumps(r['content'])[:300]}")
    check("forced named tool", r["stop_reason"] == "tool_use" and len(uses) == 1 and uses[0]["name"] == "list_dir", str(uses)[:120])
    # 5. tool_choice any: some tool must be called
    r = json.loads(post(url, "/v1/messages", {"model": "glm-5.3-flash", "max_tokens": 200, "temperature": 0, "messages": msgs,
                                              "tools": TOOLS_A, "tool_choice": {"type": "any"}}, key).read())
    uses = [b for b in r["content"] if b["type"] == "tool_use"]
    print(f"5. tool_choice any: stop_reason={r['stop_reason']} content={json.dumps(r['content'])[:300]}")
    check("forced any tool", r["stop_reason"] == "tool_use" and len(uses) >= 1 and uses[0]["name"] in ("read_file", "list_dir"))
    # 6. tool_choice none on a prompt that WOULD call a tool
    msgs2 = [{"role": "user", "content": "Use the read_file tool to read /etc/hostname."}]
    r = json.loads(post(url, "/v1/messages", {"model": "glm-5.3-flash", "max_tokens": 200, "temperature": 0, "messages": msgs2,
                                              "tools": TOOLS_A, "tool_choice": {"type": "none"}}, key).read())
    uses = [b for b in r["content"] if b["type"] == "tool_use"]
    print(f"6. tool_choice none: stop_reason={r['stop_reason']} content={json.dumps(r['content'])[:200]}")
    check("no tool call", not uses and r["stop_reason"] in ("end_turn", "max_tokens"))
    # 7. OpenAI forced function + streaming
    with post(url, "/v1/chat/completions", {"max_tokens": 200, "temperature": 0, "messages": msgs, "tools": TOOLS_O, "stream": True,
                                            "tool_choice": {"type": "function", "function": {"name": "read_file"}}}, key) as resp:
        calls, finish = [], None
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                ch = json.loads(line[6:])["choices"][0]
                calls += ch["delta"].get("tool_calls", []); finish = ch["finish_reason"] or finish
    print(f"7. openai forced function (stream): finish={finish} calls={json.dumps(calls)[:200]}")
    check("openai forced function", finish == "tool_calls" and calls and calls[0]["function"]["name"] == "read_file")
    # 8. a follow-up after the forced call must reuse the context (the forced prefix is the template's own rendering)
    if uses := [b for b in json.loads(post(url, "/v1/messages", {"model": "glm-5.3-flash", "max_tokens": 200, "temperature": 0, "messages": msgs2,
                                                                  "tools": TOOLS_A, "tool_choice": {"type": "tool", "name": "read_file"}}, key).read())["content"] if b["type"] == "tool_use"]:
        h0 = health(url, key)
        follow = msgs2 + [{"role": "assistant", "content": [uses[0]]},
                          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"], "content": "kaggle-tpu-vm\n"}]}]
        r = json.loads(post(url, "/v1/messages", {"model": "glm-5.3-flash", "max_tokens": 100, "temperature": 0, "messages": follow, "tools": TOOLS_A}, key).read())
        h1 = health(url, key)
        reused = h1["prefix_tokens_reused"] - h0["prefix_tokens_reused"]
        print(f"8. follow-up after a forced call: reused {reused} tokens, text={json.dumps(r['content'])[:160]}")
        check("forced-call context reused", reused > 0, f"{reused} tokens")
    # 9. thinking budget: a multi-step problem at a 160-token budget must end its reasoning with the wrap-up line, then answer
    body = {"model": "glm-5.3-flash", "max_tokens": 1500, "temperature": 0, "thinking": {"type": "enabled", "budget_tokens": 160},
            "messages": [{"role": "user", "content": "Three friends share a restaurant bill of $187.50 plus an 18% tip. One of them only had a "
                                                     "$12 starter and pays for that plus her share of the tip on it; the other two split the rest "
                                                     "equally. What does each person pay? Show your reasoning, then give the three amounts."}]}
    t = time.time(); r = json.loads(post(url, "/v1/messages", body, key).read()); dt = time.time() - t
    think = "".join(b.get("thinking", "") for b in r["content"] if b["type"] == "thinking")
    text = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
    print(f"9. thinking budget 160 in {dt:.1f}s: stop={r['stop_reason']} usage={r['usage']} thinking={think[-160:]!r} text={text[:160]!r}")
    check("reasoning closed by the wrap-up line", "thinking budget is used up" in think)
    check("an answer follows", len(text.strip()) > 40 and r["stop_reason"] == "end_turn", f"{len(text)} chars")
    # 10. keep-alive: a tool call whose arguments take a while to generate must still reach the client, with pings meanwhile
    tools = [{"name": "write_file", "description": "Write a text file.", "input_schema": {"type": "object", "properties": {
              "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}]
    body = {"model": "glm-5.3-flash", "max_tokens": 6000, "temperature": 0, "tools": tools, "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "Use the write_file tool to write /tmp/linked_list.py: a complete Python module with a "
                                                     "singly linked list class (append, prepend, insert_at, remove, find, reverse, __len__, __iter__, "
                                                     "__repr__), docstrings on every method, and a __main__ demo. Aim for about 150 lines."}]}
    t = time.time(); evs, gap = anthropic_stream_events(url, key, body); dt = time.time() - t
    pings = sum(1 for e in evs if e["type"] == "ping")
    uses = [e["content_block"] for e in evs if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use"]
    args = "".join(e["delta"]["partial_json"] for e in evs if e["type"] == "content_block_delta" and e["delta"].get("type") == "input_json_delta")
    md = [e for e in evs if e["type"] == "message_delta"]
    print(f"10. long tool call (stream) in {dt:.1f}s: pings={pings} longest silence={gap:.1f}s tool={[u['name'] for u in uses]} "
          f"args={len(args)} chars stop={md[-1]['delta'] if md else None} usage={md[-1]['usage'] if md else None}")
    check("tool call delivered", uses and uses[0]["name"] == "write_file" and len(args) > 1500 and evs[-1]["type"] == "message_stop")
    check("kept alive while the call was generated", pings >= 1 and gap < 45, f"{pings} pings, {gap:.1f}s silence")
    if a.queue:
        n = h["max_streams"] + h["max_queue"] + 1
        codes = [None] * n
        def fire(i):
            try:
                post(url, "/v1/chat/completions", {"max_tokens": 64, "messages": [{"role": "user", "content": f"Write a limerick number {i}."}]}, key).read(); codes[i] = 200
            except urllib.error.HTTPError as e:
                codes[i] = e.code
        ths = [threading.Thread(target=fire, args=(i,)) for i in range(n)]
        [t.start() for t in ths]; [t.join() for t in ths]
        print(f"11. bounded queue: {n} concurrent -> {codes}")
        check("one 429", codes.count(429) >= 1 and codes.count(200) >= n - 2)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
