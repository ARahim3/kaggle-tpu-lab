"""CPU smoke test of kernel/serve_glm53.py (run from the tpu_trial checkout: `.venv-glm/bin/python
kaggle-tpu-lab/glm53-flash/tools/harness_serve.py`, ~40 s; needs the GLM tokenizer in ~/.cache/glm_hf): the tiny resident engine + the real GLM tokenizer in a runner-like namespace,
JOB_TUNNEL=False. Exercises the self-test in the job, then Anthropic streaming x2 concurrently, count_tokens, a
mid-stream disconnect, a follow-up turn (prefix reuse of a live context), stop sequences (unit + the completions
endpoint's cancel path), tool_choice (none / any / named), keep-alive pings while a tool call is buffered (both APIs),
the thinking budget (a forced </think>), and the bounded queue (429)."""
import os, sys, glob, json, threading, time, urllib.request, urllib.error
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import numpy as np, jax
from transformers import AutoTokenizer
from glm53.tests.test_batched_rows import build_engine

rng = np.random.default_rng(0)
eng, V = build_engine(True, rng, max_len=512)
snap = glob.glob(os.path.expanduser("~/.cache/glm_hf/models--zai-org--GLM-5.3-Flash/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(snap)
t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
ids = np.asarray(tok("Write a haiku about the sea.", add_special_tokens=False).input_ids, np.int32)[None]
eos = {tok.convert_tokens_to_ids("<|endoftext|>")}
ns = dict(eng=eng, tok=tok, ids=ids, eos=eos, cfg=eng.cfg, VISION=None, VISION_FWD=None, SERVE_FOREVER=False,
          CFG_PRESET=dict(skip_runtime=True, tunnel=False, port=8765, streams=2, sets=2, sched_piece=32, api_key="k", base_min=10 ** 9,
                          snap_min=1, snap_rows=8, snap_warm_tokens=64, max_new_default=24, max_queue=1, bucket_min=64, vision=False,
                          min_free_gb=0, dflash=""))      # (dflash="": the default would fetch the 2.3 GB drafter)
exec(compile(open(os.path.join(HERE, "..", "kernel", "serve_glm53.py")).read(), "serve_glm53", "exec"), ns)
STATE, SCHED = ns["STATE"], ns["SCHED"]
assert eng.PREFILL_BUCKETS[0] == 64, eng.PREFILL_BUCKETS
URL = "http://127.0.0.1:8765"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer k"}

def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers=HDR)
    return urllib.request.urlopen(req, timeout=600)

def stream_anthropic(body, out, key, cut_after=None):
    t = time.time(); n = 0; first = None
    with post("/v1/messages", {**body, "stream": True}) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                if ev["type"] == "content_block_delta":
                    n += 1; first = first or time.time() - t
                    if cut_after and n >= cut_after:
                        r.fp.raw._sock.close() if hasattr(r.fp, "raw") else None
                        out[key] = ("cut", n); return
                if ev["type"] == "message_delta":
                    out[key] = (ev["delta"]["stop_reason"], ev["usage"]["output_tokens"], round(first, 2), round(time.time() - t, 2))

log("=== count_tokens")
r = json.loads(post("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hello there"}]}).read())
log("count_tokens ->", r)
log("=== two concurrent Anthropic streams")
out = {}
ths = [threading.Thread(target=stream_anthropic, args=({"max_tokens": 20, "messages": [{"role": "user", "content": p}]}, out, i))
       for i, p in enumerate(["Tell me about the moon.", "Tell me about the sea and the sky."])]
[t.start() for t in ths]; [t.join() for t in ths]
log("streams ->", out, "| steps", STATE["steps"], "step_tokens", STATE["step_tokens"], "active", len(SCHED.active), "live", len(SCHED.live))
assert all(v[0] in ("max_tokens", "end_turn") for v in out.values()), out
log("=== disconnect mid-stream")
out = {}
try:
    stream_anthropic({"max_tokens": 40, "messages": [{"role": "user", "content": "A long story please."}]}, out, "cut", cut_after=3)
except Exception as e:
    out["cut_err"] = repr(e)[:80]
time.sleep(3)
log("after cut ->", out, "| active", len(SCHED.active), "pending", len(SCHED.pending), "live", len(SCHED.live))
assert len(SCHED.active) == 0
log("=== follow-up turn (prefix reuse of a live context)")
r1 = json.loads(post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Tell me about the moon."}], "max_tokens": 12}).read())
msgs = [{"role": "user", "content": "Tell me about the moon."}, {"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
        {"role": "user", "content": "And more?"}]
h0 = dict(STATE)
r2 = json.loads(post("/v1/chat/completions", {"messages": msgs, "max_tokens": 8}).read())
log("turn 2 usage", r2["usage"], "| prefix hits", STATE["prefix_hits"] - h0["prefix_hits"], "reused", STATE["prefix_tokens_reused"] - h0["prefix_tokens_reused"], "snap hits", STATE.get("snap_hits", 0))

log("=== StopFilter unit")
SF = ns["StopFilter"]
f = SF(["</end>", "STOP"])
got = []
for piece in ["hello wo", "rld <", "/en", "d> tail"]:
    o, hit = f.feed(piece); got.append((o, hit))
log("filter ->", got)
assert "".join(o for o, _ in got) == "hello world " and got[-1][1] == "</end>", got
f = SF(["STOP"]); o1, _ = f.feed("abc ST"); o2, _ = f.feed("x"); o3 = f.flush()
assert (o1, o2, o3) == ("abc ", "STx", ""), (o1, o2, o3)      # a held tail that turns out not to be a stop is released
log("=== run_request with a scripted generate: streaming events, held tail, cancel at the stop sequence")
T = ns["T"]
script = tok("some reasoning", add_special_tokens=False).input_ids + [T["</think>"]] + tok("Answer: yes. END OF ANSWER more text", add_special_tokens=False).input_ids + [T["<|user|>"]]
real_generate = ns["generate"]
def fake_generate(prompt, max_new, temperature, top_p, on_token=None, imgs=None, rid=None, on_idle=None, budget=None):
    n = 0
    for t in script:
        n += 1
        if on_token(t):
            return script[:n], 0.1, 0.2, 0, "stop_sequence"
    return script, 0.1, 0.2, 0, "stop"
ns["generate"] = fake_generate
try:
    evs = []
    r = ns["run_request"]([1, 2, 3], 100, 0.0, 1.0, None, "x", None, stops=["END OF ANSWER"], on_event=lambda k, v: evs.append((k, v)))
    log("run_request ->", {k: r[k] for k in ("reasoning", "text", "stop", "stop_sequence", "out_len")}, "| events", evs)
    assert r["stop"] == "stop_sequence" and r["stop_sequence"] == "END OF ANSWER" and r["text"] == "Answer: yes. ", r
    assert r["reasoning"] == "some reasoning" and r["out_len"] < len(script), r
    assert "".join(v for k, v in evs if k == "text") == "Answer: yes. ", evs
    evs = []
    r = ns["run_request"]([1, 2, 3], 100, 0.0, 1.0, None, "x", None, stops=["NOPE"], on_event=lambda k, v: evs.append((k, v)))
    assert r["stop"] == "stop" and r["text"] == "Answer: yes. END OF ANSWER more text" and r["stop_sequence"] is None, r
    # a forced tool call: the prefix goes through the parser first (thinking closed, tool mode), then a scripted body
    body = tok("read_file", add_special_tokens=False).input_ids + [T["<arg_key>"]] + tok("path", add_special_tokens=False).input_ids + [T["</arg_key>"], T["<arg_value>"]] + tok("/etc/hostname", add_special_tokens=False).input_ids + [T["</arg_value>"], T["</tool_call>"], T["<|observation|>"]]
    script = body
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    r = ns["run_request"]([1, 2, 3], 100, 0.0, 1.0, None, "x", tools, prefix=ns["forced_prefix"]("any"))
    log("forced any ->", r["calls"], r["stop"])
    assert len(r["calls"]) == 1 and r["calls"][0]["name"] == "read_file" and r["calls"][0]["input"] == {"path": "/etc/hostname"}, r
    script = body[len(tok("read_file", add_special_tokens=False).input_ids):]
    r = ns["run_request"]([1, 2, 3], 100, 0.0, 1.0, None, "x", tools, prefix=ns["forced_prefix"]("read_file"))
    assert len(r["calls"]) == 1 and r["calls"][0]["name"] == "read_file" and r["calls"][0]["input"] == {"path": "/etc/hostname"}, r
finally:
    ns["generate"] = real_generate
log("=== tool_choice through the HTTP APIs (tiny random model: only the prompt shapes are checked)")
tools_a = [{"name": "read_file", "description": "Read a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
base = {"max_tokens": 6, "messages": [{"role": "user", "content": "Read /etc/hostname."}], "tools": tools_a}
r_auto = json.loads(post("/v1/messages", base).read())
r_none = json.loads(post("/v1/messages", {**base, "tool_choice": {"type": "none"}}).read())
r_any = json.loads(post("/v1/messages", {**base, "tool_choice": {"type": "any"}}).read())
r_tool = json.loads(post("/v1/messages", {**base, "tool_choice": {"type": "tool", "name": "read_file"}}).read())
log("input tokens auto/none/any/tool:", [x["usage"]["input_tokens"] for x in (r_auto, r_none, r_any, r_tool)], "stop", r_tool["stop_reason"])
assert r_none["usage"]["input_tokens"] < r_auto["usage"]["input_tokens"] == r_any["usage"]["input_tokens"] == r_tool["usage"]["input_tokens"]
r_o = json.loads(post("/v1/chat/completions", {"max_tokens": 6, "messages": [{"role": "user", "content": "Read /etc/hostname."}],
                                                "tools": [{"type": "function", "function": t} for t in [{"name": "read_file", "parameters": {}}]],
                                                "tool_choice": {"type": "function", "function": {"name": "read_file"}}, "stop": ["zzz"]}).read())
log("openai forced ->", r_o["choices"][0]["finish_reason"], r_o["usage"])
log("=== keep-alive pings while a tool call is buffered (scripted slow generate, KEEPALIVE_S 0.05)")
body = tok("read_file", add_special_tokens=False).input_ids + [T["<arg_key>"]] + tok("path", add_special_tokens=False).input_ids + [T["</arg_key>"], T["<arg_value>"]] + tok("/etc/hostname", add_special_tokens=False).input_ids + [T["</arg_value>"], T["</tool_call>"], T["<|observation|>"]]
script = tok("brief thought", add_special_tokens=False).input_ids + [T["</think>"], T["<tool_call>"]] + body
def slow_generate(prompt, max_new, temperature, top_p, on_token=None, imgs=None, rid=None, on_idle=None, budget=None):
    for t in script:
        time.sleep(0.03)
        on_token(t)
    return script, 0.1, 0.5, 0, "stop"
keep = ns["KEEPALIVE_S"]
ns["generate"], ns["KEEPALIVE_S"] = slow_generate, 0.05
try:
    lines = []
    with post("/v1/messages", {"max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "Read /etc/hostname."}], "tools": tools_a}) as r:
        lines = [line.decode().rstrip() for line in r]
    pings = sum(1 for l in lines if l == "event: ping")
    evs = [json.loads(l[6:]) for l in lines if l.startswith("data: ")]
    types = [e["type"] for e in evs]
    log("anthropic: pings", pings, "| events", types)
    assert pings >= 2 and types[-1] == "message_stop" and all(e["type"] == "ping" for e in evs if "ping" in e.get("type", "")), lines
    assert any(e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use" and e["content_block"]["name"] == "read_file" for e in evs), evs
    assert [e for e in evs if e["type"] == "message_delta"][0]["delta"]["stop_reason"] == "tool_use"
    with post("/v1/chat/completions", {"stream": True, "max_tokens": 64, "messages": [{"role": "user", "content": "Read /etc/hostname."}],
                                       "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]}) as r:
        lines = [line.decode().rstrip() for line in r]
    comments = sum(1 for l in lines if l.startswith(": keep-alive"))
    chunks = [json.loads(l[6:]) for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    log("openai: keep-alive comments", comments, "| chunks", len(chunks), "| finish", chunks[-1]["choices"][0]["finish_reason"])
    assert comments >= 2 and "data: [DONE]" in lines and chunks[-1]["choices"][0]["finish_reason"] == "tool_calls", lines
    assert any("tool_calls" in c["choices"][0]["delta"] for c in chunks), chunks
finally:
    ns["generate"], ns["KEEPALIVE_S"] = real_generate, keep
log("=== thinking budget: </think> forced after budget_tokens (real scheduler, tiny engine)")
r = json.loads(post("/v1/messages", {"max_tokens": 24, "temperature": 0, "messages": [{"role": "user", "content": "Think hard about the sea."}],
                                     "thinking": {"type": "enabled", "budget_tokens": 1}}).read())
kinds = [c["type"] for c in r["content"]]
log("budget 1 ->", kinds, "| thinking", repr(r["content"][0].get("thinking", ""))[:100], "| usage", r["usage"], r["stop_reason"])
assert kinds[0] == "thinking" and "thinking budget is used up" in r["content"][0]["thinking"], r["content"]
assert "text" in kinds and r["usage"]["output_tokens"] > len(ns["THINK_WRAP"]) + 1, r
r = json.loads(post("/v1/messages", {"max_tokens": 24, "temperature": 0, "messages": [{"role": "user", "content": "Think hard about the sea."}],
                                     "thinking": {"type": "enabled", "budget_tokens": 4096}}).read())
assert "thinking budget is used up" not in (r["content"][0].get("thinking") or ""), r["content"]      # not reached: untouched
log("=== completions endpoint: stop on a substring of the greedy text (cancel path through the scheduler)")
r1 = json.loads(post("/v1/completions", {"prompt": "Once upon a time", "max_tokens": 16, "temperature": 0}).read())
text = r1["choices"][0]["text"]
assert len(text) > 6, text
stop = text[3:6]
r2 = json.loads(post("/v1/completions", {"prompt": "Once upon a time", "max_tokens": 16, "temperature": 0, "stop": stop}).read())
log("greedy text", repr(text), "| stop", repr(stop), "->", repr(r2["choices"][0]["text"]), r2["choices"][0]["finish_reason"], r2["usage"])
assert r2["choices"][0]["text"] == text[:text.index(stop)] and r2["choices"][0]["finish_reason"] == "stop"
assert r2["usage"]["completion_tokens"] < r1["usage"]["completion_tokens"]
time.sleep(1)
assert len(SCHED.active) == 0
log("=== bounded queue: 2 streams + 1 waiting -> the 4th concurrent request gets 429")
codes = [None] * 4
def fire(i):
    try:
        post("/v1/chat/completions", {"messages": [{"role": "user", "content": f"Story {i}, please."}], "max_tokens": 24}).read(); codes[i] = 200
    except urllib.error.HTTPError as e:
        codes[i] = e.code
assert SCHED.pause(30)                         # nothing is admitted while the four requests land in the queue
ths = [threading.Thread(target=fire, args=(i,)) for i in range(4)]
[t.start() for t in ths]; time.sleep(1.0); SCHED.resume(); [t.join() for t in ths]
log("codes ->", codes, "| rejected", STATE.get("rejected"))
assert sorted(codes) == [200, 200, 200, 429], codes
log("=== saturated OpenAI streaming request returns an SSE error, not a second HTTP response")
stream_payloads = [None] * 4
def fire_openai_stream(i):
    with post("/v1/chat/completions", {"stream": True, "max_tokens": 24,
                                        "messages": [{"role": "user", "content": f"Stream {i}, please."}]}) as response:
        stream_payloads[i] = response.read().decode()
assert SCHED.pause(30)
ths = [threading.Thread(target=fire_openai_stream, args=(i,)) for i in range(4)]
[t.start() for t in ths]; time.sleep(1.0); SCHED.resume(); [t.join() for t in ths]
openai_errors = [p for p in stream_payloads if '"error"' in p]
log("OpenAI error events", len(openai_errors))
assert len(openai_errors) == 1 and "data: [DONE]" in openai_errors[0], stream_payloads
assert all("HTTP/1." not in p for p in stream_payloads), stream_payloads
log("=== saturated Anthropic streaming request returns an error event")
stream_payloads = [None] * 4
def fire_anthropic_stream(i):
    with post("/v1/messages", {"stream": True, "max_tokens": 24,
                                "messages": [{"role": "user", "content": f"Stream {i}, please."}]}) as response:
        stream_payloads[i] = response.read().decode()
assert SCHED.pause(30)
ths = [threading.Thread(target=fire_anthropic_stream, args=(i,)) for i in range(4)]
[t.start() for t in ths]; time.sleep(1.0); SCHED.resume(); [t.join() for t in ths]
anthropic_errors = [p for p in stream_payloads if "event: error" in p]
log("Anthropic error events", len(anthropic_errors))
assert len(anthropic_errors) == 1 and '"type": "rate_limit_error"' in anthropic_errors[0], stream_payloads
assert all("HTTP/1." not in p for p in stream_payloads), stream_payloads
h = json.loads(urllib.request.urlopen(URL + "/health").read())
log("health", {k: v for k, v in h.items() if k in ("active", "pending", "live_contexts", "max_streams", "max_sets", "max_queue", "piece", "buckets", "requests", "steps", "step_tokens", "snap_parks", "snap_hits", "rejected")})
log("HARNESS OK")
os._exit(0)
