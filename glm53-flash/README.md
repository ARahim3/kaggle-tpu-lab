# GLM-5.3-Flash on a free Kaggle TPU v5e-8

**Serve GLM-5.3-Flash, a 320B mixture-of-experts model (18B active), on Kaggle's free TPU, with a
public endpoint that speaks both the OpenAI and the Anthropic API.** Point Claude Code, Codex CLI,
opencode or `curl` at it.

As far as we know, this is the first inference engine to run GLM-5.3-Flash on a TPU at all, and it
runs on the free one. Nothing existing could: vLLM's TPU backend has no kernels for the model's
linear-attention layers, and no format of the weights fits eight 16 GB chips. So we wrote an engine
in JAX for this model: the routed experts live in HBM as 3-bit codebook weights, custom Pallas
kernels read them straight into the matrix multiply, and everything else is int8. What you get:

| What | Measured (TPU v5e-8) |
|---|---|
| Decode, one stream | **~64 tok/s** at any context length up to 262k |
| Decode, three streams | **~30 tok/s each**, ~90 tok/s aggregate |
| Prefill | **~1,500–1,800 tok/s**; a 27k-token prompt in 15–17 s, 260k in ~3 min |
| Context | the model's native **262,144 tokens** (needle test passes at 260k) |
| Agent sessions | a 27k-token conversation resumes in about a second after an interleaved call |
| Images | supported in both APIs (screenshots from a coding agent, for example) |
| Time to live endpoint | **~16 min** with the serve dataset attached (weights 4 min, warm-up 9 min, the rest is the runtime, the tunnel and Kaggle's own start); ~22 min from the FP8 checkpoint |

The output is that of a 3-bit-expert quantization, not the bf16 model. Agent tasks, long-context
retrieval and tool calling behave as expected in our use; a formal comparison against the full
model is on the list.

## Quick start A — as a Kaggle notebook

**Copy & Edit** [the published notebook](https://www.kaggle.com/code/rahim3/glm-5-3-flash-on-a-free-kaggle-tpu-64-tok-s-api) and run it, or upload
[`notebook/glm53-tpu-serve.ipynb`](notebook/glm53-tpu-serve.ipynb) yourself. Set
**Accelerator = TPU VM v5e-8**, **Internet = ON**, attach the datasets named in the first cell,
and run top to bottom. The last cell *is* the server; the endpoint URL and API key appear in its
output.

## Quick start B — from your terminal

```bash
pip install kaggle          # one-time; then put your API token at ~/.kaggle/kaggle.json
git clone https://github.com/ARahim3/kaggle-tpu-lab
cd kaggle-tpu-lab
python launch.py serve --model glm53-flash
```

The launcher pushes the kernel, follows its progress and prints the endpoint and key when the
server is up. `python launch.py status` re-attaches, `python launch.py stop` ends the session.

## Using it with coding agents

The banner prints the exact lines. Claude Code talks to `/v1/messages` (thinking arrives as
`thinking` blocks, tool calls as `tool_use` blocks); authenticate with `ANTHROPIC_AUTH_TOKEN`, not
`ANTHROPIC_API_KEY`, and tell Claude Code the real context window:

```bash
export ANTHROPIC_BASE_URL="https://<your-tunnel>.trycloudflare.com"
export ANTHROPIC_AUTH_TOKEN="glm-<your-key>"
export ANTHROPIC_MODEL="glm-5.3-flash"
export ANTHROPIC_SMALL_FAST_MODEL="glm-5.3-flash"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144
claude
```

Codex CLI, opencode, aider and anything else OpenAI-compatible:

```bash
export OPENAI_BASE_URL="https://<your-tunnel>.trycloudflare.com/v1"
export OPENAI_API_KEY="glm-<your-key>"
# model name: glm-5.3-flash
```

Per request you can set `reasoning_effort` (`low` or `high`; via `chat_template_kwargs` on the
OpenAI API or the `thinking` field on the Anthropic one), a thinking budget, `stop_sequences` /
`stop`, `tool_choice`, and temperature / top_p. The server default is `low` reasoning; change it
with `--reasoning-effort` or in the notebook's config cell.

The thinking budget is worth knowing about: GLM can reason for tens of thousands of tokens on an
open-ended task and run out of `max_tokens` before it answers. With `thinking.budget_tokens`
(Anthropic API; `thinking_budget` on the OpenAI one) the server closes the reasoning at the budget
with a short wrap-up line and the answer follows. pi and Claude Code send a budget for their
thinking levels, so those levels mean what they say here.

## The engine

The engine is about 5,400 lines of JAX and Pallas in [`engine/glm53/`](engine/glm53/), built for
this one model rather than as a framework. GLM-5.3-Flash has 45 layers: 34 are KDA linear
attention (a recurrent state per layer, no KV cache), 11 are multi-head latent attention with a
sparse top-k indexer, and 42 of them route each token to 8 of 288 experts plus one shared expert.
The routed experts are 95 % of the weights, so the whole design follows from where they can live.

- **Experts in HBM at 3 bits.** Unsloth's UD-IQ3_XXS GGUF gives 110 GB of expert weights for the
  whole model. We keep them on the chips in a planar layout, the codebook fields of each block
  split into dense 32-bit planes, and read them with Pallas kernels that dequantize inside the
  matrix multiply: a fused dequant-matvec for decode, where the cost is the number of distinct
  experts a step touches, and a grouped GEMM over the routed rows for prefill. The rest of the
  model is int8. After the build the chips hold 15.4 of their 16.9 GB.
- **The model itself**, written in JAX: the KDA recurrence with its chunked prefill, the latent
  attention with the sparse indexer, the multi-stream residual mixing, tensor-parallel over the
  eight chips. It matches the reference implementation on the CPU, and a tiny random model runs
  the same code on eight virtual CPU devices, which is how the 100-odd tests work without a TPU.
- **262k context on 16 GB chips.** The attention caches are sharded across the chips by position,
  the latent cache is int8 with a per-token scale, and prefill runs in 1024-token pieces with
  donated cache buffers, so no temporary grows with the prompt.
- **Continuous batching.** Several requests decode in one step, each with its own cache set; new
  requests are admitted between steps and prefilled a piece at a time so running streams keep
  flowing. Sampling happens on the chips too, so token ids, positions and the sampler state never
  leave the device between steps.
- **Prefix caching for agents.** A finished conversation stays on the chips for its next turn;
  when another session needs the room, it is parked in host memory as a compact snapshot and
  restored when its next turn arrives. System prompts are pinned, so a new session starts from
  them instead of re-reading them.
- **The vision tower** runs sharded on the chips as well (4–34 ms per image), so screenshots work
  in both APIs.
- **The server** speaks the OpenAI and Anthropic APIs with streaming, thinking blocks, tool calls,
  `tool_choice`, stop sequences and images, in [`kernel/serve_glm53.py`](kernel/serve_glm53.py).

## What's in this folder

```
README.md                           this file
kernel/serve_glm53.py               the Kaggle kernel: runtime → weights → vision → warm-up → tunnel → READY
notebook/glm53-tpu-serve.ipynb      the same flow as a run-it-yourself notebook (generated by ../tools/pack_notebook.py)
engine/glm53/                       the JAX engine: model, kernels, caches, scheduler, vision, tests
tools/bench_endpoint.py             latency / throughput / prefix-cache checks against a live endpoint
tools/bench_concurrent.py           serial vs concurrent streams
tools/check_features.py             stop sequences, tool_choice, thinking budget, keep-alive pings, the bounded queue
tools/harness_serve.py              the whole kernel on a tiny CPU model (40 s), for changes to the serving code
```

## Good to know

- **Three concurrent streams** is the HBM limit at the full context; a fourth request waits for a
  free slot (up to 90 s, then a 503). More than eight waiting requests get a 429.
- **Startup.** The weights take ~4 minutes: the base store in under a minute, then the experts, read from
  the dataset mount (~650 MB/s to parallel readers) and packed onto the chips in eight threads. The warm-up is ~9 minutes with
  the serve dataset's compile cache and ~14 without; what the cache cannot skip is JAX tracing the
  programs. A cold run leaves `base/` and `jax_cache/` in `/kaggle/working`, which is how the serve
  dataset is made.
- **Sessions.** Kaggle stops a TPU session after nine hours; the kernel shuts itself down after
  `keepalive_min` minutes so a forgotten run does not burn your quota. Each run gets a new
  tunnel URL. A fresh Cloudflare hostname can take a minute to resolve.
- **Stop sequences** apply to the text output, not to the reasoning.
- **Long tool calls and the tunnel.** Cloudflare closes a response that stays silent for about
  100 s. Thinking and text stream as they come, but a tool call is parsed whole, so while the model
  writes a big file into a tool argument, or while a request waits for a free slot, the server
  sends a keep-alive every 15 s (`ping` events on the Anthropic API, SSE comments on the OpenAI
  one). Non-streaming requests send nothing until they finish: stream anything that may take
  longer than ~100 s.
- **Quantization.** The experts are Unsloth's dynamic 3-bit mix (IQ2_S / IQ3_S / IQ4_XS by layer),
  the rest of the weights int8, the latent cache int8. The 260k needle test passes; we have not
  run a formal benchmark against the bf16 model yet.

## If it fails

The kernel looks before it installs anything, so the common problems show up in the first
minute with a plain message:

- **"this session has no working TPU"**: Kaggle started the session without a TPU attached. It
  happens, most often on new or not-yet-verified accounts, and nothing in the notebook can fix it.
  Stop the session and start it again; `import jax; print(jax.device_count())` in a fresh cell
  must print 8. Some accounts are also asked for full identity verification (KYC via Persona)
  before Kaggle grants TPU access — phone verification alone is not always enough.
- **"no Internet from this session"**: turn Internet on in Session options. Kaggle only allows
  that on phone-verified accounts.
- **"datasets not attached"**: Add Input in the right sidebar and attach the names it lists (the
  two expert datasets and the serve dataset).
- **The tunnel gives no URL**: the kernel tries three times. If it still fails, the server is
  running but only reachable inside the kernel; start the session again. A fresh URL can take a
  minute to resolve, and the kernel replaces one that never does, so watch the log for a
  `NEW ENDPOINT` line before giving up.
- **The server stopped on its own**: it exits after `keepalive_min` minutes (8 hours by default)
  and Kaggle ends TPU sessions after nine hours.
- **Anything else**: the log prints the step it was in and the error; paste that into an issue.

## Credits

- [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) for GLM-5.3-Flash, released under MIT.
- [Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) for the UD-IQ3_XXS quantization
  the experts are loaded from.
- JAX and Pallas, which made writing TPU kernels for a model no stack supported a
  two-week job instead of a research project.

## License

The code in this folder is MIT. The model weights keep their own licenses.
