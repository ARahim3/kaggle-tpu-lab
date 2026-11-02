#!/usr/bin/env python3
"""Generate a model's run-all Kaggle notebook from its kernel script (and engine package, if it has one), so the
notebook and the kernel always ship the same code.

    python tools/pack_notebook.py glm53-flash      # writes glm53-flash/notebook/glm53-tpu-serve.ipynb

The notebook: an intro, a config cell (serve_config.json), one cell that writes the engine files as plain text
(nothing to edit there), the serving script, and the launch cell that is the server.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INTRO = {"glm53-flash": """# GLM-5.3-Flash on a free Kaggle TPU v5e-8

This notebook serves **GLM-5.3-Flash**, a 320B-parameter mixture-of-experts model (18B active), on Kaggle's free
TPU and gives you a public endpoint that speaks both the OpenAI and the Anthropic API. Use it from your laptop,
Claude Code, Codex CLI, opencode, or plain `curl`.

As far as we know this is the first inference engine to run GLM-5.3-Flash on a TPU, and it runs on the free one.
We wrote it in JAX for this model: the experts sit in HBM as 3-bit weights that custom TPU kernels read straight
into the matrix multiply, and a small draft model proposes blocks of tokens that the big model verifies in one step.
Measured here: about **165 tok/s** for one stream, about **170 tok/s** aggregate over three, the model's full
**262,144-token context**, prefix caching that resumes an agent session in about a second, and image inputs.

## Before you run: three clicks in the right sidebar
1. **Accelerator → TPU VM v5e-8**
2. **Internet → ON** (the tunnel needs it)
3. **Add Input** → attach these three datasets:
   - `rahim3/glm53-flash-iq3xxs-1` and `rahim3/glm53-flash-iq3xxs-2`: the experts (Unsloth's UD-IQ3_XXS GGUF)
   - `rahim3/glm53-flash-serve-v3`: the rest of the weights, the vision tower, the tokenizer and the programs

Then run all cells. The last cell is the server and keeps running on purpose; the endpoint URL and the API key
appear in its `READY` banner after about 6–13 minutes (the weights are most of that). The draft model
(`incoai/GLM-5.3-Flash-DFlash2`, CC BY-NC-ND 4.0, research use) comes from Hugging Face when the server starts;
set `dflash` to `""` in the config cell to run without it, at about 79 tok/s.

If the last cell stops in its first minute, its message says why: no TPU in this session (Kaggle does that
sometimes; stop and start the session again), Internet off, or a dataset not attached.

The code, the engine and a guide for the same recipe on a rented TPU VM are in the GitHub repo:
https://github.com/ARahim3/kaggle-tpu-lab. Keep an eye on it for more models in the future!
""",
         "qwen38-27b": """# Qwen3.8-27B on a free Kaggle TPU v5e-8 🚀

This notebook serves **Qwen3.8-27B (bf16)** with vLLM on Kaggle's free TPU and gives you a
public OpenAI-compatible endpoint you can use from anywhere — your laptop, Claude Code,
Codex CLI, opencode, or plain `curl`.

Measured on this exact setup: **~130 tok/s** single-stream decode (verified-lossless MTP
speculative decoding), **~540 tok/s** aggregate across 8 streams, **10,000+ tok/s prefill**,
context up to the model's native **262,144 tokens**.

## Before you run — 3 clicks in the right sidebar
1. **Accelerator → TPU VM v5e-8** (Session options)
2. **Internet → ON** (Session options — needed for the tunnel)
3. **Add Input** → search and attach these two datasets:
   - `rahim3/qwen3-8-27b-bf16` — the model weights (skips a 55 GB download)
   - `rahim3/qwen38-tpu-env-v5e8` — pre-compiled TPU graphs for this exact setup
     (**~22 min to a live endpoint** instead of ~50; ~12 min with `"text_only": true`,
     ~6 min with `"fast_start": true`)

Then run the cells top to bottom. The last cell keeps running **on purpose** — that's your
server. The endpoint URL + API key appear in a `READY` banner in its output.
""",
         "qwen38-27b-gpu": """# Qwen3.8-27B on a free Kaggle GPU (T4 x2) 🚀

This notebook serves **Qwen3.8-27B (AWQ-INT4)** with stock vLLM on Kaggle's free GPU quota
and gives you a public OpenAI-compatible endpoint. Measured on this exact shape:
**READY in 5:29** with the compile-cache dataset (13:26 cold), **61 tok/s** at 4 streams,
19.4–23.4 tok/s single-stream.

**This is a batch/regression lane, not the interactive one.** The concurrency-8 p99 was
16.8 s (measured) — for chat and agents use the TPU notebook (`qwen38-tpu-serve.ipynb`)
or the Modal appliance (`modal/`). Use this lane to soak tests, run regression suites and
batch evals: it draws **GPU quota (30 h/week)** and leaves your scarcer **TPU quota
(20 h/week)** untouched, and sessions cap at 12 h.

The 54 GB bf16 checkpoint cannot fit 32 GB across two T4s, so this serves the community
AWQ-INT4 quant (~16 GB) with tensor parallelism 2. AWQ dequant runs on Turing; the
flashinfer/marlin paths need SM80+, so this is stock vLLM — no MTP speculative decoding.

## Before you run — 3 clicks in the right sidebar
1. **Accelerator → GPU T4 x2** (Session options)
2. **Internet → ON** (Session options — needed for the tunnel)
3. **Add Input** (optional but recommended):
   - a weights dataset with the AWQ checkpoint's `config.json` + `*.safetensors`
     (skips a ~16 GB Hugging Face download; set its owner/slug in the config cell)
   - a compile-cache dataset (from a previous session's `vllm_cache.tar` output —
     engine init drops from 665 s to 182 s; set `vllm_cache_dataset` in the config cell)

Then run the cells top to bottom. The last cell keeps running **on purpose** — that's your
server. The endpoint URL + API key appear in a `READY` banner in its output.
""",
}

CONFIG = {"glm53-flash": {"streams": 3, "max_len": 262144, "reasoning_effort_default": "low", "vision": True, "keepalive_min": 90},
          "qwen38-27b": {"max_model_len": 262144, "max_num_seqs": 4, "mtp_tokens": 3,
                         "reasoning_effort_default": "xhigh", "keepalive_min": 90},
          "qwen38-27b-gpu": {"max_model_len": 16384, "max_num_seqs": 4, "gpu_mem_util": 0.88,
                             "keepalive_min": 120}}

CONFIG_NOTES = {"glm53-flash": """### Configuration
The defaults above are what we serve. Things you might change:
- `streams`: how many requests decode together. Three fit in HBM at the full 262k context; a fourth request waits
  for a free slot. One stream runs at about 165 tok/s, three together at about 170 tok/s aggregate.
- `dflash`: where the draft model comes from: `hf:<repo>@<revision>` (the default, fetched from Hugging Face), an
  attached dataset or directory holding its `config.json` and `model.safetensors`, or `""` for no speculative
  decoding.
- `reasoning_effort_default`: `"low"`, `"medium"` or `"high"`; clients can still set it per request.
- `vision`: `false` skips the vision tower (a minute and a little HBM saved; image inputs then error out).
- `keepalive_min`: the server shuts itself down after this long. Raise it (up to about 480) for a long-lived
  endpoint; the default keeps a test run cheap on your TPU quota.
- `api_key`: set your own; otherwise one is generated and printed in the banner.
- `ntfy_topic`: optional; the script posts its progress to `ntfy.sh/<topic>` so you can follow it from your phone.
""",
            "qwen38-27b": """### Configuration
The defaults above are the sweet spot from our benchmarks. Things you might change:
- `max_model_len`: defaults to the native `262144`; for many parallel streams use
  `131072` with `"max_num_seqs": 16`. Both of these configs have pre-compiled graphs in
  the env dataset; any other combination still works but compiles cold (+~10 min)
- `mtp_tokens`: `3` = MTP speculative decoding (+34% in our A/B test; ~130 tok/s single-stream). The runtime includes a fix
  (port of tpu-inference PR #3178) that makes it lossless on TPU — verified 12/12 greedy
  exact-match. `0` disables
- `reasoning_effort_default`: `"xhigh"` (default), `"medium"`, or `"low"` — clients can
  still override per request with `chat_template_kwargs`
- `keepalive_min`: the server shuts itself down after this long — raise it (up to ~480)
  when you want a long-lived endpoint; the default keeps a test run cheap on your TPU quota
- `"text_only": true` skips Qwen3.8's vision tower: ~10 min faster start, but image inputs
  then error out. The default keeps images working (screenshots in coding agents, etc.);
  note the first image at a new resolution takes ~1 min to compile (later ones are instant)
- `"fast_start": true` skips the TPU graph precompile: the endpoint is live in ~4 min and
  the script then warms the common request shapes itself (short/long prompts, a few
  parallel streams). An unusual request shape waits ~1 min the first time it is seen
- `"async_scheduling": false` passes `--no-async-scheduling` to vLLM. Set it if your client uses JSON mode /
  structured outputs while `mtp_tokens` > 0: vllm-tpu 0.28.0 otherwise exits with `AttributeError: __delitem__`
  on the first such request. It costs some throughput
- `"idle_min": 15` shuts the server down after this many idle minutes (no requests running
  or queued) to give the TPU slot back — set `0` to serve the full `keepalive_min` window
  regardless of traffic. The heartbeat lines show idle time and the fraction of polls that
  were busy, so you can see whether the session is earning its quota
- `"harvest_cache": true` tars the completed XLA compile cache into the session output on
  exit — publish it as a private dataset to make your next cold start restore a cache built
  for *your* exact config
- `"burst_gateway"` / `"burst_instance"` / `"burst_gateway_token"`: if you run the Burst
  control plane (github.com/aliasfoxkde/Burst-Compute-Testing), set these and the kernel
  announces itself to `POST /admin/endpoints` on READY and on heartbeats so the gateway can
  adopt the session; `burst_credential_ref` / `burst_model` fill the payload
- `"verbose": true` shows every vLLM log line (they are always saved to `vllm.log` anyway)
""",
            "qwen38-27b-gpu": """### Configuration
The defaults above are the measured T4x2 sweet spot. Things you might change:
- `max_model_len` / `max_num_seqs`: 16384 × 4 is the measured fit (32k × 8 OOM'd both
  T4s: 14.56 GiB usable per card, ~9.8 GiB of AWQ pack per card with TP=2). Raise the
  context only if you drop the sequences to match.
- `gpu_mem_util`: 0.88 leaves room for the AWQ `create_weights` overshoot.
- `weights_dataset`: owner/slug of a mounted AWQ weights dataset — skips a ~16 GB
  Hugging Face download (the biggest cold-start lever after the cache).
- `vllm_cache_dataset`: owner/slug of a previous session's compile-cache output
  (`vllm_cache.tar` + `manifest.json` published as a dataset). Engine init 665 → 182 s
  measured. The manifest's versions (vllm / torch / python) are checked against the
  runtime; a mismatch still runs, but loudly warns the cache may miss.
- `enforce_eager`: `true` skips CUDA-graph capture (shorter startup, slower decode).
- `harvest`: on the exit path the script tars the torch.compile cache and writes a
  `manifest.json` into the session output — publish them as a dataset to make your next
  boot the warm 5:29 one.
- `keepalive_min` / `idle_min`: GPU sessions cap at 12 h; the script also exits after
  10 idle minutes (no request running or queued) to stop the meter.
- `api_key`: set your own; otherwise one is generated and printed in the banner.
- `ntfy_topic`: optional; the script posts its progress to `ntfy.sh/<topic>`.
""",
}

LAUNCH_NOTES = {"glm53-flash": """### Launch (leave this cell running: it is the server)
What you will see:
1. **Runtime**: a pinned TPU runtime and a few packages (about a minute)
2. **Weights**: the 3-bit experts and the int8 non-expert weights go straight onto the eight chips (3–10 minutes,
   depending on how fast the dataset mount reads)
3. **Vision**: the vision tower (seconds), then the draft model from Hugging Face (about a minute)
4. **Warm-up**: every program is loaded in its exported and compiled form from the serve dataset (under a minute);
   without the matching dataset every program is traced and compiled (about 28 minutes)
5. **Tunnel**: a public URL
6. **READY banner** with `ENDPOINT`, `API KEY` and `MODEL`, then a short self-test

Use the endpoint from anywhere:
```bash
curl <ENDPOINT>/v1/chat/completions -H "Authorization: Bearer <API_KEY>" \\
  -H "Content-Type: application/json" -d '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
Claude Code (the banner prints this line filled in):
```bash
ANTHROPIC_BASE_URL=<ENDPOINT> ANTHROPIC_AUTH_TOKEN=<API_KEY> ANTHROPIC_MODEL=glm-5.3-flash \\
ANTHROPIC_SMALL_FAST_MODEL=glm-5.3-flash CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144 claude
```
""",
           "qwen38-27b": """### Launch 🎬 (leave this cell running — it IS the server)
What you'll see, step by step (times with both datasets attached):
1. **Runtime** — a TPU check first (Kaggle sometimes starts a session without one; the script stops
   with a clear message instead of failing minutes later), then `uv` builds a venv with vllm-tpu (~30 s)
2. **Compile cache** — restore the pre-compiled TPU graphs (seconds)
3. **Weights** — found on the mounted dataset
4. **Starting vLLM** — reads 55 GB of weights onto the 8 chips (~3 min), then compiles/loads
   the TPU graphs (~17 min cached, ~35 min cold; ~10 min less with `"text_only": true`)
5. **Public URL** — printed early so you can set up your client; it answers 502 until…
6. **READY banner** with `ENDPOINT` / `API KEY` / `MODEL`, then a short self-test.

The server then keeps running until `keepalive_min` elapses, `idle_min` idle minutes pass
(no requests for 15 min by default — it exits to free the TPU slot), or you stop the session.
Only a few progress lines are shown; the complete vLLM log is in `/kaggle/working/vllm.log`.

Use the endpoint from anywhere:
```bash
curl <ENDPOINT>/chat/completions -H "Authorization: Bearer <API_KEY>" \\
  -H "Content-Type: application/json" -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "chat_template_kwargs": {"reasoning_effort": "low"}
  }'
```
""",
           "qwen38-27b-gpu": """### Launch (leave this cell running: it IS the server)
What you will see (times with cache + weights datasets attached; cold ≈ G1 column):
1. **GPU check** — the granted cards and their compute capability are logged first
2. **Runtime** — a venv with stock vLLM 0.29.0 via `uv` (CUDA torch wheels, ~2–4 min)
3. **Weights + cache** — found on the mounted datasets, else the AWQ downloads (~16 GB)
4. **Starting vLLM** — TP=2, then torch.compile (182 s warm cache / 665 s cold) and CUDA graphs
5. **Public URL** — printed early; it answers 502 until…
6. **READY banner** with `ENDPOINT` / `API KEY` / `MODEL qwen38-27b-awq`, then a self-test
   (~19–23 tok/s single stream)

Use the endpoint from anywhere:
```bash
curl <ENDPOINT>/v1/chat/completions -H "Authorization: Bearer <API_KEY>" \\
  -H "Content-Type: application/json" -d '{
    "model": "qwen38-27b-awq",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

On exit (keepalive, idle, or you stop the session) the script harvests
`vllm_cache.tar` + `manifest.json` into the session output — publish them as a private
dataset and attach it next time (set `vllm_cache_dataset`) for the 5:29 warm start.
""",
}

SCRIPT_NOTES = {"glm53-flash": """### The serving script
The next cell writes the serving script (the same one `launch.py` pushes; the source and docs are in the [kaggle-tpu-lab repo](https://github.com/ARahim3/kaggle-tpu-lab)).
""",
                 "qwen38-27b": """### The serving script
The next cell writes the full serving script (same one the CLI launcher pushes —
see the [kaggle-tpu-lab repo](https://github.com/ARahim3/kaggle-tpu-lab) for the source and docs).
""",
                 "qwen38-27b-gpu": """### The serving script
The next cell writes the full serving script (same one `launch.py serve --model qwen38-27b-gpu` pushes —
see the [kaggle-tpu-lab repo](https://github.com/ARahim3/kaggle-tpu-lab) for the source and docs).
""",
}

def cell(kind, text):
    lines = text.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def engine_cell(engine_dir: Path, pkg: str):
    files = sorted(p for p in engine_dir.glob("*.py"))
    parts = [f"# The engine: the `{pkg}` package ({len(files)} files), generated from the repo's {engine_dir.relative_to(ROOT)}/ — nothing to edit here.\n",
             "import os\n", f'os.makedirs("{pkg}", exist_ok=True)\n', "FILES = {}\n"]
    for p in files:
        src = p.read_text()
        assert "'''" not in src, f"{p} contains ''' — the engine cell cannot embed it"
        assert not src.endswith("\\"), p
        parts.append(f'FILES["{pkg}/{p.name}"] = r\'\'\'{src}\'\'\'\n')
    parts.append("for path, src in FILES.items():\n    open(path, \"w\").write(src)\n")
    parts.append(f'print(f"wrote {{len(FILES)}} files of the {pkg} package")\n')
    return "".join(parts)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "glm53-flash"
    folder = ROOT / model
    kernel = next((folder / "kernel").glob("serve_*.py"))
    engine_dirs = list((folder / "engine").glob("*/__init__.py")) if (folder / "engine").is_dir() else []
    cells = [cell("markdown", INTRO[model]),
             cell("code", "%%writefile serve_config.json\n" + json.dumps(CONFIG[model], indent=2) + "\n"),
             cell("markdown", CONFIG_NOTES[model])]
    for init in engine_dirs:
        cells.append(cell("markdown", "### The engine\nOne cell that writes the engine package next to the script. It is generated from the repo, so there is nothing to edit here; scroll past it.\n"))
        cells.append(cell("code", engine_cell(init.parent, init.parent.name)))
    cells.append(cell("markdown", SCRIPT_NOTES[model]))
    cells.append(cell("code", f"%%writefile {kernel.name}\n" + kernel.read_text()))
    cells.append(cell("markdown", LAUNCH_NOTES[model]))
    cells.append(cell("code", f"!python {kernel.name}\n"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = folder / "notebook" / f"{model.split('-')[0]}-{'gpu' if model.endswith('-gpu') else 'tpu'}-serve.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(cells)} cells, {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
