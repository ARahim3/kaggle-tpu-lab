# kaggle-tpu-lab

Run open models on Kaggle's free TPU v5e-8 and get a public endpoint that speaks the
OpenAI and Anthropic APIs. Point Claude Code, Codex CLI, opencode or anything else at it.
No GPU, no cloud bill, about twenty minutes from pressing Run to a URL.

Each model has its own folder with a run-all Kaggle notebook, the kernel script behind
it, and a write-up of how it works and what we measured.

| Model | Weights on the TPU | Context | One stream | Many streams | Prefill | Run → URL | Engine | |
|---|---|---|---|---|---|---|---|---|
| [Qwen3.8-27B](qwen38-27b/) | bf16, no quantization | 262k | ~130 tok/s | ~540 tok/s at 8 | 10,300 tok/s | ~22 min | vllm-tpu + one patch | [notebook](https://www.kaggle.com/code/rahim3/qwen3-8-27b-bf16-on-kaggle-tpu-130-tok-s-api) |
| [GLM-5.3-Flash](glm53-flash/) (320B MoE) | 3-bit experts, int8 rest | 262k | ~64 tok/s | ~90 tok/s at 3 | ~1,600 tok/s | ~16 min | our own JAX engine | [notebook](https://www.kaggle.com/code/rahim3/glm-5-3-flash-on-a-free-kaggle-tpu-64-tok-s-api) |

Numbers are measured on the shipped configuration; the folder READMEs say how. Qwen runs on
vllm-tpu with one patch. GLM-5.3-Flash runs on an engine we wrote in JAX for it; as far as we
know it is the first to run that model on a TPU.

## What you need

A Kaggle account with TPU access (phone-verify it under Settings) and its free quota,
around 20 TPU hours a week. Note that some accounts are also asked for full identity
verification (KYC via Persona) before Kaggle grants TPU access — phone verification
alone is not always enough. Nothing to install for the notebook route. For the terminal
route, Python 3.9+ and the Kaggle CLI.

## How a session works

The notebook attaches public datasets holding the weights (and, where it helps, a
pre-built compile cache), builds the engine across the eight chips, opens a Cloudflare
tunnel and prints the URL and an API key. The last cell is the server: leave it running.
A keepalive holds the session up to Kaggle's limit, a little under nine hours, after
which you run it again and get a new URL.

Wire a coding agent with one line, for example Claude Code:

```bash
ANTHROPIC_BASE_URL=<url> ANTHROPIC_AUTH_TOKEN=<key> ANTHROPIC_MODEL=<model> claude
```

Each folder README has the exact lines for Claude Code, Codex CLI and opencode.

## From a terminal

```bash
git clone https://github.com/ARahim3/kaggle-tpu-lab
cd kaggle-tpu-lab
python launch.py serve            # Qwen3.8-27B today; --model picks another recipe once there is one
```

`launch.py` pushes the kernel with the Kaggle CLI and follows its progress; `status`
and `stop` do what they say.

## Adding a model

One folder at the top level, named after the model: `README.md` with the numbers and the
how, `kernel/` with the serving script, `notebook/` with the run-all notebook generated
from it, plus whatever the recipe needs (a patch, an engine). The launcher and the
notebook share the same config block, so a setting changed in one place means the same
thing in the other.

## License

The code here is MIT. Model weights keep their own licenses; each folder says which.
