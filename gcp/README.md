# Running the recipe on a Google Cloud TPU VM

Kaggle's free TPU session is one host with eight TPU v5e chips, a `v5litepod-8`. Google Cloud rents the same machine,
and its faster sibling the `v6e-8`, by the hour: no queue, no nine-hour cap, a VM you can SSH into. This folder is how
we run the GLM-5.3-Flash recipe there, which we do daily for development: the same kernel script, the same public
datasets, Kaggle's paths recreated on the VM. It answers [issue #11](https://github.com/ARahim3/kaggle-tpu-lab/issues/11).
An optional [Terraform module](terraform/README.md) creates an eight-chip Spot TPU VM; use the runbook below to
install the runtime and start serving. It uses direct provisioning, so use the queued-resource `gcloud` commands below
if the selected zone has no immediate capacity.

**You need**: a Google Cloud project with billing, the `gcloud` CLI, TPU quota in one zone (below), and a Kaggle API
token (the datasets are public, the Kaggle CLI still wants a token to download them).

## Which TPU

| | `v5litepod-8` | `v6e-8` |
|---|---|---|
| Chips | 8 × TPU v5e, 16 GB HBM each: Kaggle's machine | 8 × TPU v6e, 32 GB each |
| The recipe | identical to Kaggle, including the serve dataset's exported and compiled programs | runs unchanged, about 1.3× faster; the exported and compiled programs are per chip generation, so the first build traces and compiles every program (~25 min to READY); later starts on the same VM load them (about 2 min) |
| `--runtime-version` | `v2-alpha-tpuv5-lite` | `v2-alpha-tpuv6e` |
| Quota to request | "Preemptible TPU v5 lite pod cores **for serving**" in your zone: a single-host v5e counts as a serving TPU, not against the pod quota | "Preemptible TPU v6e cores" (no serving split) |

**Eight chips are required.** The routed experts are sharded over eight devices (13.8 GB of 3-bit weights per chip)
and the engine assumes eight. A `v6e-4` does not fit the experts and is not supported.

Spot ("preemptible") VMs cost a fraction of on-demand ones; see the TPU pricing page for your region. A spot VM is
reclaimed without notice, in our experience several times a day, and everything on it is lost: the scripts below are
idempotent, so a restart is the same three commands again. The datasets live in RAM by default (the download is the
slow part, 10 to 30 minutes); a persistent disk for them (`DATA_DISK` below) survives preemptions at a monthly cost.

## 1. Get a VM

A queued request waits for capacity instead of failing:

```bash
gcloud config set project <your-project>
Z=<zone with quota>            # e.g. us-east1-d or europe-west4-a for v6e; check the quota page for v5e zones
gcloud compute tpus queued-resources create glm-q --zone=$Z --node-id=glm-tpu \
  --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e --spot
#   v5e: --accelerator-type=v5litepod-8 --runtime-version=v2-alpha-tpuv5-lite
gcloud compute tpus queued-resources list --zone=$Z      # WAITING_FOR_RESOURCES -> PROVISIONING -> ACTIVE
gcloud compute tpus tpu-vm ssh glm-tpu --zone=$Z
```

Minutes to hours until ACTIVE, depending on the zone's spare capacity. After a preemption the request shows
`SUSPENDED` and the VM `PREEMPTED`: delete the request (`gcloud compute tpus queued-resources delete glm-q --zone=$Z
--force`) and create it again. **A TPU VM bills while it exists, idle or not**: delete the request when you are done.

## 2. Set up the VM

```bash
git clone https://github.com/ARahim3/kaggle-tpu-lab && cd kaggle-tpu-lab/gcp
./setup.sh            # Kaggle's Python 3.12 / JAX 0.10.2 / libtpu pins in a venv, /kaggle paths, the datasets' space
mkdir -p ~/.kaggle && nano ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token   # your Kaggle API token
./download.sh         # ~130 GB from three public datasets, in parallel; re-run if it is interrupted
```

`setup.sh` mounts a 160 GB tmpfs in RAM for the datasets (both hosts have hundreds of GB). To use a persistent disk
instead, attach one of 200 GB or more to the VM and run `DATA_DISK=/dev/sdb ./setup.sh`; it is formatted when empty
and mounted at the same place, so `download.sh` only runs once in its lifetime.

## 3. Run

```bash
./run.sh              # or: API_KEY=my-key STREAMS=3 PORT=8000 ./run.sh
tmux attach -t glm    # the server's log; detach with Ctrl-b d
```

The kernel runs exactly as on Kaggle, minus the Cloudflare tunnel: weights onto the chips, the speculative-decoding
drafter from Hugging Face, the warm-up (under a minute when the exported and compiled programs match the chip, ~27 minutes when every program is traced and compiled),
then the `READY` banner with the endpoint, the API key and the model name. Reach it from your laptop through an SSH
port forward:

```bash
gcloud compute tpus tpu-vm ssh glm-tpu --zone=$Z -- -N -L 8000:localhost:8000
curl localhost:8000/v1/chat/completions -H "Authorization: Bearer <API_KEY>" -H "Content-Type: application/json" \
  -d '{"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The OpenAI API is at `/v1`, the Anthropic API at `/v1/messages`; the model README has the client settings for Claude
Code and other agents. `/health` and `/v1/models` also require the API key. The local server binds to loopback;
`run.sh` creates its configuration file with mode `0600`. It keeps running until you stop it (`tmux kill-session -t glm`)
or the VM goes away.

## Good to know

- **Exported and compiled programs.** The kernel writes every program it traces to `/kaggle/working/exported` and
  every executable it compiles to `/kaggle/working/jax_cache`, so a second start on the same VM is fast on either
  chip. The serve dataset's copies were made on a v5e; a v6e makes its own on the first run.
- **Memory is the same budget as on Kaggle only on the v5e.** The engine is tuned to 16 GB chips; on the v6e it
  simply has headroom.
- **Quota.** Request it before the VM: the serving quota for a single-host v5e is easy to miss, and a new project
  often starts at a limit below eight chips. The request is a few minutes in the Cloud Console.
- **What is different from Kaggle**: no tunnel (your SSH forward instead), no nine-hour limit, no idle shutdown (the
  `keepalive_min` in `serve_config.json` is set very high; the VM is yours to delete), and the drafter is fetched from
  Hugging Face on the VM, so it needs outbound Internet like Kaggle does.
