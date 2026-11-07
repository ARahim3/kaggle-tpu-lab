# Gateway — an on-demand OpenAI-compatible front for the Qwen TPU recipe

A Kaggle TPU session is free but scarce: **~20 TPU-hours per account per week**,
one session at a time, and Kaggle stops a session at a little under nine hours.
Leaving one running 24/7 burns the quota while nobody is using it.

This gateway keeps the model **cold at rest**. It only pushes a Kaggle kernel
when a request actually arrives, waits for the tunnel, streams the request
through, and deletes the kernel again after `idleMinutes` of silence. The quota
is spent on tokens, not on uptime.

```
 client ──> http://127.0.0.1:9790/v1   (stable, local API key)
                │
                ├─ cold  → pick the account with the most TPU quota left,
                │          push the Qwen kernel, follow ntfy until READY
                └─ hot   → stream bytes through to the cloudflared URL
```

## What it gives you

* **One stable OpenAI-compatible endpoint and key** (`sk-ktl-…`), no matter
  which Kaggle account is serving or how many times the tunnel URL changes.
* **Streaming**, including SSE passthrough — the gateway does not buffer the
  response.
* **True lazy lifecycle**: cold start on first use, idle stop, 9-hour cap
  enforcement, health watch, automatic failover to the next account.
* **Per-account quota accounting** from Kaggle's own
  `/api/v1/kernels/quota`, so the "compute secured" report is authoritative,
  not an estimate from local timers.
* **Anthropic passthrough** (`/v1/messages`) so Claude Code can use the same
  endpoint (`ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`).

## Setup

```bash
# 1. Kaggle CLI (the gateway shells out to it)
pip install kaggle

# 2. tokens: kaggle.com → Settings → API → Create New Token
#    KGAT_… tokens are accepted directly; no username/password pair needed.
cp gateway/accounts.example.json gateway/accounts.local.json
$EDITOR gateway/accounts.local.json          # one entry per account

# 3. optional config
cp gateway/config.example.json gateway/config.local.json
$EDITOR gateway/config.local.json            # pythonBin, port, idleMinutes, …

# 4. run it
node gateway/server.mjs
```

The first start prints the local API key and writes it to
`gateway/state.local.json` (gitignored). Treat that file as a secret.

> **TPU access is per account.** Kaggle hands out a TPU only to accounts that
> are phone-verified (and, for some accounts, identity-checked through Persona).
> An unverified account still *pushes* the kernel, but the session comes up on a
> CPU box: the kernel sees `jax.device_count() == 1` and fails at the TPU step.
> The gateway catches that (`failed step=no-tpu`), marks the account
> `tpuEligible: false`, and moves on. Verify the phone on
> kaggle.com/settings to unlock that account's 20 h.

## Use it

```bash
KEY=$(node -e "console.log(require('./gateway/state.local.json').localApiKey)")

# models never wakes the backend
curl -s http://127.0.0.1:9790/v1/models -H "Authorization: Bearer $KEY"

# warm it explicitly (otherwise the first request does this and waits)
curl -s -X POST http://127.0.0.1:9790/v1/admin/warm -H "Authorization: Bearer $KEY"

# streaming chat
curl -N http://127.0.0.1:9790/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","stream":true,
       "messages":[{"role":"user","content":"Hello!"}]}'

# quota + compute report
curl -s http://127.0.0.1:9790/v1/admin/status -H "Authorization: Bearer $KEY" | jq

# stop now (also happens on idle)
curl -s -X POST http://127.0.0.1:9790/v1/admin/stop -H "Authorization: Bearer $KEY"
```

Any OpenAI client:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:9790/v1
export OPENAI_API_KEY=$KEY
# model: qwen3.8-27b
```

### Cold-start behaviour

A request that arrives while the backend is cold triggers the boot. A streaming
request gets an immediate `200 text/event-stream` and `: ktl booting` keepalive
comments while Kaggle queues the TPU and vLLM loads the weights; the real tokens
follow in the same stream. Non-streaming requests wait for the boot and then get
the normal JSON response. If the boot exceeds `bootWaitSeconds` (or the kernel
fails), you get a `503` with `type: model_booting` / the failure reason.

Boot time is dominated by Kaggle's TPU queue and vLLM start-up, not the gateway:
with `fastStart: true` the README's measured figure is ~6 min after the kernel
starts, plus however long the v5e-8 queue is (it is busy at peak times). Use
`POST /v1/admin/warm` a few minutes before a session if the first-token latency
matters.

## Quota and "compute secured"

Kaggle's quota is wall-clock **accelerator hours per rolling week**, reset weekly
(the API returns `quotaRefreshTime`). The gateway reads it per account:

```
GET https://www.kaggle.com/api/v1/kernels/quota      (Bearer KGAT_…)
  gpuQuota.totalTimeAllowed = 108000 s  = 30 h
  tpuQuota.totalTimeAllowed =  72000 s  = 20 h
```

`/v1/admin/status` sums those into `compute.securedGpuHoursPerWeek`,
`compute.securedTpuHoursPerWeek` and `tpuChipHoursPerWeek` (a v5e-8 is eight
chips), and projects token capacity with the recipe's measured throughput
(~130 tok/s single stream, ~540 tok/s at eight streams). The `ledger` section
records every gateway-driven session with its billable and serving seconds.

The same report without the server:

```bash
node gateway/compute.mjs          # table + totals
node gateway/compute.mjs --json   # machine-readable
```

It separates the **nominal** allowance (every account, 30 GPU-h + 20 TPU-h each)
from what is **claimable today** (TPU-verified accounts only) and the hours
**locked** behind missing TPU verification.

The gateway only counts a session as billable from **push** (the moment Kaggle
starts provisioning) to **delete**, because that is what Kaggle charges — the
cold start included, not just the time the endpoint was live.

## Files

| File | Committed | Purpose |
|---|---|---|
| `server.mjs` | yes | the gateway |
| `compute.mjs` | yes | quota / "compute secured" report |
| `smoke.mjs` | yes | end-to-end check (models, stream, non-stream, status) |
| `accounts.example.json` | yes | token registry template |
| `config.example.json` | yes | defaults |
| `accounts.local.json` | **no** | your KGAT tokens |
| `config.local.json` | **no** | your overrides |
| `state.local.json` | **no** | local key, live backend, session ledger |
| `.kgcfg/` | **no** | per-account Kaggle CLI config dirs |
| `.work/` | **no** | per-push launcher state files |

## Limits worth knowing

* One live session per account; the gateway runs one backend at a time and
  fails over only when an account cannot boot.
* A gateway restart does not orphan a queued kernel: if a boot is in flight it
  is adopted (same kernel and ntfy topic) instead of pushed again.
* Kaggle can refuse a TPU at peak times (`QUEUED` for a long while) or hand out
  a session without one. Both are Kaggle-side and retried on the next account.
* The tunnel URL and the kernel's own API key are regenerated on every push;
  the local key is not.
* Kaggle's terms apply. This is a personal-use front for the free tier, not a
  way to resell compute.
