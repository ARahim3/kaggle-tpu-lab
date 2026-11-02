# Upstream issues and pull request audit

Reviewed 2026-10-30 against `ARahim3/kaggle-tpu-lab` `main`. The applicable fixes and
follow-up hardening are merged into this fork's `main`; source branches backing upstream
PRs remain intact.

## Fixes included in fork main

| Upstream item | Finding and disposition |
| --- | --- |
| Issue #20 | The supplied log shows the Kaggle weights on `FUSE.GCSFUSE`, automatic prefetch disabled, and the loader still at `0/18` shards after repeated startup attempts. The Qwen command now explicitly uses vLLM's `prefetch` safetensors strategy. Upstream PR [#21](https://github.com/ARahim3/kaggle-tpu-lab/pull/21). 2026-10-31 follow-up: a comment claimed uv re-downloads ~1 GB of wheels every run and that vllm-tpu 0.28.0 falls back to a slow CPU PyTorch path for `Qwen3_5ForConditionalGeneration`. Verified against sources: the cache miss costs ~1 min (log: `Prepared 238 packages in 1m 01s`, `PHASE installed {"secs": 81}`), fixed by `UV_CACHE_DIR=/tmp/uv-cache` (branch commit 006e55a); the arch warning is pre-existing 0.28.0 behavior (arch absent from `_VLLM_PREFERRED_ARCHITECTURES`, added on main 2026-10-21 in 7b82896c/#3837, in no release through v0.32.0, cf. unmerged #3533), and the fallback is the supported `VllmModelWrapper` torchax path on TPU, not CPU conversion. Corrected on the PR ([comment](https://github.com/ARahim3/kaggle-tpu-lab/pull/21#issuecomment-6550208212)); Kaggle run validation queued. |
| Issue #18 | The GLM ready event sent its bearer key through ntfy. GLM relay events omit it, diagnostic GET routes require the key, key checks are constant-time, and the GCP server binds to loopback. Follow-up review found the Qwen ready event had the same leak; both the kernel and notebook now recursively redact credential fields before progress logging or ntfy publication. Upstream PR [#22](https://github.com/ARahim3/kaggle-tpu-lab/pull/22), plus fork hardening. |
| Issue #17 | Queue/timeout errors after SSE headers previously wrote a second HTTP response into the stream. OpenAI and Anthropic streams now receive protocol-shaped error events; non-streaming requests retain HTTP 429/503. The CPU harness now exercises queue saturation for both protocols. Upstream PR [#23](https://github.com/ARahim3/kaggle-tpu-lab/pull/23). |
| Existing PR #9 | Its useful fixes were ported and reconciled with current `main`: Internet preflight, removal of broken single-host TPU metadata variables, clearer engine-exit diagnostics, and KYC guidance. The original PR conflicts with current `main` and remains open upstream. |
| Issue #3 | The Qwen README explains why the supported TPU recipe uses BF16 and why a third-party INT8 checkpoint is not substituted without its own accuracy/runtime validation. |

## Workarounds and remaining validation

| Upstream item | Status |
| --- | --- |
| Issues #10 and #14 | The attached #10 log confirms vLLM 0.28.0 crashes in `update_draft_token_ids_in_output` with `AttributeError: __delitem__` when MTP and async scheduling are combined. The launcher already supports `--no-async-scheduling` while retaining `--mtp 3`; this is documented and the exact command was added to the issue discussion. The requested Hermes throughput and live TPU behavior remain unverified. |
| Issue #20 backend split | The same log shows the old `tpu-inference` package falling back to vLLM's PyTorch model implementation for this architecture. This patch addresses the directly observed GCSFuse load stall. Migrating the full recipe to `vllm-torchtpu` is a separate runtime change and was not claimed or bundled without a Kaggle TPU validation run. |
| Issue #11 | Added `gcp/terraform/` for an opt-in eight-chip v5e-8/v6e-8 Spot TPU VM and documented credentials, cost, cleanup, and the direct-provisioning limit. It does not run `terraform apply` or upload Kaggle credentials. Use the existing queued-resource `gcloud` flow when capacity is scarce. |
| Issue #12 | The Android companion link remains a maintainer-curated decision. It handles endpoint credentials, so review its credential storage and release practices before advertising it to users. |
| Issue #15 | Added a cited feasibility assessment for DeepSeek V4.1 Flash and Qwen3.8-Flash-Next. It recommends a separate Qwen Q2_0/T4x2 POC first; no new model recipe is claimed until it passes runtime testing. |
| Issue #19 | The playlist guest-mode report does not map to a playlist component in this repository. |

## Open pull request review

- PR #9 contains relevant fixes for issues #1, #5, #6, #7, and #8. Its branch conflicts with current `main`; the applicable changes are reconciled in fork main. The INT8 question (#3) is addressed in the Qwen README.
- PR #4 changes the default model to an abliterated variant and adds a P100 recipe. `CONTRIBUTING.md` explicitly excludes default-model changes and other platform ports from this repository's merge scope, so it is not included.
- Upstream PRs #21, #22, and #23 remain open. Their focused fixes are merged into fork main through fork PRs #1, #2, and #4; fork PR #3 was closed as a conflict and replaced by #4. The original `fix/issue-*` branches remain present for the upstream PRs.

## Validation limits

Passed locally: Python syntax checks, shell syntax, both notebook JSON parses, exact notebook/kernel source parity, focused contracts for prefetch, MTP scheduling flags, TPU metadata sanitization, API-key handling, state-file permissions, Qwen progress-event credential redaction, and provider-shaped SSE overload errors. The isolated CPU JAX/NumPy/Transformers harness completed with `HARNESS OK`, including tool streams, disconnect handling, context reuse, API-key-protected diagnostics, and both OpenAI and Anthropic queue-saturation SSE behavior. The full changed-line Atheon scan reported no findings but skipped one binary test fixture; a later per-file heuristic scan found only low/medium pattern matches in code comments and contract-test lines, with no high/critical finding. Terraform 1.17.1 formatting, provider initialization, and validation passed with Google providers 8.9.0; no plan or apply was run.

The repository has no configured GitHub Actions checks. Kaggle authentication now succeeds using the protected `~/.kaggle/access_token`; quota showed 16.66 TPU hours remaining. Kaggle refused a second TPU kernel because the account already had one queued session (`burst-job-46`), which was left untouched. Therefore no issue-20 checkpoint-load run was started. The live GCSFuse prefetch behavior and Hermes MTP throughput still require a Kaggle runtime run after that queued job finishes.
