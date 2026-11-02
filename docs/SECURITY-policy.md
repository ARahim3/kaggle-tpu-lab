# Security Policy

GitHub serves this file from `.github/SECURITY.md` (Security tab → policy). The
canonical copy lives in `docs/SECURITY-policy.md`; `.github/SECURITY.md` is a symlink
so both locations work.

## Supported versions

Only `main` is supported. Kaggle kernels and notebooks are stateless — always pull the
latest kernel script or regenerate the notebook before serving; older copies may predate
security fixes (see the advisory notes below for examples that have already happened).

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), or contact the maintainers directly if you cannot.
Include: the affected model folder / kernel, the setup a reproducer needs (accelerator,
datasets), and your assessment of impact. You will get an acknowledgment within a week
and a fix or a documented mitigation for anything confirmed.

## Threat model, in one page

These kernels deliberately run on **shared free infrastructure with public endpoints**.
The design assumes the endpoint URL is discoverable and treats the API key as the only
boundary:

- **The tunnel is public.** cloudflared hands out a guessable-pattern URL on the open
  internet. Every route (including diagnostics) requires the generated bearer key;
  key checks are constant-time. Treat the URL+key pair as a credential: do not paste
  them into issues, logs, or shared notebooks.
- **The key must never leave the session through a side channel.** Progress events
  (log, ntfy) are redacted recursively for credential-shaped fields; the ntfy topic is
  a progress relay only. This fixed real leaks before (the GLM and Qwen ready events
  once sent the key through ntfy — both patched).
- **Remote content is untrusted.** Image URLs supplied by clients are fetched only
  from public addresses — the kernel refuses private, loopback, link-local and
  reserved targets (SSRF guard; the kernel runs next to cloud metadata services).
- **The launcher holds state locally.** `launch.py` writes the endpoint and key to
  `~/.kaggle-tpu-lab.json` (mode 0600 on the user's machine) and never sends the key
  to Kaggle, ntfy, or the log beyond the local banner.
- **Modal endpoints are authenticated.** The Modal appliance runs vLLM with
  `--api-key`; deploying without the secret configured is a misdeployment, not a mode.
- **Third-party artifacts are quarantined by role.** Quantized checkpoints (AWQ/GPTQ)
  from community repos change model behavior, not execution — but treat their outputs
  as unvalidated; run your own spot-checks before trusting anything careful.
- **The burst gateway seam is opt-in and bearer-gated.** `burst_gateway*` config keys
  are empty by default; when set, the kernel announces only its own URL plus a
  credential *reference* (never the key itself) to the configured gateway.

## Known limitations (accepted risk)

- The endpoint has no rate limiting beyond the engine's bounded queue; a key-holder
  can exhaust the session's quota.
- Kaggle sessions run with internet egress and account credentials attached to the
  kernel; the kernels make no write API calls beyond their own tunnel and (optional)
  ntfy/gateway reports, which are visible in the source.
- API keys are generated per session and never persisted server-side; anyone with
  kernel-push access to your Kaggle account can read the key from the kernel source.
  Protect the Kaggle account accordingly (2FA).
