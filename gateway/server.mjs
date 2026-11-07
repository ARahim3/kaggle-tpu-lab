#!/usr/bin/env node
/**
 * kaggle-tpu-lab gateway — an OpenAI-compatible, on-demand front for the
 * Qwen3.8-27B Kaggle TPU recipe.
 *
 * Why it exists: a Kaggle TPU session is free but scarce (20 TPU-h/week per
 * account, one session at a time, ~9 h maximum). Running one 24/7 burns the
 * quota while nobody is using it. This gateway keeps the endpoint *cold* and
 * only pushes a Kaggle kernel when a request actually arrives; it deletes the
 * kernel again after `idleMinutes` of silence (and at the 9 h session cap), so
 * the quota is spent on work, not on uptime.
 *
 *   client ──> http://127.0.0.1:9790/v1  (stable local API key)
 *                 │
 *                 ├─ cold: pick account by remaining TPU quota, push kernel,
 *                 │        follow https://ntfy.sh/<topic> until READY
 *                 └─ hot:  stream bytes through to the cloudflared URL
 *
 * No dependencies beyond Node 18+ and the Kaggle CLI (see README).
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const VERSION = "1.0.0";

// ---------------------------------------------------------------- config ----
const DEFAULTS = {
  host: "127.0.0.1",
  port: 9790,
  pythonBin: process.platform === "win32" ? "python" : "python3",
  model: "qwen38-27b",              // recipe folder used by launch.py
  servedModel: "qwen3.8-27b",       // name the kernel actually serves
  slug: "ktl-qwen-gw",              // Kaggle kernel slug (one per account)
  idleMinutes: 10,                  // delete the kernel after this much silence
  maxSessionMinutes: 530,           // Kaggle hard-caps a session at ~9 h
  bootTimeoutMinutes: 45,           // queue + cold start ceiling
  bootWaitSeconds: 600,             // how long a request waits for a cold boot
  forceModel: true,                 // rewrite any client model name to servedModel
  textOnly: false,                  // true: skip the vision tower, ~8 min faster
  fastStart: true,                  // true: READY in ~4-6 min, warm shapes after
  maxModelLen: 262144,
  maxNumSeqs: 4,
  mtp: 3,
  reasoningEffort: "low",
  allowAnthropic: true,             // also proxy /v1/messages
  // planning figures used only for the "compute secured" report
  assumedTokPerSecSingle: 130,
  assumedTokPerSecAggregate: 540,
};

function readJson(p, fallback) {
  try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch { return fallback; }
}

const configPath = process.env.KTL_CONFIG || path.join(HERE, "config.local.json");
const cfg = { ...DEFAULTS, ...readJson(configPath, {}), ...envOverrides() };
function envOverrides() {
  const o = {};
  const map = {
    KTL_HOST: ["host", String], KTL_PORT: ["port", Number],
    KTL_IDLE_MIN: ["idleMinutes", Number], KTL_MAX_SESSION_MIN: ["maxSessionMinutes", Number],
    KTL_BOOT_TIMEOUT_MIN: ["bootTimeoutMinutes", Number], KTL_BOOT_WAIT_S: ["bootWaitSeconds", Number],
    KTL_PYTHON: ["pythonBin", String], KTL_TEXT_ONLY: ["textOnly", bool],
    KTL_FAST_START: ["fastStart", bool], KTL_MAX_MODEL_LEN: ["maxModelLen", Number],
    KTL_MAX_NUM_SEQS: ["maxNumSeqs", Number], KTL_MTP: ["mtp", Number],
    KTL_REASONING_EFFORT: ["reasoningEffort", String],
  };
  for (const [env, [key, cast]] of Object.entries(map)) {
    if (process.env[env] !== undefined && process.env[env] !== "") o[key] = cast(process.env[env]);
  }
  return o;
}
function bool(v) { return ["1", "true", "yes", "on"].includes(String(v).toLowerCase()); }

const ACCOUNTS_P = process.env.KTL_ACCOUNTS || path.join(HERE, "accounts.local.json");
const STATE_P = process.env.KTL_STATE || path.join(HERE, "state.local.json");
const WORK_P = path.join(HERE, ".work");

// ---------------------------------------------------------------- state -----
function loadState() {
  const s = readJson(STATE_P, null);
  if (s) return s;
  return {
    localApiKey: "sk-ktl-" + crypto.randomBytes(24).toString("hex"),
    backend: null,
    ledger: [],
    counters: { requests: 0, promptTokens: 0, completionTokens: 0, streamedRequests: 0 },
    startedAt: new Date().toISOString(),
  };
}
let state = loadState();
function saveState() {
  const tmp = STATE_P + ".tmp";
  fs.mkdirSync(path.dirname(STATE_P), { recursive: true });
  fs.writeFileSync(tmp, JSON.stringify(state, null, 2));
  fs.renameSync(tmp, STATE_P);
}
saveState();

const log = (...a) => console.log(new Date().toISOString(), "gateway", ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const secs = (o) => (o?.seconds || 0) + (o?.nanos || 0) / 1e9;
const hours = (s) => Math.round((s / 3600) * 100) / 100;

// ---------------------------------------------------------------- accounts --
function loadAccounts() {
  const j = readJson(ACCOUNTS_P, { accounts: [] });
  return (j.accounts || []).filter((a) => a && a.token && a.username && a.enabled !== false);
}

const accountState = new Map(); // name -> { lastBootAt, lastError, tpuEligible }

function accountsView() {
  return loadAccounts().map((a) => ({
    name: a.name, username: a.username,
    tpuEligible: a.tpuEligible !== false && accountState.get(a.name)?.tpuEligible !== false,
    lastBootAt: accountState.get(a.name)?.lastBootAt || null,
    lastError: accountState.get(a.name)?.lastError || null,
  }));
}

async function kaggleQuota(account) {
  const r = await fetch("https://www.kaggle.com/api/v1/kernels/quota", {
    headers: { Authorization: `Bearer ${account.token}` },
    signal: AbortSignal.timeout(30000),
  });
  if (!r.ok) throw new Error(`quota ${r.status}`);
  return r.json();
}

function run(cmd, args, { env = {}, cwd = ROOT, timeoutMs = 15 * 60 * 1000, input = null } = {}) {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { cwd, env: { ...process.env, ...env }, windowsHide: true });
    let out = "", err = "";
    const t = setTimeout(() => { try { p.kill(); } catch {} }, timeoutMs);
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    if (input) { p.stdin.write(input); p.stdin.end(); }
    p.on("close", (code) => { clearTimeout(t); resolve({ code, out, err, all: out + err }); });
    p.on("error", (e) => { clearTimeout(t); resolve({ code: -1, out, err: err + String(e), all: out + err + String(e) }); });
  });
}

function accountEnv(account) {
  const dir = path.join(HERE, ".kgcfg", account.name);
  fs.mkdirSync(dir, { recursive: true });
  return { KAGGLE_API_TOKEN: account.token, KAGGLE_CONFIG_DIR: dir };
}

async function kaggleCli(account, args, opts = {}) {
  return run(cfg.pythonBin, ["-m", "kaggle", ...args], { ...opts, env: accountEnv(account) });
}

async function kernelStatus(account, kernel) {
  const r = await kaggleCli(account, ["kernels", "status", kernel], { timeoutMs: 120000 });
  const m = (r.all || "").match(/KernelWorkerStatus\.(\w+)/);
  return m ? m[1] : "UNKNOWN";
}

// ------------------------------------------------------------------ boot ----
class BootError extends Error {
  constructor(message, { step = null, account = null, noTpu = false } = {}) {
    super(message);
    this.step = step; this.account = account; this.noTpu = noTpu;
  }
}

async function pushKernel(account) {
  const stateFile = path.join(WORK_P, `${account.name}-${cfg.slug}.json`);
  fs.mkdirSync(WORK_P, { recursive: true });
  try { fs.rmSync(stateFile, { force: true }); } catch {}

  const args = [
    path.join(ROOT, "launch.py"), "push",
    "--model", cfg.model,
    "--user", account.username,
    "--slug", cfg.slug,
    "--state", stateFile,
    "--keepalive-min", String(cfg.maxSessionMinutes),
    "--max-model-len", String(cfg.maxModelLen),
    "--max-num-seqs", String(cfg.maxNumSeqs),
    "--mtp", String(cfg.mtp),
    "--reasoning-effort", cfg.reasoningEffort,
  ];
  if (cfg.textOnly) args.push("--text-only");
  if (cfg.fastStart) args.push("--fast-start");

  log(`pushing ${account.username}/${cfg.slug} (${cfg.model})`);
  const r = await run(cfg.pythonBin, args, { env: accountEnv(account), timeoutMs: 20 * 60 * 1000 });
  if (r.code !== 0) throw new BootError(`push failed: ${(r.all || "").slice(-600)}`, { account: account.name });

  const m = (r.all || "").match(/KERNEL_STATE=(\{[^\n]*\})/);
  if (m) { try { return JSON.parse(m[1]); } catch {} }
  const st = readJson(stateFile, null);
  if (st) return st;
  throw new BootError(`push produced no state: ${(r.all || "").slice(-600)}`, { account: account.name });
}

async function ntfyEvents(topic, since) {
  try {
    const r = await fetch(
      `https://ntfy.sh/${encodeURIComponent(topic)}/json?poll=1&since=${since}`,
      { signal: AbortSignal.timeout(15000) },
    );
    if (!r.ok) return [];
    const text = await r.text();
    const out = [];
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      let e; try { e = JSON.parse(line); } catch { continue; }
      if (e.event !== "message") continue;
      try { out.push({ time: e.time, ...JSON.parse(e.message) }); } catch {}
    }
    return out;
  } catch { return []; }
}

async function waitForReady(account, st, onPhase) {
  let since = Math.floor(Date.now() / 1000) - 30;
  let lastCheck = 0;
  const deadline = Date.now() + cfg.bootTimeoutMinutes * 60000;
  const events = [];

  while (Date.now() < deadline) {
    for (const ev of await ntfyEvents(st.topic, since)) {
      since = Math.max(since, ev.time);
      events.push(ev);
      onPhase?.(ev);
      if (ev.phase === "ready") {
        if (!ev.endpoint) throw new BootError("kernel reported READY without a tunnel URL", { account: account.name });
        return { ...st, ...ev, events };
      }
      if (ev.phase === "failed") {
        const noTpu = ev.step === "no-tpu";
        throw new BootError(`kernel failed at ${ev.step || "?"}: ${ev.cause || ev.hint || "see kernel log"}`,
          { step: ev.step, account: account.name, noTpu });
      }
      if (ev.phase === "stopped" && (ev.cause || ev.hint)) {
        throw new BootError(`server exited: ${ev.cause || ev.hint}`, { account: account.name });
      }
    }

    if (Date.now() - lastCheck > 60000) {
      lastCheck = Date.now();
      const status = await kernelStatus(account, st.kernel);
      if (["ERROR", "CANCELACKNOWLEDGED", "COMPLETE"].includes(status)) {
        throw new BootError(`kernel finished with status ${status}`, { account: account.name });
      }
    }
    await sleep(5000);
  }
  throw new BootError(`boot timed out after ${cfg.bootTimeoutMinutes} min`, { account: account.name });
}

async function pickAccount(exclude = new Set()) {
  const accounts = loadAccounts().filter((a) =>
    !exclude.has(a.name) &&
    a.tpuEligible !== false &&
    accountState.get(a.name)?.tpuEligible !== false);
  if (!accounts.length) throw new BootError("no account marked TPU-eligible", { step: "no-account" });

  const rows = [];
  for (const a of accounts) {
    try {
      const q = await kaggleQuota(a);
      const tpuLeft = secs(q.tpuQuota?.totalTimeAllowed) - secs(q.tpuQuota?.timeUsed) - secs(q.tpuQuota?.timeReserved);
      rows.push({ a, tpuLeft, q });
    } catch (e) {
      accountState.set(a.name, { ...(accountState.get(a.name) || {}), lastError: `quota: ${e.message}` });
    }
  }
  const usable = rows.filter((r) => r.tpuLeft > 600).sort((x, y) => y.tpuLeft - x.tpuLeft);
  if (!usable.length) throw new BootError("every account is out of TPU quota this week", { step: "quota" });
  return usable[0];
}

let booting = null;

async function boot() {
  log("cold start requested");
  const tried = [];
  for (let attempt = 0; attempt < loadAccounts().length; attempt++) {
    const { a: account, tpuLeft } = await pickAccount(new Set(tried));
    tried.push(account.name);
    log(`using account ${account.name} (${hours(tpuLeft)} TPU-h left this week)`);

    let pushed = null;
    try {
      pushed = await pushKernel(account);
      state.backend = {
        status: "booting",
        account: account.name,
        username: account.username,
        kernel: pushed.kernel,
        topic: pushed.topic,
        upstreamKey: pushed.api_key,
        endpoint: null,
        model: pushed.model || cfg.servedModel,
        pushedAt: new Date().toISOString(),
        readyAt: null,
        lastUsedAt: null,
        phase: "queued",
      };
      saveState();

      const ready = await waitForReady(account, pushed, (ev) => {
        if (state.backend && ev.phase) { state.backend.phase = ev.phase; saveState(); }
      });
      state.backend.status = "live";
      state.backend.endpoint = ready.endpoint.replace(/\/+$/, "");
      state.backend.readyAt = new Date().toISOString();
      state.backend.lastUsedAt = new Date().toISOString();
      state.backend.phase = "ready";
      state.backend.startupSeconds = ready.startup_secs ?? null;
      saveState();
      accountState.set(account.name, { ...(accountState.get(account.name) || {}), lastBootAt: new Date().toISOString(), lastError: null });
      log(`live at ${state.backend.endpoint} (startup ${ready.startup_secs ?? "?"} s)`);
      return state.backend;
    } catch (e) {
      accountState.set(account.name, { ...(accountState.get(account.name) || {}), lastError: e.message, ...(e.noTpu ? { tpuEligible: false } : {}) });
      if (e.noTpu) log(`account ${account.name} has no TPU — marked ineligible`);
      try { await kaggleCli(account, ["kernels", "delete", pushed?.kernel || `${account.username}/${cfg.slug}`], { input: "yes\n", timeoutMs: 120000 }); } catch {}
      state.backend = null;
      saveState();
      if (attempt === loadAccounts().length - 1) throw e;
    }
  }
  throw new BootError("all accounts failed to boot", { step: "boot" });
}

async function adoptBoot() {
  const b = state.backend;
  const account = loadAccounts().find((a) => a.name === b.account);
  if (!account) throw new BootError(`unknown account ${b.account}`, { account: b.account });
  log(`adopting in-flight boot ${b.kernel} (pushed ${b.pushedAt})`);
  const ready = await waitForReady(account, { kernel: b.kernel, topic: b.topic }, (ev) => {
    if (state.backend && ev.phase) { state.backend.phase = ev.phase; saveState(); }
  });
  state.backend.status = "live";
  state.backend.endpoint = ready.endpoint.replace(/\/+$/, "");
  state.backend.readyAt = new Date().toISOString();
  state.backend.lastUsedAt = new Date().toISOString();
  state.backend.phase = "ready";
  state.backend.startupSeconds = ready.startup_secs ?? null;
  saveState();
  accountState.set(account.name, { ...(accountState.get(account.name) || {}), lastBootAt: new Date().toISOString(), lastError: null });
  log(`live at ${state.backend.endpoint} (startup ${ready.startup_secs ?? "?"} s)`);
  return state.backend;
}

async function ensureBackend() {
  if (state.backend?.status === "live") return state.backend;
  // a gateway restart must not orphan (and re-push) a kernel already in Kaggle's queue
  if (state.backend?.status === "booting" && state.backend.kernel && state.backend.topic
      && Date.now() - Date.parse(state.backend.pushedAt) < cfg.bootTimeoutMinutes * 60000) {
    if (!booting) booting = adoptBoot().finally(() => { booting = null; });
    return booting;
  }
  if (!booting) booting = boot().finally(() => { booting = null; });
  return booting;
}

async function backendHealthy(b) {
  if (!b) return false;
  try {
    const r = await fetch(`${b.endpoint}/models`, {
      headers: { Authorization: `Bearer ${b.upstreamKey}` },
      signal: AbortSignal.timeout(10000),
    });
    return r.ok;
  } catch { return false; }
}

async function stopBackend(reason) {
  const b = state.backend;
  if (!b) return { stopped: false };
  const account = loadAccounts().find((a) => a.name === b.account);
  const now = new Date().toISOString();
  if (account && b.kernel) {
    try { await kaggleCli(account, ["kernels", "delete", b.kernel], { input: "yes\n", timeoutMs: 180000 }); }
    catch (e) { log(`delete failed: ${e.message || e}`); }
  }
  if (b.pushedAt) {
    state.ledger.push({
      account: b.account, kernel: b.kernel, reason,
      pushedAt: b.pushedAt, readyAt: b.readyAt, endedAt: now,
      billableSeconds: Math.max(0, Math.round((Date.parse(now) - Date.parse(b.pushedAt)) / 1000)),
      servingSeconds: b.readyAt ? Math.max(0, Math.round((Date.parse(now) - Date.parse(b.readyAt)) / 1000)) : 0,
    });
  }
  state.backend = null;
  saveState();
  log(`stopped (${reason})`);
  return { stopped: true, reason };
}

// --------------------------------------------------------------- watchdog ---
setInterval(async () => {
  const b = state.backend;
  if (!b || b.status !== "live") return;
  const now = Date.now();
  if (b.lastUsedAt && now - Date.parse(b.lastUsedAt) > cfg.idleMinutes * 60000) {
    await stopBackend("idle");
    return;
  }
  if (b.pushedAt && now - Date.parse(b.pushedAt) > cfg.maxSessionMinutes * 60000) {
    await stopBackend("session-cap");
    return;
  }
  if (!(await backendHealthy(b))) {
    state.backend.healthFailures = (b.healthFailures || 0) + 1;
    if (state.backend.healthFailures >= 3) await stopBackend("unhealthy");
    else saveState();
  } else if (b.healthFailures) {
    state.backend.healthFailures = 0; saveState();
  }
}, 30000).unref();

// ----------------------------------------------------------------- proxy ----
function sendJson(res, code, obj, extraHeaders = {}) {
  const body = JSON.stringify(obj);
  res.writeHead(code, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
    ...extraHeaders,
  });
  res.end(body);
}

function openAiError(res, code, message, type = "api_error", extra = {}) {
  sendJson(res, code, { error: { message, type, code: type, ...extra } });
}

function authorized(req) {
  const h = req.headers.authorization || "";
  const bearer = h.startsWith("Bearer ") ? h.slice(7) : "";
  const key = req.headers["x-api-key"] || bearer;
  if (!key) return false;
  const a = Buffer.from(String(key));
  const b = Buffer.from(state.localApiKey);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let n = 0; const chunks = [];
    req.on("data", (c) => { n += c.length; if (n > 64 * 1024 * 1024) { reject(new Error("body too large")); req.destroy(); } else chunks.push(c); });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function addUsage(obj) {
  const u = obj?.usage;
  if (!u) return;
  state.counters.promptTokens += u.prompt_tokens || 0;
  state.counters.completionTokens += u.completion_tokens || 0;
}

async function forward(req, res, { suffix, body, stream, anthropic = false }) {
  let b;
  try {
    b = await ensureBackend();
  } catch (e) {
    if (stream && res.headersSent) {
      res.write(`data: ${JSON.stringify({ error: { message: e.message, type: "boot_failed" } })}\n\n`);
      res.end();
    } else {
      openAiError(res, 503, e.message, "model_booting", { step: e.step || null });
    }
    return;
  }

  const headers = {
    "content-type": "application/json",
    accept: stream ? "text/event-stream" : "application/json",
    authorization: `Bearer ${b.upstreamKey}`,
  };
  if (anthropic) { headers["x-api-key"] = b.upstreamKey; headers["anthropic-version"] = req.headers["anthropic-version"] || "2023-06-01"; }

  const controller = new AbortController();
  res.on("close", () => controller.abort());
  let upstream;
  try {
    upstream = await fetch(`${b.endpoint}${suffix}`, {
      method: "POST", headers, body, signal: controller.signal,
    });
  } catch (e) {
    if (!res.headersSent) openAiError(res, 502, `upstream unreachable: ${e.message}`, "upstream_error");
    else res.end();
    return;
  }

  state.backend.lastUsedAt = new Date().toISOString();
  saveState();

  if (!stream) {
    const buf = Buffer.from(await upstream.arrayBuffer());
    try { addUsage(JSON.parse(buf.toString())); saveState(); } catch {}
    res.writeHead(upstream.status, {
      "content-type": upstream.headers.get("content-type") || "application/json",
      "content-length": buf.length,
      "x-ktl-account": b.account || "",
      "x-ktl-backend": b.kernel || "",
    });
    res.end(buf);
    return;
  }

  if (!res.headersSent) {
    res.writeHead(upstream.status, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
      "x-ktl-account": b.account || "",
      "x-ktl-backend": b.kernel || "",
    });
  }
  let pending = "";
  try {
    for await (const chunk of upstream.body) {
      const buf = Buffer.from(chunk);
      res.write(buf);
      pending += buf.toString("utf8");
      if (pending.includes("data:")) {
        const lines = pending.split("\n");
        pending = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const d = line.slice(5).trim();
          if (!d || d === "[DONE]") continue;
          try { addUsage(JSON.parse(d)); } catch {}
        }
      }
    }
  } catch (e) {
    if (!controller.signal.aborted) res.write(`data: ${JSON.stringify({ error: { message: `stream error: ${e.message}`, type: "upstream_error" } })}\n\n`);
  }
  res.end();
  saveState();
}

// ------------------------------------------------------------------ routes --
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const p = url.pathname;

  // unauthenticated liveness only
  if (p === "/" || p === "/health") {
    return sendJson(res, 200, {
      name: "kaggle-tpu-lab gateway", version: VERSION,
      backend: state.backend ? { status: state.backend.status, account: state.backend.account, phase: state.backend.phase } : null,
      auth: "Bearer <local key>",
    });
  }

  if (!authorized(req)) return openAiError(res, 401, "invalid local API key", "invalid_api_key");

  try {
    if (req.method === "GET" && p === "/v1/models") {
      return sendJson(res, 200, {
        object: "list",
        data: [{ id: cfg.servedModel, object: "model", created: 0, owned_by: "kaggle-tpu-lab", context_length: cfg.maxModelLen }],
      });
    }

    if (req.method === "POST" && (p === "/v1/admin/warm" || p === "/v1/admin/boot")) {
      const b = await ensureBackend();
      return sendJson(res, 200, { status: "live", backend: b });
    }

    if (req.method === "POST" && p === "/v1/admin/stop") {
      return sendJson(res, 200, await stopBackend("manual"));
    }

    if (req.method === "GET" && p === "/v1/admin/status") {
      const accounts = [];
      for (const a of loadAccounts()) {
        const row = { ...accountsView().find((x) => x.name === a.name) };
        try {
          const q = await kaggleQuota(a);
          const tpuAllowed = secs(q.tpuQuota?.totalTimeAllowed);
          const tpuUsed = secs(q.tpuQuota?.timeUsed);
          const gpuAllowed = secs(q.gpuQuota?.totalTimeAllowed);
          const gpuUsed = secs(q.gpuQuota?.timeUsed);
          row.quota = {
            refreshAt: q.quotaRefreshTime,
            tpu: { allowedH: hours(tpuAllowed), usedH: hours(tpuUsed), remainingH: hours(tpuAllowed - tpuUsed) },
            gpu: { allowedH: hours(gpuAllowed), usedH: hours(gpuUsed), remainingH: hours(gpuAllowed - gpuUsed) },
          };
          row.tpuAvailable = tpuAllowed - tpuUsed > 600;
        } catch (e) { row.quotaError = e.message; }
        accounts.push(row);
      }
      const securedTpuH = accounts.reduce((s, a) => s + (a.quota?.tpu.allowedH || 0), 0);
      const securedGpuH = accounts.reduce((s, a) => s + (a.quota?.gpu.allowedH || 0), 0);
      const remainingTpuH = accounts.reduce((s, a) => s + (a.quota?.tpu.remainingH || 0), 0);
      const usableTpuH = accounts.reduce((s, a) => s + (a.tpuEligible && a.tpuAvailable ? (a.quota?.tpu.remainingH || 0) : 0), 0);
      const weekAgo = Date.now() - 7 * 24 * 3600e3;
      const delivered = state.ledger.filter((e) => Date.parse(e.endedAt || 0) > weekAgo);
      return sendJson(res, 200, {
        gateway: {
          version: VERSION,
          uptimeSeconds: Math.round(process.uptime()),
          counters: state.counters,
          idleMinutes: cfg.idleMinutes,
          maxSessionMinutes: cfg.maxSessionMinutes,
        },
        backend: state.backend,
        accounts,
        compute: {
          accounts: accounts.length,
          tpuEligibleAccounts: accounts.filter((a) => a.tpuEligible).length,
          // nominal: every account's weekly allowance, including accounts Kaggle
          // has not TPU-verified yet (their quota exists but cannot be claimed)
          securedGpuHoursPerWeek: Math.round(securedGpuH * 100) / 100,
          securedTpuHoursPerWeek: Math.round(securedTpuH * 100) / 100,
          // usable right now through TPU-verified accounts
          usableTpuHours: Math.round(usableTpuH * 100) / 100,
          lockedTpuHours: Math.round((remainingTpuH - usableTpuH) * 100) / 100,
          usableTpuChipHours: Math.round(usableTpuH * 8 * 100) / 100,
          tpuChipHoursPerWeek: Math.round(securedTpuH * 8 * 100) / 100,
          remainingTpuHours: Math.round(remainingTpuH * 100) / 100,
          // what the usable TPU hours can carry at the measured speeds
          estimatedTokensPerWeekSingleStream: Math.round(usableTpuH * 3600 * cfg.assumedTokPerSecSingle),
          estimatedTokensPerWeekAggregate: Math.round(usableTpuH * 3600 * cfg.assumedTokPerSecAggregate),
        },
        ledger: {
          sessionsLast7d: delivered.length,
          billableTpuSecondsLast7d: delivered.reduce((s, e) => s + (e.billableSeconds || 0), 0),
          servingSecondsLast7d: delivered.reduce((s, e) => s + (e.servingSeconds || 0), 0),
          recent: state.ledger.slice(-10),
        },
      });
    }

    if (req.method === "POST" && p === "/v1/chat/completions") {
      const raw = await readBody(req);
      let body; try { body = JSON.parse(raw.toString()); } catch { return openAiError(res, 400, "invalid JSON", "invalid_request_error"); }
      if (cfg.forceModel) body.model = cfg.servedModel;
      state.counters.requests++;
      const stream = body.stream === true;
      if (stream) {
        state.counters.streamedRequests++;
        // send keepalive comments while a cold boot is in progress
        if (!state.backend || state.backend.status !== "live") {
          res.writeHead(200, {
            "content-type": "text/event-stream; charset=utf-8",
            "cache-control": "no-cache", connection: "keep-alive", "x-accel-buffering": "no",
          });
          const timer = setInterval(() => res.write(": ktl booting\n\n"), 15000);
          try { await ensureBackend(); } catch (e) {
            clearInterval(timer);
            res.write(`data: ${JSON.stringify({ error: { message: e.message, type: "boot_failed" } })}\n\n`);
            return res.end();
          }
          clearInterval(timer);
        }
      }
      return forward(req, res, { suffix: "/chat/completions", body: Buffer.from(JSON.stringify(body)), stream });
    }

    if (req.method === "POST" && p === "/v1/completions") {
      const raw = await readBody(req);
      let body; try { body = JSON.parse(raw.toString()); } catch { return openAiError(res, 400, "invalid JSON", "invalid_request_error"); }
      if (cfg.forceModel) body.model = cfg.servedModel;
      state.counters.requests++;
      return forward(req, res, { suffix: "/completions", body: Buffer.from(JSON.stringify(body)), stream: body.stream === true });
    }

    if (req.method === "POST" && p === "/v1/messages" && cfg.allowAnthropic) {
      const raw = await readBody(req);
      let body; try { body = JSON.parse(raw.toString()); } catch { return openAiError(res, 400, "invalid JSON", "invalid_request_error"); }
      if (cfg.forceModel) body.model = cfg.servedModel;
      state.counters.requests++;
      return forward(req, res, { suffix: "/messages", body: Buffer.from(JSON.stringify(body)), stream: body.stream === true, anthropic: true });
    }

    openAiError(res, 404, `no route ${req.method} ${p}`, "not_found");
  } catch (e) {
    log("handler error", e);
    if (!res.headersSent) openAiError(res, 500, e.message);
    else res.end();
  }
});

server.listen(cfg.port, cfg.host, () => {
  log(`listening on http://${cfg.host}:${cfg.port}/v1`);
  log(`local API key: ${state.localApiKey}`);
  log(`accounts: ${loadAccounts().map((a) => a.username).join(", ") || "(none)"}`);
  if (!fs.existsSync(ACCOUNTS_P)) log(`WAIT: ${ACCOUNTS_P} not found (copy accounts.example.json)`);
});

server.on("error", (e) => {
  if (e.code === "EADDRINUSE") log(`port ${cfg.port} is already in use — set KTL_PORT or change config.local.json`);
  else log(`server error: ${e.message}`);
  process.exit(1);
});

function shutdown() { log("shutting down (backend is left running; POST /v1/admin/stop to kill it)"); server.close(); process.exit(0); }
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
