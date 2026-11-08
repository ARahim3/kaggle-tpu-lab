#!/usr/bin/env node
/**
 * Smoke test for the gateway. Never boots anything by itself unless --warm is
 * passed, so it is safe to run against a cold gateway.
 *
 *   node gateway/smoke.mjs [--base http://127.0.0.1:9792] [--warm]
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const arg = (n, d) => { const i = process.argv.indexOf(n); return i > -1 ? process.argv[i + 1] : d; };
const base = arg("--base", "http://127.0.0.1:9792").replace(/\/+$/, "");
const warm = process.argv.includes("--warm");
const key = JSON.parse(fs.readFileSync(path.join(HERE, "state.local.json"), "utf8")).localApiKey;
const H = { Authorization: `Bearer ${key}`, "Content-Type": "application/json" };

let fails = 0;
const ok = (name, cond, extra = "") => { console.log(`${cond ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`); if (!cond) fails++; };

// 1. models (must not boot)
const t0 = Date.now();
const models = await (await fetch(`${base}/v1/models`, { headers: H })).json();
ok("GET /v1/models", models?.data?.[0]?.id === "qwen3.8-27b", `(${Date.now() - t0} ms)`);

if (warm) {
  console.log("warming (this can take a while: Kaggle queue + vLLM start)...");
  const w = await fetch(`${base}/v1/admin/warm`, { method: "POST", headers: H });
  ok("POST /v1/admin/warm", w.ok, `${w.status}`);
  if (!w.ok) process.exit(1);
}

// 2. non-streaming chat
const t1 = Date.now();
const nr = await fetch(`${base}/v1/chat/completions`, {
  method: "POST", headers: H,
  body: JSON.stringify({ model: "qwen3.8-27b", max_tokens: 24, temperature: 0,
    messages: [{ role: "user", content: "Reply with exactly: gateway ok" }] }),
});
const nj = await nr.json();
ok("POST /v1/chat/completions (non-stream)", nr.ok && typeof nj?.choices?.[0]?.message?.content === "string",
  `(${Date.now() - t1} ms) ${JSON.stringify(nj?.choices?.[0]?.message?.content || nj?.error || "").slice(0, 120)}`);

// 3. streaming chat: verify SSE, token deltas, and time-to-first-token
const t2 = Date.now();
const sr = await fetch(`${base}/v1/chat/completions`, {
  method: "POST", headers: H,
  body: JSON.stringify({ model: "qwen3.8-27b", stream: true, max_tokens: 32, temperature: 0,
    messages: [{ role: "user", content: "Count to five." }] }),
});
ok("stream returns text/event-stream", (sr.headers.get("content-type") || "").includes("text/event-stream"));
let ttft = null, chunks = 0, text = "";
const dec = new TextDecoder();
for await (const c of sr.body) {
  const s = dec.decode(c, { stream: true });
  for (const line of s.split("\n")) {
    if (!line.startsWith("data:")) continue;
    const d = line.slice(5).trim();
    if (!d || d === "[DONE]") continue;
    try {
      const o = JSON.parse(d);
      const delta = o?.choices?.[0]?.delta?.content || "";
      if (delta && ttft === null) ttft = Date.now() - t2;
      text += delta; chunks++;
    } catch {}
  }
}
ok("stream produced deltas", chunks > 0, `ttft=${ttft ?? "?"} ms chunks=${chunks} text=${JSON.stringify(text.slice(0, 80))}`);

// 4. status + compute
const st = await (await fetch(`${base}/v1/admin/status`, { headers: H })).json();
ok("GET /v1/admin/status", !!st?.compute, JSON.stringify(st?.compute || {}).slice(0, 220));

console.log(fails ? `\n${fails} check(s) failed` : "\nall checks passed");
process.exit(fails ? 1 : 0);
