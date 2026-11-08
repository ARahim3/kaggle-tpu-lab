#!/usr/bin/env node
/**
 * What the fleet is worth, straight from Kaggle's quota endpoint.
 *
 *   node gateway/compute.mjs [--json]
 *
 * Reads accounts.local.json, asks Kaggle for each account's weekly allowance
 * and usage, and prints the nominal allowance (every account) next to the part
 * that is actually claimable today (accounts Kaggle has TPU-verified), plus a
 * rough GCP list-price equivalent and token capacity at the recipe's measured
 * throughput.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const accounts = JSON.parse(fs.readFileSync(process.env.KTL_ACCOUNTS || path.join(HERE, "accounts.local.json"), "utf8")).accounts;

const asJson = process.argv.includes("--json");
const secs = (o) => (o?.seconds || 0) + (o?.nanos || 0) / 1e9;
const h = (s) => Math.round((s / 3600) * 100) / 100;

// Rough GCP on-demand list prices, only for a sense of scale.
const PRICE = { tpuChipHour: 1.2, gpuHour: 0.35 }; // v5e chip-hour, T4
const TOK = { single: 130, aggregate: 540 };       // measured for this recipe

const rows = [];
for (const a of accounts) {
  try {
    const r = await fetch("https://www.kaggle.com/api/v1/kernels/quota", {
      headers: { Authorization: `Bearer ${a.token}` }, signal: AbortSignal.timeout(30000),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const q = await r.json();
    const tpuAllowed = secs(q.tpuQuota?.totalTimeAllowed), tpuUsed = secs(q.tpuQuota?.timeUsed);
    const gpuAllowed = secs(q.gpuQuota?.totalTimeAllowed), gpuUsed = secs(q.gpuQuota?.timeUsed);
    rows.push({
      account: a.name, username: a.username, tpuEligible: a.tpuEligible !== false,
      refreshAt: q.quotaRefreshTime,
      gpuAllowedH: h(gpuAllowed), gpuUsedH: h(gpuUsed), gpuLeftH: h(gpuAllowed - gpuUsed),
      tpuAllowedH: h(tpuAllowed), tpuUsedH: h(tpuUsed), tpuLeftH: h(tpuAllowed - tpuUsed),
    });
  } catch (e) {
    rows.push({ account: a.name, username: a.username, tpuEligible: a.tpuEligible !== false, error: e.message });
  }
}

const sum = (k, pred = () => true) => rows.filter(pred).reduce((s, r) => s + (r[k] || 0), 0);
const tpuNominal = sum("tpuAllowedH");
const tpuUsable = sum("tpuLeftH", (r) => r.tpuEligible);
const tpuLeft = sum("tpuLeftH");
const gpuNominal = sum("gpuAllowedH");

const report = {
  generatedAt: new Date().toISOString(),
  refreshAt: rows.find((r) => r.refreshAt)?.refreshAt || null,
  accounts: rows,
  totals: {
    accounts: rows.length,
    tpuEligibleAccounts: rows.filter((r) => r.tpuEligible).length,
    gpuHoursPerWeek: Math.round(gpuNominal * 100) / 100,
    tpuHoursPerWeekNominal: Math.round(tpuNominal * 100) / 100,
    tpuHoursPerWeekUsable: Math.round(tpuUsable * 100) / 100,
    tpuHoursLocked: Math.round((tpuLeft - tpuUsable) * 100) / 100,
    tpuChipHoursPerWeekNominal: Math.round(tpuNominal * 8 * 100) / 100,
    tpuChipHoursUsable: Math.round(tpuUsable * 8 * 100) / 100,
    gcpListEquivalentUsdPerWeek: {
      gpu: Math.round(gpuNominal * 2 * PRICE.gpuHour * 100) / 100,
      tpuNominal: Math.round(tpuNominal * 8 * PRICE.tpuChipHour * 100) / 100,
      tpuUsable: Math.round(tpuUsable * 8 * PRICE.tpuChipHour * 100) / 100,
    },
    tokenCapacityPerWeek: {
      "single-stream@130tok/s": Math.round(tpuUsable * 3600 * TOK.single),
      "8-stream@540tok/s": Math.round(tpuUsable * 3600 * TOK.aggregate),
    },
  },
};

if (asJson) { console.log(JSON.stringify(report, null, 2)); process.exit(0); }

console.table(rows.map((r) => ({
  account: r.account, username: r.username, "TPU ok": r.tpuEligible,
  "GPU h/wk": r.error ? r.error : `${r.gpuUsedH}/${r.gpuAllowedH}`,
  "TPU h/wk": r.error ? r.error : `${r.tpuUsedH}/${r.tpuAllowedH}`,
})));
const t = report.totals;
console.log(`
Weekly allowance (refreshes ${report.refreshAt})
  nominal      ${t.gpuHoursPerWeek} GPU-h + ${t.tpuHoursPerWeekNominal} TPU-h  (${t.tpuChipHoursPerWeekNominal} v5e chip-h)
  claimable    ${t.tpuHoursPerWeekUsable} TPU-h  (${t.tpuChipHoursUsable} chip-h) on ${t.tpuEligibleAccounts}/${t.accounts} TPU-verified accounts
  locked       ${t.tpuHoursLocked} TPU-h — accounts that have the quota but no TPU access yet

GCP on-demand list-price equivalent
  GPU part     $${t.gcpListEquivalentUsdPerWeek.gpu}/week
  TPU nominal  $${t.gcpListEquivalentUsdPerWeek.tpuNominal}/week
  TPU usable   $${t.gcpListEquivalentUsdPerWeek.tpuUsable}/week

Token capacity of the claimable TPU hours
  ${Object.entries(t.tokenCapacityPerWeek).map(([k, v]) => `${k}: ${(v / 1e6).toFixed(2)}M tokens/week`).join("\n  ")}
`);
