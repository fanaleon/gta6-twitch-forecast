// GTA VI launch-day Twitch viewers: hour-by-hour quantile forecast.
//
// Method: weighted analogue pooling in log space.
//   For every past launch in the case's `comparables`, take its release-day curve as log ratios
//   y_h = ln(viewers in hour h / viewers in hour 1), h = 2..24 (all days start 00:00 UTC, so the
//   regional unlock waves line up in clock time). Each analogue is weighted by how much it looks
//   like this case at the end of hour 1:
//     - momentum f = ln(peak of the 24 h before the day / hour-1 print): near 0 or below when the
//       game is already unlocked and live at hour 1, large when hour 1 is still pre-launch;
//     - scale s = ln(hour-1 print): big categories move differently from small ones;
//     - early-access launches (whose release day is not the real first day) count less.
//   The forecast for hour h is the weighted quantile of the analogues' y_h, smoothed with a
//   Gaussian kernel whose width grows with the horizon, then scaled by hour 1's print.
//
// The weights' bandwidths were picked by leave-one-out over the rolling-midnight launches of the
// practice comparables (loss ~0.26 vs ~0.37 for persistence).
//
// Node 22, standard library only, deterministic (no randomness).

import fs from "node:fs";
import path from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR || "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR || "/output";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const HOUR_MS = 3600 * 1000;

// Tuned by leave-one-out (see header).
const BW_MOMENTUM = 1.0; // Gaussian bandwidth on f
const BW_SCALE = 1.0; // Gaussian bandwidth on ln(v1)
const EARLY_ACCESS_WEIGHT = 0.3;
const KERNEL_FAR = 0.2; // smoothing width at hour 24 (log units), scales with sqrt(h-1)
const KERNEL_MIN = 0.02;
const WEIGHT_FLOOR = 1e-4; // keeps every analogue in play a little

type Analogue = { name: string; f: number; s: number; ea: boolean; y: (number | null)[] };

function normCdf(x: number): number {
  // Abramowitz-Stegun 26.2.17, |error| < 7.5e-8
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x > 0 ? 1 - p : p;
}

// Quantile p of a weighted Gaussian-kernel mixture centred on vals, found by bisection.
function kernelQuantile(vals: number[], ws: number[], p: number, h: number): number {
  let W = 0;
  for (const w of ws) W += w;
  const cdf = (x: number) => {
    let s = 0;
    for (let i = 0; i < vals.length; i++) s += ws[i] * normCdf((x - vals[i]) / h);
    return s / W;
  };
  let lo = Math.min(...vals) - 8 * h;
  let hi = Math.max(...vals) + 8 * h;
  for (let k = 0; k < 80; k++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function momentum(prevVals: number[], v1: number): number {
  let peak = 1;
  for (const v of prevVals) if (v > peak) peak = v;
  return Math.log(peak / Math.max(v1, 1));
}

function analogues(c: any): Analogue[] {
  const out: Analogue[] = [];
  const gameName: string = (c.game && c.game.name) || "";
  for (const m of c.comparables || []) {
    // The target game's own pre-release history is not a launch-day curve.
    if (!m || !Array.isArray(m.hourly)) continue;
    if (/target-history/i.test(m.unlock || "") || (gameName && String(m.name).startsWith(gameName))) continue;
    const t0 = Date.parse(m.day_start);
    const by = new Map<number, number>();
    for (const r of m.hourly) {
      if (r && typeof r.viewers === "number") by.set(Math.round((Date.parse(r.from) - t0) / HOUR_MS), r.viewers);
    }
    const v1 = by.get(0);
    if (!v1 || v1 <= 0) continue;
    const prev: number[] = [];
    for (let k = -24; k < 0; k++) {
      const v = by.get(k);
      if (v !== undefined) prev.push(v);
    }
    const y: (number | null)[] = [];
    for (let h = 2; h <= 24; h++) {
      const v = by.get(h - 1);
      y.push(v !== undefined && v > 0 ? Math.log(v / v1) : null);
    }
    out.push({
      name: String(m.name),
      f: momentum(prev, v1),
      s: Math.log(v1),
      ea: /early-access/i.test(m.unlock || ""),
      y,
    });
  }
  return out;
}

function hourOnePrint(c: any): number {
  const k = (c.known || []).find((r: any) => r && r.hour === 1);
  const v = k && typeof k.viewers === "number" ? k.viewers : 1;
  return Math.max(v, 1);
}

function caseMomentum(c: any, v1: number): number {
  const start = Date.parse(c.day_start);
  const prev: number[] = [];
  for (const r of c.history_hourly || []) {
    const t = Date.parse(r.from);
    if (t >= start - 24 * HOUR_MS && t < start && typeof r.viewers === "number") prev.push(r.viewers);
  }
  return momentum(prev, v1);
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

function forecast(c: any): { hours: { hour: number; q: number[] }[] } {
  const v1 = hourOnePrint(c);
  const f = caseMomentum(c, v1);
  const s = Math.log(v1);
  const pool = analogues(c);
  const rows: { hour: number; q: number[] }[] = [];
  for (let h = 2; h <= 24; h++) {
    const i = h - 2;
    const vals: number[] = [];
    const ws: number[] = [];
    for (const a of pool) {
      const y = a.y[i];
      if (y === null) continue;
      const w =
        Math.exp(-0.5 * ((a.f - f) / BW_MOMENTUM) ** 2) *
          Math.exp(-0.5 * ((a.s - s) / BW_SCALE) ** 2) *
          (a.ea ? EARLY_ACCESS_WEIGHT : 1) +
        WEIGHT_FLOOR;
      vals.push(y);
      ws.push(w);
    }
    let q: number[];
    if (vals.length === 0) {
      // No analogue for this hour: fall back to persistence with a random-walk spread.
      const z = [-1.645, -0.674, 0, 0.674, 1.645];
      q = z.map((zz) => v1 * Math.exp(zz * 0.15 * Math.sqrt(h - 1)));
    } else {
      const width = KERNEL_FAR * Math.sqrt((h - 1) / 23) + KERNEL_MIN;
      q = LEVELS.map((p) => v1 * Math.exp(kernelQuantile(vals, ws, p, width)));
    }
    // Valid shape: finite, in (0, 1e9], non-decreasing.
    let last = 0.01;
    q = q.map((x) => {
      let v = Number.isFinite(x) ? x : v1;
      v = Math.min(Math.max(round2(v), 0.01), 1e9);
      if (v < last) v = last;
      last = v;
      return v;
    });
    rows.push({ hour: h, q });
  }
  return { hours: rows };
}

function main(): void {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = fs.readdirSync(INPUT_DIR).filter((n) => n.endsWith(".in")).sort();
  for (const name of files) {
    const id = name.slice(0, -3);
    let out;
    let c: any = null;
    try {
      c = JSON.parse(fs.readFileSync(path.join(INPUT_DIR, name), "utf8"));
      out = forecast(c);
    } catch {
      // Never leave a case without a valid file: the house's persistence band on hour 1's print.
      let v1 = 1;
      try {
        v1 = hourOnePrint(c);
      } catch {}
      const z = [-1.645, -0.674, 0, 0.674, 1.645];
      out = { hours: [] as { hour: number; q: number[] }[] };
      for (let h = 2; h <= 24; h++) {
        out.hours.push({ hour: h, q: z.map((zz) => round2(Math.max(v1, 1) * Math.exp(zz * 0.15 * Math.sqrt(h - 1)))) });
      }
    }
    fs.writeFileSync(path.join(OUTPUT_DIR, id + ".out"), JSON.stringify(out));
  }
}

main();
