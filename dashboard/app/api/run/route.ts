import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import parquet from "parquetjs-lite";

export const runtime = "nodejs";

const BACKTESTS_DIR = path.resolve(process.cwd(), "..", "data", "results", "backtests");
const safeName = (s: string) => /^[A-Za-z0-9._-]+$/.test(s);

async function exists(p: string) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function listParquetFiles(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
  return entries
    .filter((e) => e.isFile())
    .map((e) => path.join(dir, e.name))
    .filter((p) => p.toLowerCase().endsWith(".parquet"))
    .sort((a, b) => a.localeCompare(b));
}

function pickPreferred(paths: string[]) {
  return (
    paths.find((p) => /bt|curve|curves/i.test(path.basename(p))) ??
    paths.find((p) => /trade|trades/i.test(path.basename(p))) ??
    paths[0]
  );
}

function parseLimit(raw: string | null, fallback: number) {
  if (!raw) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(200_000, Math.floor(n)));
}

async function readParquetToJson(parquetPath: string, limit = 50000): Promise<any[]> {
  console.log("[parquet] attempting to open:", parquetPath);
  const stat = await fs.stat(parquetPath);
  console.log("[parquet] file size (bytes):", stat.size);

  try {
    const reader = await parquet.ParquetReader.openFile(parquetPath);
    try {
      const cursor = reader.getCursor();
      const rows: any[] = [];
      for (let i = 0; i < limit; i++) {
        const row = await cursor.next();
        if (!row) break;
        rows.push(row);
      }
      return rows;
    } finally {
      await reader.close();
    }
  } catch (e: any) {
    console.error("[parquet] open/read failed:", parquetPath, e);
    throw e;
  }
}

function toNum(x: any): number | null {
  if (x == null) return null;
  if (typeof x === "number") return Number.isFinite(x) ? x : null;
  const n = Number(String(x));
  return Number.isFinite(n) ? n : null;
}

function mean(arr: number[]) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdevSample(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const v = arr.reduce((acc, x) => acc + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(v);
}

/**
 * Compute a minimal-but-useful metrics object from curves.
 * Assumes daily returns; uses 252 scaling.
 */
function computeMetricsFromCurves(curves: any[]) {
  if (!curves?.length) return null;

  const rets = curves
    .map((r) => toNum(r.returns ?? r.net_ret ?? r.strategy_ret))
    .filter((v): v is number => v != null);

  if (!rets.length) return null;

  const n = rets.length;
  const mu = mean(rets);
  const sd = stdevSample(rets);

  const annual_return = mu * 252;
  const annual_volatility = sd * Math.sqrt(252);
  const sharpe =
    annual_volatility && annual_volatility > 0 ? annual_return / annual_volatility : null;

  // drawdown: prefer drawdown column if present
  const dd = curves
    .map((r) => toNum(r.drawdown ?? r.drawdown_net ?? r.drawdown_gross))
    .filter((v): v is number => v != null);
  const max_drawdown = dd.length ? dd.reduce((mn, v) => Math.min(mn, v), 0) : null;

  // time in market from position
  const eps = 1e-12;
  const pos = curves
    .map((r) => toNum(r.position ?? r.position_lag ?? r.target_position))
    .filter((v): v is number => v != null);
  const pct_time_in_market =
    pos.length ? pos.filter((p) => Math.abs(p) > eps).length / pos.length : null;

  // hit rate from returns (exclude zeros)
  const nonzero = rets.filter((r) => Math.abs(r) > eps);
  const hit_rate =
    nonzero.length ? nonzero.filter((r) => r > 0).length / nonzero.length : null;

  return {
    annual_return,
    annual_volatility,
    sharpe,
    // Your UI labels this "log"; your drawdown may or may not be log. You can rename later.
    max_drawdown_log: max_drawdown,
    avg_turnover_per_year: null, // not available from your current curve_df
    pct_time_in_market,
    hit_rate,
  };
}

/**
 * Normalize your curve_df schema (equity/returns/drawdown/position)
 * into the fields your dashboard currently expects.
 *
 * Your curve_df columns:
 *  - date (string)
 *  - equity (string/number)        -> cum_ret_net (for now)
 *  - returns (string/number)       -> net_ret
 *  - drawdown (string/number)      -> drawdown_net
 *  - position (string/number)      -> position_lag
 */
function normalizeCurvesForDashboard(curves: any[]) {
  return curves.map((r) => {
    const date = r.date ?? r.dt ?? r.timestamp ?? null;

    const equity = toNum(r.equity);
    const ret = toNum(r.returns);
    const dd = toNum(r.drawdown);
    const pos = toNum(r.position);

    return {
      ...r,
      date,

      // dashboard contract
      cum_ret_net: equity,      // model equity curve
      cum_ret_gross: equity,    // you don't have gross yet; mirror net for now
      drawdown_net: dd,
      drawdown_gross: dd,
      position_lag: pos,
      target_position: pos,     // you don't have target yet; mirror position
      turnover: null,           // you don't have it yet
      net_ret: ret,
      strategy_ret: ret,

      // probabilities/signals not available yet
      p_class_0: null,
      p_class_1: null,
      p_class_2: null,
      y_pred: null,
    };
  });
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);

    const asset = url.searchParams.get("asset") || "";
    const modelType = url.searchParams.get("model") || "xgb";
    const runId = url.searchParams.get("run") || "";

    if (!asset || !safeName(asset) || !safeName(modelType) || !runId || !safeName(runId)) {
      return NextResponse.json(
        { ok: false, error: "Invalid query params", asset, modelType, runId },
        { status: 400 }
      );
    }

    const limitCurves = parseLimit(url.searchParams.get("limitCurves"), 50000);
    const limitTrades = parseLimit(url.searchParams.get("limitTrades"), 50000);

    const runDir = path.join(BACKTESTS_DIR, asset, modelType, runId);

    if (!(await exists(runDir))) {
      return NextResponse.json(
        { ok: false, error: "Run folder not found", meta: { asset, model_type: modelType, run: runId, runDir } },
        { status: 404 }
      );
    }

    const warnings: string[] = [];

    // --- Load metrics from runs.parquet (optional) ---
    const runsParquetPath = path.join(runDir, "runs.parquet");
    let metrics: any = null;

    if (await exists(runsParquetPath)) {
      try {
        const rows = await readParquetToJson(runsParquetPath, 10000);
        metrics =
          rows.find((r) => String(r.run_id) === String(runId)) ??
          (rows.length === 1 ? rows[0] : null);
      } catch (e: any) {
        warnings.push(`runs.parquet read failed: ${e?.message ?? String(e)}`);
      }
    }

    // --- Load curves/trades parquet ---
    const curvesDir = path.join(runDir, "curves");
    const tradesDir = path.join(runDir, "trades");

    if (!(await exists(curvesDir))) {
      const entries = await fs.readdir(runDir).catch(() => []);
      return NextResponse.json(
        {
          ok: false,
          error: "curves folder not found",
          meta: { asset, model_type: modelType, run: runId, runDir },
          entries,
        },
        { status: 404 }
      );
    }

    const curveParquets = await listParquetFiles(curvesDir);
    const tradeParquets = (await exists(tradesDir)) ? await listParquetFiles(tradesDir) : [];

    if (!curveParquets.length) {
      const entries = await fs.readdir(curvesDir).catch(() => []);
      return NextResponse.json(
        {
          ok: false,
          error: "No parquet found in curves/",
          meta: { asset, model_type: modelType, run: runId, runDir },
          entries,
        },
        { status: 404 }
      );
    }

    const curvesParquetPath = pickPreferred(curveParquets);
    const tradesParquetPath = tradeParquets.length ? pickPreferred(tradeParquets) : null;

    // Curves
    let curvesRaw: any[] = [];
    if (limitCurves > 0) {
      try {
        curvesRaw = await readParquetToJson(curvesParquetPath, limitCurves);
      } catch (e: any) {
        return NextResponse.json(
          {
            ok: false,
            error: "Failed reading curves parquet",
            parquetPath: curvesParquetPath,
            message: e?.message ?? String(e),
          },
          { status: 500 }
        );
      }
    }

    // Normalize for dashboard contract
    const curves = normalizeCurvesForDashboard(curvesRaw);

    // If metrics missing, compute from curves
    if (!metrics) {
      const computed = computeMetricsFromCurves(curvesRaw);
      if (computed) metrics = computed;
      else warnings.push("metrics missing and could not be computed from curves");
    }

    // Trades (optional)
    let trades: any[] = [];
    if (tradesParquetPath && limitTrades > 0) {
      try {
        trades = await readParquetToJson(tradesParquetPath, limitTrades);
      } catch (e: any) {
        warnings.push(`trades read failed: ${e?.message ?? String(e)}`);
        trades = [];
      }
    }

    return NextResponse.json({
      ok: true,
      meta: {
        asset,
        model_type: modelType,
        run: runId,
        runDir,
        limits: { curves: limitCurves, trades: limitTrades },
      },
      metrics,
      bt: curves,   // ✅ now matches your dashboard’s expected keys
      trades,
      warnings,
      sources: {
        curvesParquet: path.basename(curvesParquetPath),
        tradesParquet: tradesParquetPath ? path.basename(tradesParquetPath) : null,
        runsParquet: (await exists(runsParquetPath)) ? "runs.parquet" : null,
      },
    });
  } catch (e: any) {
    console.error("[api/run] fatal:", e);
    return NextResponse.json(
      { ok: false, error: "api/run crashed", message: e?.message ?? String(e), stack: e?.stack ?? null },
      { status: 500 }
    );
  }
}
