import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import parquet from "parquetjs-lite";

export const runtime = "nodejs";

const BACKTESTS_DIR = path.resolve(process.cwd(), "..", "data", "results", "backtests");
const safeName = (s: string) => /^[A-Za-z0-9._-]+$/.test(s);

async function exists(p: string) {
  try { await fs.access(p); return true; } catch { return false; }
}

async function listParquetFiles(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isFile())
    .map((e) => path.join(dir, e.name))
    .filter((p) => p.toLowerCase().endsWith(".parquet"))
    .sort((a, b) => a.localeCompare(b));
}

async function readParquetToJson(parquetPath: string, limit = 5000): Promise<any[]> {
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
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const asset = url.searchParams.get("asset") || "";
  const modelType = url.searchParams.get("model") || "xgb";
  const h = url.searchParams.get("h"); // optional

  if (!asset || !safeName(asset) || !safeName(modelType) || (h && !safeName(h))) {
    return NextResponse.json({ ok: false, error: "Invalid query params" }, { status: 400 });
  }

  const base = path.join(BACKTESTS_DIR, asset, modelType);
  const runDir = h ? path.join(base, String(h)) : base;

  if (!(await exists(runDir))) {
    return NextResponse.json({ ok: false, error: "Run folder not found", runDir }, { status: 404 });
  }

  const curvesDir = path.join(runDir, "curves");
  const tradesDir = path.join(runDir, "trades");

  if (!(await exists(curvesDir))) {
    const entries = await fs.readdir(runDir).catch(() => []);
    return NextResponse.json(
      { ok: false, error: "curves folder not found", runDir, entries },
      { status: 404 }
    );
  }

  // Find parquet files in each folder
  const curveParquets = await listParquetFiles(curvesDir);
  const tradeParquets = (await exists(tradesDir)) ? await listParquetFiles(tradesDir) : [];

  if (!curveParquets.length) {
    const entries = await fs.readdir(curvesDir).catch(() => []);
    return NextResponse.json(
      { ok: false, error: "No parquet found in curves/", curvesDir, entries },
      { status: 404 }
    );
  }

  // Choose which parquet to read
  // If multiple: prefer names containing 'bt' or 'curve'
  const pickPreferred = (paths: string[]) =>
    paths.find((p) => /bt|curve|curves/i.test(path.basename(p))) ?? paths[0];

  const curvesParquetPath = pickPreferred(curveParquets);
  const tradesParquetPath = tradeParquets.length ? pickPreferred(tradeParquets) : null;

  // Read parquet (cap rows so you don’t ship huge data to browser)
  const curves = await readParquetToJson(curvesParquetPath, 20000);
  const trades = tradesParquetPath ? await readParquetToJson(tradesParquetPath, 20000) : [];

  return NextResponse.json({
    ok: true,
    meta: { asset, model_type: modelType, horizon: h ?? null, runDir },
    metrics: null, // fill later if you have metrics.json
    bt: curves,    // keep naming compatible with your dashboard (bt == curves timeseries)
    trades,        // extra payload for later trade table/stats
    sources: {
      curvesParquet: path.basename(curvesParquetPath),
      tradesParquet: tradesParquetPath ? path.basename(tradesParquetPath) : null,
    },
  });
}
