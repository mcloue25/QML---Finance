import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

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

async function listDirs(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
  return entries
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b));
}

/**
 * A folder is considered a valid "run" iff it contains a `curves/` directory.
 * This automatically excludes non-run folders/files like `runs.parquet` and
 * anything like `graphs/` (unless it incorrectly contains `curves/`).
 */
async function isValidRunDir(runDir: string) {
  return exists(path.join(runDir, "curves"));
}

export async function GET(req: Request) {
  const url = new URL(req.url);

  // Optional filters:
  // /api/catalog?asset=AAPL
  // /api/catalog?asset=AAPL&model=xgb
  const assetFilter = url.searchParams.get("asset");
  const modelFilter = url.searchParams.get("model");

  if (assetFilter && !safeName(assetFilter)) {
    return NextResponse.json({ ok: false, error: "Invalid asset param" }, { status: 400 });
  }
  if (modelFilter && !safeName(modelFilter)) {
    return NextResponse.json({ ok: false, error: "Invalid model param" }, { status: 400 });
  }

  if (!(await exists(BACKTESTS_DIR))) {
    return NextResponse.json(
      { ok: false, error: "Backtests dir not found", baseDir: BACKTESTS_DIR },
      { status: 404 }
    );
  }

  const assetsAll = await listDirs(BACKTESTS_DIR);
  const assets = assetFilter ? assetsAll.filter((a) => a === assetFilter) : assetsAll;

  const runs: Record<string, Record<string, string[]>> = {};
  const modelTypesSet = new Set<string>();

  for (const asset of assets) {
    if (!safeName(asset)) continue;

    const assetDir = path.join(BACKTESTS_DIR, asset);
    const modelsAll = await listDirs(assetDir);
    const models = modelFilter ? modelsAll.filter((m) => m === modelFilter) : modelsAll;

    const perAsset: Record<string, string[]> = {};

    for (const model of models) {
      if (!safeName(model)) continue;

      const modelDir = path.join(assetDir, model);

      // New layout: <asset>/<model>/<UUID>/(curves|trades|graphs|runs.parquet...)
      // We only include UUID folders that contain curves/
      const subdirs = await listDirs(modelDir);

      const runIds: string[] = [];
      for (const sub of subdirs) {
        if (!safeName(sub)) continue;

        const runDir = path.join(modelDir, sub);
        if (await isValidRunDir(runDir)) {
          runIds.push(sub);
        }
      }

      if (runIds.length) {
        modelTypesSet.add(model);
        perAsset[model] = runIds.sort((a, b) => a.localeCompare(b));
      }
    }

    if (Object.keys(perAsset).length) {
      runs[asset] = perAsset;
    }
  }

  const assetsWithBacktests = Object.keys(runs).sort((a, b) => a.localeCompare(b));
  const model_types = Array.from(modelTypesSet).sort((a, b) => a.localeCompare(b));

  return NextResponse.json({
    ok: true,
    baseDir: BACKTESTS_DIR,
    assets: assetsWithBacktests,
    model_types,
    runs, // runs[asset][model] = ["<UUID>", "<UUID>", ...]
  });
}
