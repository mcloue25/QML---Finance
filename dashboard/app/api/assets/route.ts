import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";

// What assets exist in my training CSV folder?

const CSV_DIR = path.resolve(process.cwd(),"..", "data", "csv", "historical", "training", "cleaned");
const PARQUET_DIR = path.resolve(process.cwd(),"..", "data", "results", "backtests");

export async function GET() {
  try {
    const entries = await fs.readdir(CSV_DIR, { withFileTypes: true });

    const csv_files = entries
      .filter((e) => e.isFile())
      .map((e) => e.name)
      .filter((name) => name.toLowerCase().endsWith(".csv"))
      .sort((a, b) => a.localeCompare(b));

    const assets = csv_files.map((f) => f.replace(/\.csv$/i, ""));

    return NextResponse.json({
      ok: true,
      baseDir: CSV_DIR,
      count: assets.length,
      assets,
      csv_files,
    });
  } catch (err: any) {
    return NextResponse.json(
      { ok: false, baseDir: CSV_DIR, error: err?.message ?? String(err) },
      { status: 500 }
    );
  }
}
