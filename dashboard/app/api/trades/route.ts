import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import parquet from "@dsnp/parquetjs";

export const runtime = "nodejs";

const BACKTESTS_DIR = path.resolve(process.cwd(), "..", "data", "results", "backtests");
const safeName = (s: string) => /^[A-Za-z0-9._-]+$/.test(s);

/**
 * JSON-safe conversion:
 * - BigInt -> string
 * - Long (from "long" lib) -> string
 */
function jsonSafe<T>(value: T): T {
  return JSON.parse(
    JSON.stringify(value, (_k, v) => {
      if (typeof v === "bigint") return v.toString();
      if (v && typeof v === "object" && v.constructor?.name === "Long" && typeof v.toString === "function") {
        return v.toString();
      }
      return v;
    })
  );
}

export async function GET(req: Request) {
  const url = new URL(req.url);

  const asset = url.searchParams.get("asset");
  const model = url.searchParams.get("model");
  const runId = url.searchParams.get("run");

  const limitParam = url.searchParams.get("limit");
  const limitRaw = limitParam ? Number(limitParam) : 10_000;
  const limit = Number.isFinite(limitRaw) ? Math.min(Math.max(1, limitRaw), 50_000) : 10_000;

  if (![asset, model, runId].every(Boolean)) {
    return NextResponse.json({ ok: false, error: "Missing params (asset, model, run)" }, { status: 400 });
  }
  if (![asset, model, runId].every((v) => safeName(v!))) {
    return NextResponse.json({ ok: false, error: "Invalid params" }, { status: 400 });
  }

  const tradesPath = path.join(
    BACKTESTS_DIR,
    asset!,
    model!,
    runId!,
    "trades",
    `${runId}.parquet`
  );

  try {
    await fs.access(tradesPath);
  } catch {
    return NextResponse.json({ ok: false, error: "Trades parquet not found", tradesPath }, { status: 404 });
  }

  let reader: any = null;

  try {
    reader = await parquet.ParquetReader.openFile(tradesPath);
    const cursor = reader.getCursor();

    const rows: any[] = [];
    for (let i = 0; i < limit; i++) {
      const record = await cursor.next();
      if (!record) break;
      rows.push(jsonSafe(record));
    }

    await reader.close();
    return NextResponse.json({ ok: true, trades: rows, count: rows.length });
  } catch (err: any) {
    try {
      if (reader) await reader.close();
    } catch {
      // ignore
    }

    return NextResponse.json(
      {
        ok: false,
        error: "Failed to read trades parquet",
        detail: err?.message ?? String(err),
        tradesPath,
      },
      { status: 500 }
    );
  }
}
