import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import Papa from "papaparse";

export const runtime = "nodejs";

const BASE_DIR = path.resolve(process.cwd(), "..", "data", "csv", "historical", "training", "raw");
const safeName = (s: string) => /^[A-Za-z0-9._-]+$/.test(s);

export async function GET(req: Request) {
  const asset = new URL(req.url).searchParams.get("asset");

  if (!asset) {
    return NextResponse.json({ ok: false, error: "Missing asset param" }, { status: 400 });
  }

  if (!safeName(asset)) {
    return NextResponse.json({ ok: false, error: "Invalid asset param" }, { status: 400 });
  }

  const file = path.join(BASE_DIR, `${asset}.csv`);

  let text: string;
  try {
    text = await fs.readFile(file, "utf8");
  } catch {
    return NextResponse.json(
      { ok: false, error: "History file not found", file },
      { status: 404 }
    );
  }

  const parsed = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  // Papa returns (unknown)[], but this is fine for your chart/table usage.
  return NextResponse.json({ ok: true, rows: parsed.data });
}
