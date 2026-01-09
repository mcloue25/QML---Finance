import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";


const PORTFOLIO_FILE = path.resolve(
  process.cwd(),
  "..",
  "data",
  "json",
  "portfolio",
  "holdings_diversity.jsonl"
);

function safeMetric(s: string) {
  return s === "weight" || s === "euro";
}

async function exists(p: string) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

function toNum(x: any): number {
  const n = typeof x === "number" ? x : Number(x);
  return Number.isFinite(n) ? n : 0;
}

function buildTreemap(record: any) {
  const dist = record?.distribution || {};
  const weightsBySector = record?.holdings_weight || {};
  const eurosBySector = record?.holdings_euro || {};
  const sectorEuroTotals = record?.sector_exposure_euro || {};

  const sectors = Object.keys(dist);

  const children = sectors.map((sector) => {
    const tickers: string[] = Array.isArray(dist[sector]) ? dist[sector] : [];
    const ws: any[] = weightsBySector?.[sector] || [];
    const es: any[] = eurosBySector?.[sector] || [];

    const leafChildren = tickers
      .map((ticker, i) => {
        const w = toNum(ws?.[i]);
        const euro = toNum(es?.[i]);

        return {
          name: ticker,      // used as label by recharts
          ticker,
          sector,
          size: euro,        // ✅ treemap area uses euro by default
          weight: w,
          euro,
        };
      })
      .filter((x) => (x.size ?? 0) > 0);

    const sectorSize =
      sectorEuroTotals?.[sector] != null
        ? toNum(sectorEuroTotals[sector])
        : leafChildren.reduce((a, b) => a + (b.size || 0), 0);

    return {
      name: sector,
      sector,
      size: sectorSize,
      children: leafChildren,
    };
  });

  const cleaned = children.filter((s) => (s.size ?? 0) > 0 && (s.children?.length ?? 0) > 0);

  return { name: "Portfolio", children: cleaned };
}


export async function GET(req: Request) {
  const url = new URL(req.url);
  const metricRaw = (url.searchParams.get("metric") || "weight").toLowerCase();

  if (!safeMetric(metricRaw)) {
    return NextResponse.json({ ok: false, error: "Invalid metric. Use weight|euro." }, { status: 400 });
  }

  if (!(await exists(PORTFOLIO_FILE))) {
    return NextResponse.json(
      { ok: false, error: "Portfolio file not found", portfolioFile: PORTFOLIO_FILE },
      { status: 404 }
    );
  }

  try {
    const text = await fs.readFile(PORTFOLIO_FILE, "utf-8");
    const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);

    if (!lines.length) {
      return NextResponse.json(
        { ok: false, error: "Portfolio file is empty", portfolioFile: PORTFOLIO_FILE },
        { status: 404 }
      );
    }

    const latest = JSON.parse(lines[lines.length - 1]);

    const treemap = buildTreemap(latest);

    return NextResponse.json({
        ok: true,
        ts: latest.ts ?? null,
        portfolio_id: latest.portfolio_id ?? null,
        treemap,
        sector_exposure_weight: latest.sector_exposure_weight ?? null,
        sector_exposure_euro: latest.sector_exposure_euro ?? null,
    });
  } catch (e: any) {
    return NextResponse.json(
      {
        ok: false,
        error: "Failed reading portfolio jsonl",
        portfolioFile: PORTFOLIO_FILE,
        message: e?.message ?? String(e),
      },
      { status: 500 }
    );
  }
}
