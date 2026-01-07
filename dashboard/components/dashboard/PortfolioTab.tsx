"use client";

import React, { useMemo } from "react";
import { Card, CardContent } from "../ui/card";
import { ResponsiveContainer, Treemap, Tooltip } from "recharts";

type Holding = {
  ticker: string;
  name: string;
  value: number;
  sector: "Layer 1" | "Stablecoins" | "Altcoins";
};

const DUMMY: Holding[] = [
  { ticker: "BTC-USD", name: "Bitcoin", value: 52000, sector: "Layer 1" },
  { ticker: "ETH-USD", name: "Ethereum", value: 21000, sector: "Layer 1" },
  { ticker: "ADA-USD", name: "Cardano", value: 4000, sector: "Altcoins" },
  { ticker: "USDT-USD", name: "Tether", value: 15000, sector: "Stablecoins" },
];

// Demo-proof colors (don’t rely on CSS vars existing)
const SECTOR_COLORS: Record<Holding["sector"], string> = {
  "Layer 1": "#2563eb", // blue
  Altcoins: "#7c3aed", // violet
  Stablecoins: "#16a34a", // green
};

function AllocationTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  return (
    <div className="rounded-xl border border-border bg-background/95 px-3 py-2 text-xs shadow-sm backdrop-blur">
      <div className="font-medium">{d.ticker ?? d.name}</div>
      {d.ticker ? <div className="text-muted-foreground">{d.name}</div> : null}
      {typeof d.size === "number" ? (
        <div className="mt-1">
          <span className="text-muted-foreground">Value: </span>€{Number(d.size).toLocaleString()}
        </div>
      ) : null}
      {typeof d.weight === "number" ? (
        <div>
          <span className="text-muted-foreground">Weight: </span>
          {(Number(d.weight) * 100).toFixed(1)}%
        </div>
      ) : null}
      {d.sector ? (
        <div>
          <span className="text-muted-foreground">Sector: </span>
          {d.sector}
        </div>
      ) : null}
    </div>
  );
}

function TreemapNode(props: any) {
  const { x, y, width, height, depth, name, ticker, sector } = props;

  if (width <= 0 || height <= 0) return null;

  // Show labels for smaller tiles too (but keep it readable)
  const showHoldingLabel = depth === 2 && width > 58 && height > 28;
  const showSectorLabel = depth === 1 && width > 90 && height > 34;

  const base = sector ? SECTOR_COLORS[sector as Holding["sector"]] : "#0f172a";

  // Fit text a bit better inside small rectangles
  const holdingFontSize = Math.max(10, Math.min(14, Math.floor(Math.min(width, height) / 4.2)));

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={10}
        ry={10}
        fill={base}
        fillOpacity={depth === 1 ? 0.22 : 0.9}
        stroke={depth === 1 ? "rgba(0,0,0,0.18)" : "rgba(255,255,255,0.75)"}
        strokeWidth={depth === 1 ? 1 : 2}
      />

      {/* Sector label (parent) */}
      {showSectorLabel ? (
        <text x={x + 10} y={y + 20} fontSize={13} fontWeight={700} fill="currentColor">
          {name}
        </text>
      ) : null}

      {/* Holding label (leaf): show the TICKER inside each box */}
      {showHoldingLabel ? (
        <text
          x={x + 10}
          y={y + 20}
          fontSize={holdingFontSize}
          fontWeight={800}
          fill="white"
          style={{ pointerEvents: "none", userSelect: "none" }}
        >
          {ticker}
        </text>
      ) : null}
    </g>
  );
}

export default function PortfolioTab() {
  const total = useMemo(() => DUMMY.reduce((a, b) => a + b.value, 0), []);

  const rows = useMemo(() => {
    return DUMMY.map((h) => ({ ...h, w: total > 0 ? h.value / total : 0 })).sort(
      (a, b) => b.value - a.value
    );
  }, [total]);

  const treemapData = useMemo(() => {
    const bySector: Record<string, any[]> = {};

    rows.forEach((r) => {
      if (!bySector[r.sector]) bySector[r.sector] = [];
      bySector[r.sector].push({
        name: r.name,
        ticker: r.ticker,
        size: r.value,
        weight: r.w,
        sector: r.sector, // ensures holdings use sector color
      });
    });

    return [
      {
        name: "Portfolio",
        children: Object.entries(bySector).map(([sector, children]) => ({
          name: sector,
          sector,
          children,
        })),
      },
    ];
  }, [rows]);

  const legend = useMemo(() => Array.from(new Set(rows.map((r) => r.sector))), [rows]);

  return (
    <div className="grid gap-4 md:grid-cols-12">
      <Card className="md:col-span-7 rounded-2xl shadow-sm">
        <CardContent className="pt-6">
          <div className="text-sm font-medium">Portfolio allocation (dummy)</div>
          <div className="mt-1 text-xs text-muted-foreground">
            This tab uses placeholder holdings, ToDo replace this with holdings from file once I have executor setup.
          </div>

          {/* Treemap */}
          <div className="mt-4 rounded-2xl border border-border bg-muted/30 p-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs font-medium text-muted-foreground">Treemap</div>
              <div className="text-[11px] text-muted-foreground">Hover for details</div>
            </div>

            {/* Legend */}
            <div className="mb-3 flex flex-wrap gap-2">
              {legend.map((s) => (
                <div key={s} className="flex items-center gap-2 rounded-full border border-border px-2 py-1">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: SECTOR_COLORS[s] }} />
                  <span className="text-[11px] text-muted-foreground">{s}</span>
                </div>
              ))}
            </div>

            <div className="h-[240px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <Treemap data={treemapData} dataKey="size" stroke="transparent" content={<TreemapNode />}>
                  <Tooltip content={<AllocationTooltip />} />
                </Treemap>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Existing bars */}
          <div className="mt-4 space-y-3">
            {rows.map((r) => (
              <div key={r.ticker} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <div className="font-medium">{r.ticker}</div>
                  <div className="text-muted-foreground">
                    €{r.value.toLocaleString()} • {(r.w * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div
                    className="h-2 rounded-full bg-foreground/70"
                    style={{ width: `${Math.max(1, r.w * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="md:col-span-5 rounded-2xl shadow-sm">
        <CardContent className="pt-6">
          <div className="text-sm font-medium">Holdings table</div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted-foreground">
                <tr>
                  <th className="py-2">Ticker</th>
                  <th className="py-2">Name</th>
                  <th className="py-2 text-right">Value</th>
                  <th className="py-2 text-right">Weight</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.ticker} className="border-t border-border">
                    <td className="py-2 font-medium">{r.ticker}</td>
                    <td className="py-2 text-muted-foreground">{r.name}</td>
                    <td className="py-2 text-right">€{r.value.toLocaleString()}</td>
                    <td className="py-2 text-right">{(r.w * 100).toFixed(1)}%</td>
                  </tr>
                ))}
                <tr className="border-t border-border">
                  <td className="py-2 font-medium" colSpan={2}>
                    Total
                  </td>
                  <td className="py-2 text-right font-medium">€{total.toLocaleString()}</td>
                  <td className="py-2 text-right text-muted-foreground">100%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
