"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "../ui/card";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Treemap,
  Cell,
  Tooltip as RechartsTooltip,
  Legend,
} from "recharts";

const DEBUG_TREEMAP = false; // set true if you want logs

const SECTOR_COLORS = [
  "#2563EB",
  "#16A34A",
  "#DC2626",
  "#9333EA",
  "#EA580C",
  "#0891B2",
  "#CA8A04",
  "#4F46E5",
  "#0D9488",
  "#BE185D",
  "#65A30D",
  "#7C2D12",
  "#1D4ED8",
  "#15803D",
  "#B91C1C",
  "#6D28D9",
  "#C2410C",
  "#155E75",
  "#854D0E",
  "#312E81",
];

type PortfolioApi = {
  ok: boolean;
  ts: string | null;
  portfolio_id: string | null;
  treemap: { name: string; children: any[] };
  sector_exposure_weight: Record<string, number> | null;
  sector_exposure_euro: Record<string, number> | null;
  error?: string;
  message?: string;
};

function normSector(s: any) {
  return String(s ?? "").trim().toLowerCase();
}

function fmtPct(x: any) {
  const n = Number(x);
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : "—";
}

function fmtMoney(x: any) {
  const n = Number(x);
  return Number.isFinite(n) ? `€${n.toLocaleString()}` : "—";
}

function TreemapTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;

  const node = payload?.[0]?.payload || {};
  const isLeaf = !node.children;

  return (
    <div className="rounded-xl border border-border bg-background/95 px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-foreground">
        {isLeaf ? node.ticker || node.name : node.name}
      </div>

      {node.sector ? (
        <div className="mt-1 text-muted-foreground">Sector: {node.sector}</div>
      ) : null}

      {isLeaf ? (
        <div className="mt-2 space-y-1">
          <div className="flex items-center justify-between gap-6">
            <span className="text-muted-foreground">Value</span>
            <span className="font-medium">{fmtMoney(node.euro ?? node.size)}</span>
          </div>
          <div className="flex items-center justify-between gap-6">
            <span className="text-muted-foreground">Weight</span>
            <span className="font-medium">{fmtPct(node.weight)}</span>
          </div>
        </div>
      ) : (
        <div className="mt-2 space-y-1">
          <div className="flex items-center justify-between gap-6">
            <span className="text-muted-foreground">Sector total</span>
            <span className="font-medium">{fmtMoney(node.size)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Custom Treemap cell:
 * - Uses props.fill (we precompute it in data) => reliable coloring
 * - Shows labels only on leaf nodes (depth > 1)
 * - Never shows sector labels
 */
function TreemapLeafLabelsOnly(props: any) {
  const {
    x,
    y,
    width,
    height,
    depth,
    name,
    fill,
    onMouseEnter,
    onMouseLeave,
    onClick,
    payload, // may be empty depending on recharts build; don't rely on it
  } = props || {};

  const X = Number(x);
  const Y = Number(y);
  const W = Number(width);
  const H = Number(height);

  if (
    !Number.isFinite(X) ||
    !Number.isFinite(Y) ||
    !Number.isFinite(W) ||
    !Number.isFinite(H) ||
    W <= 0 ||
    H <= 0
  ) {
    return <g />;
  }

  const rectFill = typeof fill === "string" && fill ? fill : "#94a3b8";

  // Leaf labels only
  const showLeafLabel = depth > 1 && W > 50 && H > 18;

  // Prefer ticker if present, else name
  const label = String(payload?.ticker ?? name ?? "");

  if (DEBUG_TREEMAP && depth === 2 && Math.random() < 0.01) {
    // eslint-disable-next-line no-console
    console.log("[TreemapCell]", { depth, name, fill: rectFill, payload });
  }

  return (
    <g>
      <rect
        x={X}
        y={Y}
        width={W}
        height={H}
        fill={rectFill}
        stroke="#fff"
        strokeWidth={1}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        onClick={onClick}
      />
      {showLeafLabel ? (
        <text
          x={X + 6}
          y={Y + 16}
          fontSize={11}
          fill="rgba(255,255,255,0.95)"
          style={{ pointerEvents: "none" }}
        >
          {label}
        </text>
      ) : null}
    </g>
  );
}

export default function PortfolioTab() {
  const [data, setData] = useState<PortfolioApi | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      setStatus("loading");
      setErr(null);
      try {
        const res = await fetch("/api/portfolio");
        const json = (await res.json()) as PortfolioApi;
        if (!res.ok || !json.ok) throw new Error(json?.error || json?.message || "Failed");
        setData(json);
        setStatus("ok");
      } catch (e: any) {
        setData(null);
        setStatus("error");
        setErr(e?.message ?? String(e));
      }
    })();
  }, []);

  const sectorRows = useMemo(() => {
    const w = data?.sector_exposure_weight || {};
    const e = data?.sector_exposure_euro || {};
    const sectors = Array.from(new Set([...Object.keys(w), ...Object.keys(e)]));
    return sectors
      .map((sector) => ({
        sector,
        weight: w[sector] ?? null,
        euro: e[sector] ?? null,
      }))
      .sort((a, b) => (b.euro ?? 0) - (a.euro ?? 0));
  }, [data]);

  const sectorOrder = useMemo(() => {
    const children = data?.treemap?.children ?? [];
    return children.map((n: any) => String(n?.name ?? "").trim()).filter(Boolean);
  }, [data]);

  const sectorColorMap = useMemo(() => {
    const combined = [
      ...sectorOrder,
      ...sectorRows.map((r) => String(r.sector ?? "").trim()),
    ].filter(Boolean);

    const uniqueNorm: string[] = [];
    for (const s of combined) {
      const ns = normSector(s);
      if (ns && !uniqueNorm.includes(ns)) uniqueNorm.push(ns);
    }

    const map = new Map<string, string>();
    uniqueNorm.forEach((sectorNorm, idx) => {
      map.set(sectorNorm, SECTOR_COLORS[idx % SECTOR_COLORS.length]);
    });

    return map;
  }, [sectorOrder, sectorRows]);

  const colorForSector = (sector: string) =>
    sectorColorMap.get(normSector(sector)) ?? "#94a3b8";

  const pieData = useMemo(
    () =>
      sectorRows
        .filter((r) => (r.weight ?? 0) > 0)
        .map((r) => ({ name: r.sector, weight: r.weight ?? 0 })),
    [sectorRows]
  );

  /**
   * ✅ Key fix: precompute `fill` on the treemap data itself.
   * Recharts passes `fill` to content reliably, even when `payload` is empty.
   */
  const treemapWithFill = useMemo(() => {
    const sectors = data?.treemap?.children ?? [];
    return sectors.map((sectorNode: any) => {
      const sectorName = String(sectorNode?.name ?? "").trim();
      const sectorFill = colorForSector(sectorName);

      const children = Array.isArray(sectorNode?.children) ? sectorNode.children : [];
      const leafChildren = children.map((leaf: any) => ({
        ...leaf,
        // leaf may already have leaf.sector from API - use it, fallback to parent sector
        fill: colorForSector(String(leaf?.sector ?? sectorName)),
      }));

      return {
        ...sectorNode,
        fill: sectorFill, // not displayed as label, but used as the parent rect fill
        children: leafChildren,
      };
    });
  }, [data, sectorColorMap]);

  return (
    <div className="grid gap-4 md:grid-cols-12">
      {/* TREEMAP */}
      <Card className="md:col-span-12 rounded-2xl shadow-sm">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium">Portfolio treemap</div>
              <div className="mt-1 text-xs text-muted-foreground">
                Area = Euro exposure • Hover a cell to see euro + weight
                {data?.ts ? ` • ${data.ts}` : ""}
              </div>
            </div>

            <div className="text-xs text-muted-foreground">
              {status === "loading" ? "Loading…" : status === "error" ? "Error" : null}
            </div>
          </div>

          <div className="mt-4">
            {status === "loading" ? (
              <div className="rounded-2xl border border-border bg-card/60 p-6 text-sm text-muted-foreground">
                Loading portfolio…
              </div>
            ) : status === "error" ? (
              <div className="rounded-2xl border border-red-200/60 bg-red-50/50 p-6 text-sm text-red-900 dark:border-red-900/30 dark:bg-red-950/20 dark:text-red-200">
                Failed to load portfolio: {err}
              </div>
            ) : !data?.treemap?.children?.length ? (
              <div className="rounded-2xl border border-border bg-card/60 p-6 text-sm text-muted-foreground">
                No treemap data found.
              </div>
            ) : (
              <div className="h-[420px]">
                <ResponsiveContainer width="100%" height="100%">
                  <Treemap
                    data={treemapWithFill}
                    dataKey="size"
                    nameKey="name"
                    stroke="rgba(0,0,0,0.08)"
                    isAnimationActive={false}
                    // ✅ makes recharts propagate each node's `fill` into props
                    fill="#94a3b8"
                    content={<TreemapLeafLabelsOnly />}
                  >
                    <RechartsTooltip content={<TreemapTooltip />} />
                  </Treemap>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* LEFT: TABLE */}
      <Card className="md:col-span-6 rounded-2xl shadow-sm">
        <CardContent className="pt-6">
          <div className="text-sm font-medium">Sector exposure</div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted-foreground">
                <tr>
                  <th className="py-2">Sector</th>
                  <th className="py-2 text-right">Euro</th>
                  <th className="py-2 text-right">Weight</th>
                </tr>
              </thead>
              <tbody>
                {sectorRows.map((r) => (
                  <tr key={r.sector} className="border-t border-border">
                    <td className="py-2 font-medium">
                      <span className="inline-flex items-center">
                        <span
                          className="mr-2 inline-block h-2 w-2 rounded-full"
                          style={{ backgroundColor: colorForSector(r.sector) }}
                        />
                        {r.sector}
                      </span>
                    </td>
                    <td className="py-2 text-right">{fmtMoney(r.euro)}</td>
                    <td className="py-2 text-right">{fmtPct(r.weight)}</td>
                  </tr>
                ))}
                {!sectorRows.length ? (
                  <tr className="border-t border-border">
                    <td className="py-3 text-sm text-muted-foreground" colSpan={3}>
                      No sector exposure values.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* RIGHT: PIE */}
      <Card className="md:col-span-6 rounded-2xl shadow-sm">
        <CardContent className="pt-6">
          <div className="text-sm font-medium">Sector weights</div>
          <div className="mt-1 text-xs text-muted-foreground">Pie shows weight allocation by sector</div>

          {!pieData.length ? (
            <div className="mt-4 rounded-2xl border border-border bg-card/60 p-6 text-sm text-muted-foreground">
              No data to plot.
            </div>
          ) : (
            <div className="mt-4 h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <RechartsTooltip
                    formatter={(value: any, name: any, props: any) => [fmtPct(value), props?.payload?.name ?? name]}
                  />
                  <Legend />
                  <Pie
                    data={pieData}
                    dataKey="weight"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius="95%"
                    innerRadius="40%"
                    isAnimationActive={false}
                  >
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={colorForSector(entry.name)} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
