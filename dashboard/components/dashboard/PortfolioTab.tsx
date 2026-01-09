"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "../ui/card";
import { ResponsiveContainer, Treemap, Tooltip } from "recharts";

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

  // Recharts treemap tooltip payload can be a bit odd; this is usually where the leaf lives:
  const node = payload?.[0]?.payload || {};

  // Only show detailed tooltip on ticker leaves (not sector parent)
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

  return (
    <div className="grid gap-4 md:grid-cols-12">
      <Card className="md:col-span-12 rounded-2xl shadow-sm">
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
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
                    data={data.treemap.children}
                    dataKey="size"
                    nameKey="name"
                    stroke="rgba(0,0,0,0.08)"
                    isAnimationActive={false}
                  >
                    <Tooltip content={<TreemapTooltip />} />
                  </Treemap>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="md:col-span-12 rounded-2xl shadow-sm">
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
                    <td className="py-2 font-medium">{r.sector}</td>
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
    </div>
  );
}
