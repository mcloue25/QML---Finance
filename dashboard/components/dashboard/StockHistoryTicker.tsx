"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { Card, CardContent } from "../ui/card";

// ---- Types & helpers ----
type HistoryRow = Record<string, any>;

function pick<T = any>(obj: any, keys: string[], fallback: T): T {
  for (const k of keys) {
    if (obj?.[k] != null) return obj[k];
  }
  return fallback;
}

function toNum(x: any): number | null {
  if (x == null) return null;
  if (typeof x === "number") return Number.isFinite(x) ? x : null;
  const n = Number(String(x));
  return Number.isFinite(n) ? n : null;
}

function toDate(x: any): Date | null {
  if (!x) return null;
  const d = x instanceof Date ? x : new Date(x);
  return Number.isNaN(d.getTime()) ? null : d;
}

function fmtMoney(n: number, digits = 4) {
  // crypto tends to need more decimals; equities less.
  const abs = Math.abs(n);
  const d = abs >= 100 ? 2 : abs >= 1 ? 4 : 6;
  return n.toLocaleString(undefined, {
    minimumFractionDigits: Math.min(digits, d),
    maximumFractionDigits: Math.max(digits, d),
  });
}

function fmtPct(n: number) {
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
}

function fmtCompact(n: number) {
  return n.toLocaleString(undefined, { notation: "compact", maximumFractionDigits: 2 });
}

function formatDateLabel(d: Date) {
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
}

type RangeKey = "1M" | "3M" | "6M" | "1Y" | "ALL";

function rangeCutoff(range: RangeKey): Date | null {
  const now = new Date();
  const d = new Date(now);
  if (range === "ALL") return null;
  if (range === "1M") d.setMonth(d.getMonth() - 1);
  if (range === "3M") d.setMonth(d.getMonth() - 3);
  if (range === "6M") d.setMonth(d.getMonth() - 6);
  if (range === "1Y") d.setFullYear(d.getFullYear() - 1);
  return d;
}

// ---- Component ----
export default function StockHistoryTicker(props: {
  asset: string;
  title?: string;
  className?: string;
  height?: number; // chart height
  refreshMs?: number; // optional auto refresh
}) {
  const { asset, title = "Price", className, height = 280, refreshMs } = props;

  const [rows, setRows] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState<RangeKey>("6M");

  async function load() {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`/api/history?asset=${encodeURIComponent(asset)}`, {
        cache: "no-store",
      });
      const json = await res.json();

      if (!res.ok || !json?.ok) {
        throw new Error(json?.error || `History fetch failed (${res.status})`);
      }

      const data: HistoryRow[] = Array.isArray(json?.rows)
        ? json.rows
        : Array.isArray(json?.history)
        ? json.history
        : Array.isArray(json?.data)
        ? json.data
        : Array.isArray(json)
        ? json
        : [];

      // normalize & sort
      const normalized = data
        .map((r) => {
          const dt =
            toDate(pick(r, ["date", "dt", "timestamp", "time"], null)) ??
            toDate(pick(r, ["Datetime", "Date"], null));

          const close = toNum(pick(r, ["close", "adj_close", "Close", "Adj Close"], null));
          const open = toNum(pick(r, ["open", "Open"], null));
          const high = toNum(pick(r, ["high", "High"], null));
          const low = toNum(pick(r, ["low", "Low"], null));
          const volume = toNum(pick(r, ["volume", "Volume"], null));

          if (!dt || close == null) return null;

          return {
            ...r,
            __dt: dt,
            __t: dt.getTime(),
            __close: close,
            __open: open,
            __high: high,
            __low: low,
            __vol: volume,
          };
        })
        .filter(Boolean) as any[];

      normalized.sort((a, b) => a.__t - b.__t);
      setRows(normalized);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asset]);

  useEffect(() => {
    if (!refreshMs) return;
    const id = setInterval(load, refreshMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asset, refreshMs]);

  const filtered = useMemo(() => {
    const cutoff = rangeCutoff(range);
    if (!cutoff) return rows;
    const t0 = cutoff.getTime();
    return rows.filter((r) => r.__t >= t0);
  }, [rows, range]);

  const latest = filtered.length ? filtered[filtered.length - 1] : null;
  const prev = filtered.length >= 2 ? filtered[filtered.length - 2] : null;

  const lastPx = latest?.__close ?? null;
  const prevPx = prev?.__close ?? null;
  const chg = lastPx != null && prevPx != null ? lastPx - prevPx : null;
  const chgPct = lastPx != null && prevPx != null && prevPx !== 0 ? (chg! / prevPx) * 100 : null;

  const badgeClass =
    chg == null
      ? "text-muted-foreground bg-muted"
      : chg >= 0
      ? "text-emerald-700 bg-emerald-50 dark:text-emerald-300 dark:bg-emerald-950/40"
      : "text-rose-700 bg-rose-50 dark:text-rose-300 dark:bg-rose-950/40";

  const chartData = useMemo(() => {
    return filtered.map((r) => ({
      t: r.__t,
      d: r.__dt,
      close: r.__close,
      open: r.__open,
      high: r.__high,
      low: r.__low,
      vol: r.__vol,
    }));
  }, [filtered]);

  return (
    <Card className={`rounded-2xl shadow-sm ${className ?? ""}`}>
      <CardContent className="pt-6">
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="text-sm font-semibold tracking-tight">{asset}</div>
              <div className="text-xs text-muted-foreground">{title}</div>
            </div>

            <div className="mt-1 flex flex-wrap items-center gap-2">
              <div className="text-2xl font-semibold tabular-nums tracking-tight">
                {lastPx == null ? "—" : fmtMoney(lastPx)}
              </div>

              <div className={`rounded-full px-2 py-0.5 text-xs font-medium tabular-nums ${badgeClass}`}>
                {chg == null || chgPct == null ? (
                  "—"
                ) : (
                  <>
                    {chg >= 0 ? "+" : ""}
                    {fmtMoney(chg, 6)} ({fmtPct(chgPct)})
                  </>
                )}
              </div>

              {latest?.__dt ? (
                <div className="text-xs text-muted-foreground">
                  {formatDateLabel(latest.__dt)}
                </div>
              ) : null}
            </div>

            {/* OHLC / Volume */}
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
              <span>
                O:{" "}
                <span className="text-foreground tabular-nums">
                  {latest?.__open == null ? "—" : fmtMoney(latest.__open)}
                </span>
              </span>
              <span>
                H:{" "}
                <span className="text-foreground tabular-nums">
                  {latest?.__high == null ? "—" : fmtMoney(latest.__high)}
                </span>
              </span>
              <span>
                L:{" "}
                <span className="text-foreground tabular-nums">
                  {latest?.__low == null ? "—" : fmtMoney(latest.__low)}
                </span>
              </span>
              <span>
                V:{" "}
                <span className="text-foreground tabular-nums">
                  {latest?.__vol == null ? "—" : fmtCompact(latest.__vol)}
                </span>
              </span>
            </div>
          </div>

          {/* Range selector */}
          <div className="flex items-center gap-1 rounded-xl border border-border bg-card p-1">
            {(["1M", "3M", "6M", "1Y", "ALL"] as RangeKey[]).map((k) => (
              <button
                key={k}
                onClick={() => setRange(k)}
                className={`rounded-lg px-2 py-1 text-xs font-medium transition ${
                  range === k
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
              >
                {k}
              </button>
            ))}
          </div>
        </div>

        {/* Body */}
        <div className="mt-4">
          {loading ? (
            <div className="text-sm text-muted-foreground">Loading history…</div>
          ) : error ? (
            <div className="text-sm text-rose-600 dark:text-rose-400">
              {error}
              <div className="mt-2">
                <button
                  onClick={load}
                  className="rounded-lg border border-border px-3 py-1 text-xs text-foreground hover:bg-muted"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : !chartData.length ? (
            <div className="text-sm text-muted-foreground">No history found.</div>
          ) : (
            <div className="grid gap-4 lg:grid-cols-[1fr_220px]">
              {/* Main chart */}
              <div className="min-w-0">
                <div className="h-[280px]" style={{ height }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ left: 6, right: 6, top: 10, bottom: 0 }}>
                      <CartesianGrid vertical={false} strokeDasharray="3 3" />
                      <XAxis
                        dataKey="t"
                        tickLine={false}
                        axisLine={false}
                        minTickGap={24}
                        tickFormatter={(t) => {
                          const d = new Date(t);
                          return d.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
                        }}
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        scale="time"
                      />
                      <YAxis
                        tickLine={false}
                        axisLine={false}
                        width={60}
                        tickFormatter={(v) => fmtMoney(Number(v))}
                        domain={["auto", "auto"]}
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const p: any = payload[0]?.payload;
                          const d = p?.d ? formatDateLabel(new Date(p.d)) : "";
                          return (
                            <div className="rounded-xl border border-border bg-card px-3 py-2 text-xs shadow-sm">
                              <div className="text-muted-foreground">{d}</div>
                              <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1">
                                <div>Close</div>
                                <div className="text-right tabular-nums">{fmtMoney(p.close)}</div>
                                {p.open != null ? (
                                  <>
                                    <div>Open</div>
                                    <div className="text-right tabular-nums">{fmtMoney(p.open)}</div>
                                  </>
                                ) : null}
                                {p.high != null ? (
                                  <>
                                    <div>High</div>
                                    <div className="text-right tabular-nums">{fmtMoney(p.high)}</div>
                                  </>
                                ) : null}
                                {p.low != null ? (
                                  <>
                                    <div>Low</div>
                                    <div className="text-right tabular-nums">{fmtMoney(p.low)}</div>
                                  </>
                                ) : null}
                                {p.vol != null ? (
                                  <>
                                    <div>Vol</div>
                                    <div className="text-right tabular-nums">{fmtCompact(p.vol)}</div>
                                  </>
                                ) : null}
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Area type="monotone" dataKey="close" strokeWidth={2} fillOpacity={0.2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Mini sparkline + quick stats */}
              <div className="rounded-2xl border border-border p-3">
                <div className="text-xs font-medium text-muted-foreground">Trend</div>
                <div className="mt-2 h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <Area type="monotone" dataKey="close" strokeWidth={2} fillOpacity={0.2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div className="mt-3 grid gap-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Rows</span>
                    <span className="tabular-nums">{chartData.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Start</span>
                    <span className="tabular-nums">
                      {chartData[0]?.d ? formatDateLabel(new Date(chartData[0].d)) : "—"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">End</span>
                    <span className="tabular-nums">
                      {chartData[chartData.length - 1]?.d
                        ? formatDateLabel(new Date(chartData[chartData.length - 1].d))
                        : "—"}
                    </span>
                  </div>
                </div>

                <button
                  onClick={load}
                  className="mt-3 w-full rounded-xl border border-border px-3 py-2 text-xs font-medium hover:bg-muted"
                >
                  Refresh
                </button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
