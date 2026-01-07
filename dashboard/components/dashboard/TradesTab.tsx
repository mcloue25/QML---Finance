"use client";

import React, { useMemo } from "react";
import { Card, CardContent } from "../ui/card";

type Trade = Record<string, any>;

const MAX_ROWS = 200;      
const MAX_HEIGHT_PX = 520; 

function formatDateTime(value: any) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

function formatNumber(value: any, opts?: Intl.NumberFormatOptions) {
  if (value == null || value === "") return "—";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return String(value);
  return new Intl.NumberFormat(undefined, opts).format(n);
}

function formatCell(key: string, value: any) {
  if (value == null) return "—";

  if (key === "entry_time" || key === "exit_time") return formatDateTime(value);

  if (key === "pnl_pct") return `${formatNumber(value, { maximumFractionDigits: 2 })}%`;
  if (key === "return") return formatNumber(value, { maximumFractionDigits: 4 });

  if (key === "entry_price" || key === "exit_price")
    return formatNumber(value, { minimumFractionDigits: 4, maximumFractionDigits: 6 });

  if (key === "bars_held") return formatNumber(value, { maximumFractionDigits: 0 });

  if (key === "avg_exposure" || key === "max_exposure")
    return formatNumber(value, { maximumFractionDigits: 3 });

  return String(value);
}

function labelFor(key: string) {
  const map: Record<string, string> = {
    symbol: "Symbol",
    entry_time: "Entry",
    exit_time: "Exit",
    entry_price: "Entry Px",
    exit_price: "Exit Px",
    pnl_pct: "PnL %",
    return: "Return",
    bars_held: "Bars",
    avg_exposure: "Avg Exp",
    max_exposure: "Max Exp",
  };
  return map[key] ?? key;
}

export default function TradesTab(props: { trades: Trade[] }) {
  const { trades } = props;

  const columns = useMemo(() => {
    const t0 = trades?.[0];
    if (!t0) return [];

    const preferred = [
      "symbol",
      "entry_time",
      "exit_time",
      "entry_price",
      "exit_price",
      "pnl_pct",
      "return",
      "bars_held",
      "avg_exposure",
      "max_exposure",
    ];

    const hidden = new Set(["run_id"]);
    const keys = Object.keys(t0).filter((k) => !hidden.has(k));

    const ordered: string[] = [];
    for (const k of preferred) if (keys.includes(k)) ordered.push(k);

    const remaining = keys
      .filter((k) => !ordered.includes(k))
      .sort((a, b) => a.localeCompare(b));

    return [...ordered, ...remaining];
  }, [trades]);

  const shownTrades = useMemo(() => trades.slice(0, MAX_ROWS), [trades]);

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {trades.length
              ? `${shownTrades.length} / ${trades.length} rows`
              : "No trades"}
          </div>
        </div>

        {/* ✅ vertical + horizontal scroll container */}
        <div
          className="mt-4 overflow-auto rounded-xl border border-border"
          style={{ maxHeight: MAX_HEIGHT_PX }}
        >
          {!trades.length ? (
            <div className="p-4 text-sm text-muted-foreground">No trades for this run.</div>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted-foreground">
                <tr>
                  {columns.map((c) => (
                    <th
                      key={c}
                      className={[
                        "py-2 pr-4 whitespace-nowrap px-3",
                        "sticky top-0 bg-background z-20", // ✅ sticky header
                        c === "symbol" ? "sticky left-0 z-30" : "",
                      ].join(" ")}
                    >
                      {labelFor(c)}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {shownTrades.map((t, i) => (
                  <tr key={i} className="border-t border-border">
                    {columns.map((c) => (
                      <td
                        key={c}
                        className={[
                          "py-2 pr-4 whitespace-nowrap px-3",
                          c === "symbol" ? "sticky left-0 bg-background z-10 font-medium" : "",
                          ["entry_price", "exit_price", "pnl_pct", "return", "bars_held", "avg_exposure", "max_exposure"].includes(c)
                            ? "text-right tabular-nums"
                            : "",
                        ].join(" ")}
                      >
                        {formatCell(c, t[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
