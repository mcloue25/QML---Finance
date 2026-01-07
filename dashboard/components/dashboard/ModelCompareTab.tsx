"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "../ui/card";
import { Input } from "../ui/input";

export default function ModelCompareTab() {
  const [rows, setRows] = useState<any[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [q, setQ] = useState("");

  useEffect(() => {
    (async () => {
      setStatus("loading");
      try {
        const res = await fetch("/api/model-compare");
        const json = await res.json();
        if (!res.ok || !json.ok) throw new Error(json?.error || "Failed");
        setRows(json.rows ?? []);
        setStatus("ok");
      } catch {
        setRows([]);
        setStatus("error");
      }
    })();
  }, []);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    if (!qq) return rows;
    return rows.filter((r) => JSON.stringify(r).toLowerCase().includes(qq));
  }, [rows, q]);

  const columns = useMemo(() => {
    const r0 = filtered?.[0] || rows?.[0];
    return r0 ? Object.keys(r0) : [];
  }, [filtered, rows]);

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="pt-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium">Model comparison</div>
            <div className="mt-1 text-xs text-muted-foreground">
              One row per model/features config. Load from a parquet via /api/model-compare.
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Input
              className="w-[280px] rounded-2xl"
              placeholder="Search…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
            <div className="text-xs text-muted-foreground">
              {status === "loading" ? "Loading…" : status === "ok" ? `${filtered.length} rows` : status === "error" ? "Error" : null}
            </div>
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          {!filtered.length ? (
            <div className="text-sm text-muted-foreground">
              {status === "loading"
                ? "Loading model comparison table…"
                : "No rows yet. Implement /api/model-compare to return { ok:true, rows:[...] }."}
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted-foreground">
                <tr>
                  {columns.map((c) => (
                    <th key={c} className="py-2 pr-4 whitespace-nowrap">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 500).map((r, idx) => (
                  <tr key={idx} className="border-t border-border">
                    {columns.map((c) => (
                      <td key={c} className="py-2 pr-4 whitespace-nowrap">
                        {r[c] == null ? "—" : String(r[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {filtered.length > 500 ? (
          <div className="mt-3 text-xs text-muted-foreground">Showing first 500 rows.</div>
        ) : null}
      </CardContent>
    </Card>
  );
}
