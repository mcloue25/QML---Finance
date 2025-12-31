"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";

import TopBar from "./TopBar";
import FiltersCard from "./FiltersCard";
import KpiRow from "./KpiRow";

import EquityCurveCard from "./charts/EquityCurveCard";
import DrawdownCard from "./charts/DrawdownCard";
import ExposureCard from "./charts/ExposureCard";
import ProbabilitiesCard from "./charts/ProbabilitiesCard";
import SignalScatterCard from "./charts/SignalScatterCard";

import type { Catalog, RunPayload, BaselinePayload, SeriesRow, HeaderVM } from "./types";

function SkeletonCard({ className = "" }: { className?: string }) {
  return (
    <div className={`rounded-2xl border border-border bg-card/60 shadow-sm ${className}`}>
      <div className="p-5">
        <div className="h-4 w-40 rounded bg-muted/60" />
        <div className="mt-4 h-48 w-full rounded-xl bg-muted/40" />
      </div>
    </div>
  );
}

function InlineAlert({
  title,
  detail,
  variant = "error",
}: {
  title: string;
  detail?: string | null;
  variant?: "error" | "info";
}) {
  const styles =
    variant === "error"
      ? "border-red-200/60 bg-red-50/50 text-red-900 dark:border-red-900/30 dark:bg-red-950/20 dark:text-red-200"
      : "border-blue-200/60 bg-blue-50/50 text-blue-900 dark:border-blue-900/30 dark:bg-blue-950/20 dark:text-blue-200";

  return (
    <div className={`rounded-2xl border px-4 py-3 text-sm ${styles}`}>
      <div className="font-medium">{title}</div>
      {detail ? <div className="mt-1 opacity-90">{detail}</div> : null}
    </div>
  );
}

export default function BacktestDashboard() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);

  const [assets, setAssets] = useState<string[]>([]);
  const [asset, setAsset] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [modelType, setModelType] = useState("xgb");

  const [horizon, setHorizon] = useState<number | null>(null);
  const [runId, setRunId] = useState<string | null>(null);

  const [run, setRun] = useState<RunPayload | null>(null);
  const [baseline, setBaseline] = useState<BaselinePayload | null>(null);

  const [view, setView] = useState<"net" | "gross">("net");
  const [posView, setPosView] = useState<"position" | "turnover">("position");

  const [runStatus, setRunStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const res = await fetch("/api/assets");
      const json = await res.json();

      if (!res.ok || !json.ok) {
        console.error("Failed to load assets:", json);
        return;
      }

      setAssets(json.assets ?? []);
      setAsset((prev) => prev ?? (json.assets?.[0] ?? null));
    })();
  }, []);

  useEffect(() => {
    (async () => {
      const res = await fetch("/api/catalog");
      if (!res.ok) return;

      const json = (await res.json()) as Catalog;
      setCatalog(json);

      setModelType((prev) => prev ?? (json.model_types?.[0] ?? "xgb"));
      setHorizon((prev) => prev ?? (json as any)?.horizons?.[0] ?? null);
    })();
  }, []);

  const allRunIds = useMemo<string[]>(() => {
    if (!catalog?.runs || !asset) return [];
    const byModel = catalog.runs[asset];
    if (!byModel) return [];
    const ids = byModel[modelType] || [];
    return Array.isArray(ids) ? ids : [];
  }, [catalog, asset, modelType]);

  const runIds = useMemo<string[]>(() => {
    if (!query) return allRunIds;
    const q = query.toLowerCase().trim();
    if (!q) return allRunIds;
    return allRunIds.filter((id) => id.toLowerCase().includes(q));
  }, [allRunIds, query]);

  useEffect(() => {
    if (!runIds.length) {
      setRunId(null);
      return;
    }
    setRunId((prev) => (prev && runIds.includes(prev) ? prev : runIds[0]));
  }, [runIds]);

  useEffect(() => {
    if (!asset || !modelType || !runId) return;

    (async () => {
      setRunStatus("loading");
      setRunError(null);

      const res = await fetch(
        `/api/run?asset=${encodeURIComponent(asset)}&model=${encodeURIComponent(
          modelType
        )}&run=${encodeURIComponent(runId)}`
      );
      const json = await res.json();

      if (!res.ok || !json.ok) {
        setRun(null);
        setRunStatus("error");
        setRunError(json?.error || json?.detail || "Unknown error");
        return;
      }

      setRun(json as RunPayload);
      setRunStatus("ok");
    })();
  }, [asset, modelType, runId]);

  useEffect(() => {
    if (!asset) return;
    (async () => {
      try {
        const res = await fetch(`/api/baseline?asset=${encodeURIComponent(asset)}`);
        if (!res.ok) {
          setBaseline(null);
          return;
        }
        setBaseline((await res.json()) as BaselinePayload);
      } catch {
        setBaseline(null);
      }
    })();
  }, [asset]);

  const series = useMemo<SeriesRow[]>(() => {
    if (!run?.bt) return [];
    const bt = run.bt;
    const base = baseline?.bt || null;
    const baseByDate = base ? new Map(base.map((d: any) => [d.date, d])) : null;

    return bt.map((d: any) => {
      const b = baseByDate ? baseByDate.get(d.date) : null;
      return {
        date: d.date,
        cum_ret_net: d.cum_ret_net,
        cum_ret_gross: d.cum_ret_gross,
        drawdown_net: d.drawdown_net,
        drawdown_gross: d.drawdown_gross,
        position: d.position_lag,
        target_position: d.target_position,
        turnover: d.turnover,
        net_ret: d.net_ret,
        strategy_ret: d.strategy_ret,
        p0: d.p_class_0,
        p1: d.p_class_1,
        p2: d.p_class_2,
        y_pred: d.y_pred,
        base_cum_ret: b?.cum_ret_net ?? null,
        base_drawdown: b?.drawdown_net ?? null,
      };
    });
  }, [run, baseline]);

  const header = useMemo<HeaderVM | null>(() => {
    if (!run?.meta || !run?.metrics) return null;

    const model = run.meta.model_type?.toUpperCase?.() || "MODEL";
    const runLabel = runId ? ` • run=${runId.slice(0, 8)}` : "";

    return {
      asset: run.meta.asset,
      model: `${model}${runLabel}`,
      featureSet: (run.meta as any).feature_set || "—",
      period: `${(run.meta as any).test_start || "?"} → ${(run.meta as any).test_end || "?"}`,
      policy: (run.meta as any).policy_summary || "Policy-aware sizing + costs",
      costs:
        (run.meta as any).transaction_cost_bps != null
          ? `${(run.meta as any).transaction_cost_bps} bps`
          : "—",
      annualReturn: run.metrics.annual_return,
      vol: run.metrics.annual_volatility,
      sharpe: run.metrics.sharpe,
      mdd: run.metrics.max_drawdown_log,
      turnover: run.metrics.avg_turnover_per_year,
      tim: run.metrics.pct_time_in_market,
      hit: run.metrics.hit_rate,
      totalLogReturn: (run.metrics as any).total_log_return ?? null,
      nEntries: (run.metrics as any).n_entries ?? null,
      avgTradeDays: (run.metrics as any).avg_trade_duration_days ?? null,
      entryThreshold: (run.metrics as any).entry_threshold ?? null,
    };
  }, [run, runId]);

  const isLoading = runStatus === "loading";
  const hasError = runStatus === "error";
  const isEmpty = runStatus === "ok" && !run?.bt?.length;

  return (
    <div className="min-h-screen bg-background">
      {/* App shell background */}
      <div className="pointer-events-none fixed inset-0 -z-10 bg-gradient-to-b from-muted/40 via-background to-background" />

      {/* Sticky header */}
      {/* <div className="sticky top-0 z-30 border-b border-border/60 bg-background/80 backdrop-blur">
        <TopBar header={header} runId={runId} />
      </div> */}

      <div className="mx-auto max-w-7xl px-4 py-6">
        <div className="space-y-4">
          <FiltersCard
            assets={assets}
            catalog={catalog}
            asset={asset}
            setAsset={setAsset}
            modelType={modelType}
            setModelType={setModelType}
            horizon={horizon}
            setHorizon={setHorizon}
            query={query}
            setQuery={setQuery}
            runId={runId}
            setRunId={setRunId}
            runIds={runIds}
            header={header}
          />

          {hasError ? (
            <InlineAlert title="Couldn’t load this run" detail={runError} variant="error" />
          ) : null}

          {isEmpty ? (
            <InlineAlert title="No backtest rows found for this run." detail="Try another run id or asset/model." variant="info" />
          ) : null}

          {/* KPIs can still show skeleton while loading */}
          <div className={isLoading ? "opacity-60" : ""}>
            <KpiRow header={header} />
          </div>
        </div>

        <Tabs defaultValue="performance" className="mt-6">
          <div className="flex items-center justify-between gap-3">
            <TabsList className="h-10 rounded-2xl bg-muted/40 p-1">
              <TabsTrigger className="rounded-xl px-3" value="performance">
                Performance
              </TabsTrigger>
              <TabsTrigger className="rounded-xl px-3" value="risk">
                Risk &amp; Exposure
              </TabsTrigger>
              <TabsTrigger className="rounded-xl px-3" value="diagnostics">
                Diagnostics
              </TabsTrigger>
              <TabsTrigger className="rounded-xl px-3" value="compare">
                Compare
              </TabsTrigger>
            </TabsList>

            {/* Optional: place lightweight status on the right */}
            <div className="text-xs text-muted-foreground">
              {isLoading ? "Loading…" : runStatus === "ok" ? "Updated" : null}
            </div>
          </div>

          <TabsContent value="performance" className="mt-4">
            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-12">
                <SkeletonCard className="md:col-span-8" />
                <SkeletonCard className="md:col-span-4" />
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-12">
                <EquityCurveCard
                  series={series}
                  hasBaseline={!!baseline?.bt}
                  view={view}
                  setView={setView}
                />
                <DrawdownCard series={series} hasBaseline={!!baseline?.bt} view={view} />
              </div>
            )}
          </TabsContent>

          <TabsContent value="risk" className="mt-4">
            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-12">
                <SkeletonCard className="md:col-span-12" />
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-12">
                <ExposureCard series={series} posView={posView} setPosView={setPosView} header={header} />
              </div>
            )}
          </TabsContent>

          <TabsContent value="diagnostics" className="mt-4">
            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-12">
                <SkeletonCard className="md:col-span-6" />
                <SkeletonCard className="md:col-span-6" />
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-12">
                <ProbabilitiesCard series={series} />
                <SignalScatterCard series={series} />
              </div>
            )}
          </TabsContent>

          <TabsContent value="compare" className="mt-4">
            <div className="rounded-2xl border border-border bg-card/60 p-6 text-sm text-muted-foreground shadow-sm">
              Cross-run comparison coming next (scatter + sortable table using catalog.runs).
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
