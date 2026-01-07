"use client";

import React, { useEffect, useMemo, useState } from "react";

import RunAnalysisTab from "./RunAnalysisTab";
import PortfolioTab from "./PortfolioTab";
import TradesTab from "./TradesTab";
import ModelCompareTab from "./ModelCompareTab";

import type { Catalog, RunPayload, BaselinePayload, SeriesRow, HeaderVM } from "./types";
import { DashboardNavbar } from "./DashboardNavbar";

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



function SectionHeader({ title }: { title: string }) {
  return (
    <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight">
      <span>{title}</span>
      <span className="h-px flex-1 bg-border" />
    </h2>
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

  // Getting backtest trade data & stock history
  const [trades, setTrades] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);


  // Load assets for dropdown + tickers list (left sidebar in Trades tab)
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


  // Load catalog for model types + run ids
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


  // UseEffect for loading trades data 
  useEffect(() => {
    if (!asset || !modelType || !runId) {
      setTrades([]);
      return;
    }
    (async () => {
      const res = await fetch(`/api/trades?asset=${asset}&model=${modelType}&run=${runId}`);
      const json = await res.json();
      setTrades(json.ok ? json.trades ?? [] : []);
    })();
  }, [asset, modelType, runId]);



  // UseEffect for loading stock historical data
  useEffect(() => {
    if (!asset) return;

    fetch(`/api/history?asset=${asset}`)
      .then((r) => r.json())
      .then((j) => setHistory(j.ok ? j.rows : []));
  }, [asset]);


  // Run IDs for current asset + model
  const allRunIds = useMemo<string[]>(() => {
    if (!catalog?.runs || !asset) return [];
    const byModel = catalog.runs[asset];
    if (!byModel) return [];
    const ids = byModel[modelType] || [];
    return Array.isArray(ids) ? ids : [];
  }, [catalog, asset, modelType]);



  // Filter run ids by query
  const runIds = useMemo<string[]>(() => {
    if (!query) return allRunIds;
    const q = query.toLowerCase().trim();
    if (!q) return allRunIds;
    return allRunIds.filter((id) => id.toLowerCase().includes(q));
  }, [allRunIds, query]);



  // Default runId when list changes
  useEffect(() => {
    if (!runIds.length) {
      setRunId(null);
      return;
    }
    setRunId((prev) => (prev && runIds.includes(prev) ? prev : runIds[0]));
  }, [runIds]);



  // Load a run
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



  // Baseline is per-asset (optional)
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



  // Series for charts
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



  // Header view model for KPI row (shown inside RunAnalysisTab now)
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

  const hasError = runStatus === "error";
  const isEmpty = runStatus === "ok" && !run?.bt?.length;



  return (
    <div className="min-h-screen bg-background" id="top">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-gradient-to-b from-muted/40 via-background to-background" />

      {/* Sticky navbar (full width) */}
      <DashboardNavbar />

      <div className="mx-auto max-w-7xl px-4 py-6 space-y-10">
        {/* Status
        <div className="flex justify-end text-xs text-muted-foreground">
          {runStatus === "loading"
            ? "Loading…"
            : runStatus === "ok"
            ? "Updated"
            : runStatus === "error"
            ? "Error"
            : null}
        </div> */}

        {/* Portfolio */}
        <section id="portfolio" className="scroll-mt-24 space-y-4">
          <SectionHeader title="Portfolio" />
          <PortfolioTab />
        </section>

        {/* Run analysis */}
        <section id="run" className="scroll-mt-24 space-y-4">
          <SectionHeader title="Run Analysis" />
          {hasError ? (
            <InlineAlert title="Couldn’t load this run" detail={runError} variant="error" />
          ) : null}

          {isEmpty ? (
            <InlineAlert
              title="No backtest rows found for this run."
              detail="Try another run id or asset/model."
              variant="info"
            />
          ) : null}

          <RunAnalysisTab
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
            baseline={baseline}
            series={series}
            view={view}
            setView={setView}
            posView={posView}
            setPosView={setPosView}
            isLoading={runStatus === "loading"}
          />
        </section>

        {/* Trades */}
        <section id="trades" className="scroll-mt-24 space-y-4">
          <SectionHeader title="Trades" />
          <TradesTab trades={trades} /> 
        </section>

        {/* Model comparison */}
        <section id="models" className="scroll-mt-24 space-y-4">
          <SectionHeader title="Model comparison" />
          <ModelCompareTab />
        </section>
      </div>
    </div>
  );
}
