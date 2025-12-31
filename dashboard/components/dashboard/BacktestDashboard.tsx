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

import type {
  Catalog,
  RunPayload,
  BaselinePayload,
  SeriesRow,
  HeaderVM,
} from "./types";

export default function BacktestDashboard() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);

  const [assets, setAssets] = useState<string[]>([]);
  const [asset, setAsset] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [modelType, setModelType] = useState("xgb");

  // Keeping horizon state around in case other UI still references it,
  // but it's no longer used to load /api/run (UUID run folders now).
  const [horizon, setHorizon] = useState<number | null>(null);

  // UUID selected from catalog.runs[asset][modelType]
  const [runId, setRunId] = useState<string | null>(null);

  const [run, setRun] = useState<RunPayload | null>(null);
  const [baseline, setBaseline] = useState<BaselinePayload | null>(null);

  const [view, setView] = useState<"net" | "gross">("net");
  const [posView, setPosView] = useState<"position" | "turnover">("position");

  // Plumbing / status
  const [runStatus, setRunStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [runError, setRunError] = useState<string | null>(null);

  // Load list of CSV files => Asset dropdown
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

  // Load catalog (models + UUID runs)
  useEffect(() => {
    (async () => {
      const res = await fetch("/api/catalog");
      if (!res.ok) return;

      const json = (await res.json()) as Catalog;
      setCatalog(json);

      // You can still default modelType from catalog if present
      setModelType((prev) => prev ?? (json.model_types?.[0] ?? "xgb"));
      // horizon no longer drives runs, but leaving default if you still show it somewhere
      setHorizon((prev) => prev ?? (json as any)?.horizons?.[0] ?? null);
    })();
  }, []);

  // Derive all UUID run ids for selected asset+model
  const allRunIds = useMemo<string[]>(() => {
    if (!catalog?.runs || !asset) return [];
    const byModel = catalog.runs[asset];
    if (!byModel) return [];
    const ids = byModel[modelType] || [];
    return Array.isArray(ids) ? ids : [];
  }, [catalog, asset, modelType]);

  // Apply query filtering to UUIDs (optional)
  const runIds = useMemo<string[]>(() => {
    if (!query) return allRunIds;
    const q = query.toLowerCase().trim();
    if (!q) return allRunIds;
    return allRunIds.filter((id) => id.toLowerCase().includes(q));
  }, [allRunIds, query]);

  // Ensure runId is valid whenever asset/model/query changes run list
  useEffect(() => {
    if (!runIds.length) {
      setRunId(null);
      return;
    }
    setRunId((prev) => (prev && runIds.includes(prev) ? prev : runIds[0]));
  }, [runIds]);

  // ✅ Load selected run payload by UUID
  useEffect(() => {
    if (!asset || !modelType || !runId) return;

    (async () => {
      setRunStatus("loading");
      setRunError(null);

      const res = await fetch(
        `/api/run?asset=${encodeURIComponent(asset)}&model=${encodeURIComponent(modelType)}&run=${encodeURIComponent(runId)}`
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

  // Baseline can still be loaded by asset (optional)
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

  // Merge run.bt + baseline by date for charting
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
  };
}, [run, runId]);



  return (
    <div className="min-h-screen bg-background">
      <TopBar header={header} runId={runId} />

      <div className="mx-auto max-w-7xl px-4 py-6">
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
          runIds={runIds}          // ✅ NEW: list of UUIDs for dropdown
          header={header}
        />

        <div className="mt-3 text-xs text-muted-foreground">
          Run load: <span className="font-medium">{runStatus}</span>
          {runError ? <span className="ml-2 text-red-500">({runError})</span> : null}
          {run?.bt ? <span className="ml-2">rows: {run.bt.length}</span> : null}
        </div>

        <KpiRow header={header} />

        <Tabs defaultValue="performance" className="mt-6">
          <TabsList className="rounded-2xl">
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="risk">Risk &amp; Exposure</TabsTrigger>
            <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
            <TabsTrigger value="compare">Compare</TabsTrigger>
          </TabsList>

          <TabsContent value="performance" className="mt-4 space-y-4">
            <div className="grid gap-4 md:grid-cols-12">
              <EquityCurveCard
                series={series}
                hasBaseline={!!baseline?.bt}
                view={view}
                setView={setView}
              />
              <DrawdownCard
                series={series}
                hasBaseline={!!baseline?.bt}
                view={view}
              />
            </div>
          </TabsContent>

          <TabsContent value="risk" className="mt-4 space-y-4">
            <div className="grid gap-4 md:grid-cols-12">
              <ExposureCard
                series={series}
                posView={posView}
                setPosView={setPosView}
                header={header}
              />
            </div>
          </TabsContent>

          <TabsContent value="diagnostics" className="mt-4 space-y-4">
            <div className="grid gap-4 md:grid-cols-12">
              <ProbabilitiesCard series={series} />
              <SignalScatterCard series={series} />
            </div>
          </TabsContent>

          <TabsContent value="compare" className="mt-4 space-y-4">
            <div className="rounded-2xl border border-border p-6 text-sm text-muted-foreground">
              Cross-run comparison coming next (scatter + sortable table using catalog.runs).
            </div>
          </TabsContent>
        </Tabs>

        {/* <div className="mt-10 text-xs text-muted-foreground">
          Data contract: run.bt rows must contain at least date, cum_ret_net, drawdown_net,
          position_lag, target_position, turnover, net_ret, strategy_ret, p_class_0/1/2, y_pred.
          Baseline.bt should include date, cum_ret_net, drawdown_net.
        </div> */}
      </div>
    </div>
  );
}
