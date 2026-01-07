"use client";

import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";

import FiltersCard from "./FiltersCard";
import KpiRow from "./KpiRow";

import EquityCurveCard from "./charts/EquityCurveCard";
import DrawdownCard from "./charts/DrawdownCard";
import ExposureCard from "./charts/ExposureCard";
import ProbabilitiesCard from "./charts/ProbabilitiesCard";
import SignalScatterCard from "./charts/SignalScatterCard";

import type { Catalog, BaselinePayload, HeaderVM, SeriesRow } from "./types";

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

export default function RunAnalysisTab(props: {
  // filters
  assets: string[];
  catalog: Catalog | null;

  asset: string | null;
  setAsset: (v: string) => void;

  modelType: string;
  setModelType: (v: string) => void;

  horizon: number | null;
  setHorizon: (v: number) => void;

  query: string;
  setQuery: (v: string) => void;

  runId: string | null;
  setRunId: (v: string) => void;

  runIds: string[];

  // data
  header: HeaderVM | null;
  baseline: BaselinePayload | null;
  series: SeriesRow[];

  // view controls
  view: "net" | "gross";
  setView: (v: "net" | "gross") => void;

  posView: "position" | "turnover";
  setPosView: (v: "position" | "turnover") => void;

  isLoading: boolean;
}) {
  const {
    assets,
    catalog,
    asset,
    setAsset,
    modelType,
    setModelType,
    horizon,
    setHorizon,
    query,
    setQuery,
    runId,
    setRunId,
    runIds,
    header,
    baseline,
    series,
    view,
    setView,
    posView,
    setPosView,
    isLoading,
  } = props;

  return (
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

      <div className={isLoading ? "opacity-60" : ""}>
        <KpiRow header={header} />
      </div>

      <Tabs defaultValue="performance">
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

          <div className="text-xs text-muted-foreground">{isLoading ? "Loading…" : null}</div>
        </div>

        <TabsContent value="performance" className="mt-4">
          {isLoading ? (
            <div className="grid gap-4 md:grid-cols-12">
              <SkeletonCard className="md:col-span-8" />
              <SkeletonCard className="md:col-span-4" />
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-12">
              <EquityCurveCard series={series} hasBaseline={!!baseline?.bt} view={view} setView={setView} />
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
            Cross-run comparison coming next (scatter + sortable table).
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
