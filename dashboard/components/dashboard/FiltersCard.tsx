import React from "react";
import { Card, CardContent } from "../ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Input } from "../ui/input";
import { Search } from "lucide-react";
import Pill from "./ui/Pill";
import { fmtNum } from "./format";
import type { Catalog, CatalogRunRow, HeaderVM } from "./types";

export default function FiltersCard(props: {
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

  runOptions: CatalogRunRow[];
  header: HeaderVM | null;
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
    runOptions,
    header,
  } = props;

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="pt-6">
        <div className="grid gap-4 md:grid-cols-12">
          <div className="md:col-span-3">
            <div className="text-xs text-muted-foreground">Asset</div>
            <Select value={asset ?? ""} onValueChange={setAsset}>
              <SelectTrigger className="mt-1 rounded-2xl">
                <SelectValue placeholder={assets.length ? "Select asset" : "Loading assets..."} />
              </SelectTrigger>
              <SelectContent>
                {(assets || []).map((a) => (
                  <SelectItem key={a} value={a}>
                    {a}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="md:col-span-3">
            <div className="text-xs text-muted-foreground">Model Type</div>
            <Select value={modelType} onValueChange={setModelType}>
              <SelectTrigger className="mt-1 rounded-2xl">
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent>
                {(catalog?.model_types || ["xgb"]).map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="md:col-span-2">
            <div className="text-xs text-muted-foreground">Horizon</div>
            <Select
              value={String(horizon ?? "")}
              onValueChange={(v) => setHorizon(Number(v))}
            >
              <SelectTrigger className="mt-1 rounded-2xl">
                <SelectValue placeholder="h" />
              </SelectTrigger>
              <SelectContent>
                {(catalog?.horizons || []).map((h) => (
                  <SelectItem key={h} value={String(h)}>
                    h={h}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="md:col-span-4">
            <div className="text-xs text-muted-foreground">
              Run (sorted by annual return)
            </div>
            <div className="mt-1 flex items-center gap-2">
              <div className="relative w-full">
                <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  className="w-full rounded-2xl pl-10"
                  placeholder="Search feature set / notes…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>
              <Select value={runId || ""} onValueChange={setRunId}>
                <SelectTrigger className="w-[240px] rounded-2xl">
                  <SelectValue placeholder="Select run" />
                </SelectTrigger>
                <SelectContent>
                  {runOptions.map((r) => (
                    <SelectItem key={r.run_id} value={r.run_id}>
                      {`${r.run_id.slice(0, 8)} • ${r.feature_set || "feat"} • SR ${fmtNum(
                        r.sharpe,
                        2
                      )}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {header ? (
          <div className="mt-5 flex flex-wrap items-center gap-2">
            <Pill>{header.model}</Pill>
            <Pill>Features: {header.featureSet}</Pill>
            <Pill>Test: {header.period}</Pill>
            <Pill>Costs: {header.costs}</Pill>
            <Pill variant="soft">{header.policy}</Pill>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
