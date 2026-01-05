import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { ToggleGroup, ToggleGroupItem } from "../../ui/toggle-group";
import { Separator } from "../../ui/separator";
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import Metric from "../ui/Metric";
import { fmtNum, fmtPct } from "../format";
import type { HeaderVM, SeriesRow } from "../types";
import {CHART, axisTick, fmtDate} from "./chartTheme"

function clamp01(x: any) {
  const v = Number(x);
  if (!Number.isFinite(v)) return null;
  return Math.min(1, Math.max(0, v));
}

export default function ExposureCard({
  series,
  posView,
  setPosView,
  header,
}: {
  series: SeriesRow[];
  posView: "position" | "turnover";
  setPosView: (v: "position" | "turnover") => void;
  header: HeaderVM | null;
}) {
  // Clean/normalize for nicer plotting (especially positions)
  const data = useMemo(() => {
    return series.map((r) => ({
      ...r,
      position_c: clamp01((r as any).position),
      target_position_c: clamp01((r as any).target_position),
      turnover_v:
        (r as any).turnover === null || (r as any).turnover === undefined
          ? null
          : Number((r as any).turnover),
    }));
  }, [series]);

  return (
    <>
      <ChartCard
        title="Exposure & trading"
        description="Executed position (lagged) + turnover"
        right={
          <ToggleGroup
            type="single"
            value={posView}
            onValueChange={(v) => v && setPosView(v as any)}
            className="rounded-2xl"
          >
            <ToggleGroupItem value="position" className="rounded-2xl">
              Position
            </ToggleGroupItem>
            <ToggleGroupItem value="turnover" className="rounded-2xl">
              Turnover
            </ToggleGroupItem>
          </ToggleGroup>
        }
      >
        {!data.length ? (
          <LoadingBlock />
        ) : (
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                <CartesianGrid stroke={CHART.grid} strokeDasharray="4 4" />
                <XAxis
                  dataKey="date"
                  tick={axisTick}
                  tickFormatter={fmtDate}
                  minTickGap={24}
                />

                {posView === "position" ? (
                  <YAxis
                    tick={axisTick}
                    width={40}
                    domain={[0, 1]}
                    ticks={[0, 0.5, 1]}
                  />
                ) : (
                  <YAxis
                    tick={axisTick}
                    width={40}
                    tickFormatter={(v) =>
                      Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })
                    }
                    // small padding so line doesn't kiss chart bounds
                    domain={["dataMin - 0.02", "dataMax + 0.02"]}
                  />
                )}

                <Tooltip
                  content={({ active, label, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <div className="rounded-xl border bg-background/95 p-3 shadow-sm">
                        <div className="text-xs text-muted-foreground">{fmtDate(label)}</div>
                        <div className="mt-1 space-y-1">
                          {payload
                            .filter((p: any) => p.value !== null && p.value !== undefined)
                            .map((p: any) => (
                              <div
                                key={p.dataKey}
                                className="flex items-center justify-between gap-6 text-sm"
                              >
                                <div className="flex items-center gap-2">
                                  <span
                                    className="inline-block h-2 w-2 rounded-full"
                                    style={{ background: p.color }}
                                  />
                                  <span className="text-muted-foreground">{p.name}</span>
                                </div>
                                <span className="font-medium tabular-nums">
                                  {posView === "position"
                                    ? `${(Number(p.value) * 100).toFixed(0)}%`
                                    : Number(p.value).toLocaleString(undefined, {
                                        maximumFractionDigits: 2,
                                      })}
                                </span>
                              </div>
                            ))}
                        </div>
                      </div>
                    );
                  }}
                />

                <Legend wrapperStyle={{ fontSize: 12, color: CHART.axis }} />

                {posView === "position" ? (
                  <>
                    {/* subtle guides */}
                    <ReferenceLine y={0} stroke={CHART.grid} strokeDasharray="2 4" />
                    <ReferenceLine y={0.5} stroke={CHART.grid} strokeDasharray="2 4" />
                    <ReferenceLine y={1} stroke={CHART.grid} strokeDasharray="2 4" />

                    <Line
                      type="monotone"
                      dataKey="position_c"
                      name="Executed position"
                      dot={false}
                      stroke={CHART.model}
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="target_position_c"
                      name="Target position"
                      dot={false}
                      stroke={CHART.baseline}
                      strokeWidth={1.5}
                      strokeDasharray="6 4"
                      isAnimationActive={false}
                    />
                  </>
                ) : (
                  <Line
                    type="monotone"
                    dataKey="turnover_v"
                    name="Turnover"
                    dot={false}
                    stroke={CHART.model}
                    strokeWidth={2}
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </ChartCard>

      <ChartCard title="Time-in-market" description="Share of days with position > threshold (computed in metrics)">
        <div className="space-y-3">
          <div className="text-sm text-muted-foreground">
            Add: exposure histogram (bins of position_lag), plus regime shading by predicted class.
          </div>
          <Separator />
          <div className="grid grid-cols-2 gap-4">
            <Metric label="Time in market" value={fmtPct(header?.tim)} hint="position > entry threshold" />
            <Metric label="Avg turnover / yr" value={fmtNum(header?.turnover, 2)} hint="|Δposition| annualized" />
          </div>
        </div>
      </ChartCard>
    </>
  );
}
