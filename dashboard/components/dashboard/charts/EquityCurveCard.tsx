import React from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from "recharts";
import { ToggleGroup, ToggleGroupItem } from "../../ui/toggle-group";
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import type { SeriesRow } from "../types";
import SmartTooltip from "./SmartTooltip";
import {CHART, axisTick, fmtDate} from "./chartTheme"

export default function EquityCurveCard({
  series, hasBaseline, view, setView,
}: {
  series: SeriesRow[];
  hasBaseline: boolean;
  view: "net" | "gross";
  setView: (v: "net" | "gross") => void;
}) {
  const key = view === "net" ? "cum_ret_net" : "cum_ret_gross";

  return (
    <ChartCard
      title="Equity curve"
      description="Cumulative log return (model vs baseline)"
      right={
        <ToggleGroup type="single" value={view} onValueChange={(v) => v && setView(v as any)} className="rounded-2xl">
          <ToggleGroupItem value="net" className="rounded-2xl">Net</ToggleGroupItem>
          <ToggleGroupItem value="gross" className="rounded-2xl">Gross</ToggleGroupItem>
        </ToggleGroup>
      }
    >
      {!series.length ? <LoadingBlock /> : (
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
              <CartesianGrid stroke={CHART.grid} strokeDasharray="4 4" />
              <XAxis
                dataKey="date"
                tick={axisTick}
                tickFormatter={fmtDate}
                minTickGap={24}
              />
              <YAxis tick={axisTick} width={40} />
              <Tooltip content={<SmartTooltip />} />
              {/* Legend optional; if you keep it, make it subtle */}
              <Legend wrapperStyle={{ fontSize: 12, color: CHART.axis }} />

              <Line
                type="monotone"
                dataKey={key}
                name="Model"
                dot={false}
                stroke={CHART.model}
                strokeWidth={2}
                isAnimationActive={false}
              />

              {hasBaseline ? (
                <Line
                  type="monotone"
                  dataKey="base_cum_ret"
                  name="Buy & Hold"
                  dot={false}
                  stroke={CHART.baseline}
                  strokeWidth={1.5}
                  strokeDasharray="6 4"
                  isAnimationActive={false}
                />
              ) : null}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartCard>
  );
}
