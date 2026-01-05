import React from "react";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from "recharts";
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import type { SeriesRow } from "../types";
import SmartTooltip from "./SmartTooltip";
import {CHART, axisTick, fmtDate} from "./chartTheme"

export default function DrawdownCard({
  series, hasBaseline, view,
}: {
  series: SeriesRow[];
  hasBaseline: boolean;
  view: "net" | "gross";
}) {
  const key = view === "net" ? "drawdown_net" : "drawdown_gross";

  return (
    <ChartCard title="Drawdown" description="Underwater curve (model vs baseline)">
      {!series.length ? <LoadingBlock /> : (
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={series} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
              <CartesianGrid stroke={CHART.grid} strokeDasharray="4 4" />
              <XAxis dataKey="date" tick={axisTick} tickFormatter={fmtDate} minTickGap={24} />
              <YAxis tick={axisTick} width={40} />
              <Tooltip content={<SmartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: CHART.axis }} />

              <Area
                type="monotone"
                dataKey={key}
                name="Model"
                stroke={CHART.drawdown}
                fill={CHART.drawdown}
                fillOpacity={0.18}
                strokeWidth={2}
                isAnimationActive={false}
              />

              {hasBaseline ? (
                <Area
                  type="monotone"
                  dataKey="base_drawdown"
                  name="Buy & Hold"
                  stroke={CHART.baseline}
                  fill={CHART.baseline}
                  fillOpacity={0.10}
                  strokeWidth={1.5}
                  strokeDasharray="6 4"
                  isAnimationActive={false}
                />
              ) : null}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartCard>
  );
}
