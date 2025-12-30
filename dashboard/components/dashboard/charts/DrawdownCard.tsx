import React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import type { SeriesRow } from "../types";

export default function DrawdownCard({
  series,
  hasBaseline,
  view,
}: {
  series: SeriesRow[];
  hasBaseline: boolean;
  view: "net" | "gross";
}) {
  return (
    <ChartCard title="Drawdown" description="Log drawdown (model vs baseline)">
      {!series.length ? (
        <LoadingBlock />
      ) : (
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={series} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Area
                type="monotone"
                dataKey={view === "net" ? "drawdown_net" : "drawdown_gross"}
                name="Model"
                fillOpacity={0.25}
                strokeWidth={2}
              />
              {hasBaseline ? (
                <Area type="monotone" dataKey="base_drawdown" name="Buy & Hold" fillOpacity={0.15} strokeWidth={1.5} />
              ) : null}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartCard>
  );
}
