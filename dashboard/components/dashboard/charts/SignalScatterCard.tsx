import React from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import type { SeriesRow } from "../types";

export default function SignalScatterCard({ series }: { series: SeriesRow[] }) {
  return (
    <ChartCard title="Signal vs return" description="Example: P(buy) vs daily net return (scatter)">
      {!series.length ? (
        <LoadingBlock />
      ) : (
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="p2" name="P(buy)" tick={{ fontSize: 12 }} />
              <YAxis dataKey="net_ret" name="Net ret" tick={{ fontSize: 12 }} />
              <Tooltip />
              <Scatter data={series} name="Daily" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartCard>
  );
}
