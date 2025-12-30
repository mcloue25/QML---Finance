import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import { ToggleGroup, ToggleGroupItem } from "../../ui/toggle-group";
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import type { SeriesRow } from "../types";

export default function EquityCurveCard({
  series,
  hasBaseline,
  view,
  setView,
}: {
  series: SeriesRow[];
  hasBaseline: boolean;
  view: "net" | "gross";
  setView: (v: "net" | "gross") => void;
}) {
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
      {!series.length ? (
        <LoadingBlock />
      ) : (
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey={view === "net" ? "cum_ret_net" : "cum_ret_gross"}
                name="Model"
                dot={false}
                strokeWidth={2}
              />
              {hasBaseline ? (
                <Line type="monotone" dataKey="base_cum_ret" name="Buy & Hold" dot={false} strokeWidth={1.5} />
              ) : null}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartCard>
  );
}
