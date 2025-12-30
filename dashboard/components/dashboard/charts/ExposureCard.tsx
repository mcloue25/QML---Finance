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
import { Separator } from "../../ui/separator";
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import Metric from "../ui/Metric";
import { fmtNum, fmtPct } from "../format";
import type { HeaderVM, SeriesRow } from "../types";

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
  return (
    <>
      <ChartCard
        title="Exposure & trading"
        description="Executed position (lagged) + turnover"
        right={
          <ToggleGroup type="single" value={posView} onValueChange={(v) => v && setPosView(v as any)} className="rounded-2xl">
            <ToggleGroupItem value="position" className="rounded-2xl">Position</ToggleGroupItem>
            <ToggleGroupItem value="turnover" className="rounded-2xl">Turnover</ToggleGroupItem>
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
                <YAxis tick={{ fontSize: 12 }} domain={posView === "position" ? [0, 1] : undefined} />
                <Tooltip />
                <Legend />
                {posView === "position" ? (
                  <>
                    <Line type="monotone" dataKey="position" name="Executed position" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="target_position" name="Target position" dot={false} strokeWidth={1.5} />
                  </>
                ) : (
                  <Line type="monotone" dataKey="turnover" name="Turnover" dot={false} strokeWidth={2} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </ChartCard>

      <ChartCard
        title="Time-in-market"
        description="Share of days with position > threshold (computed in metrics)"
      >
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
