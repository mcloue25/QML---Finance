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
import ChartCard from "../ui/ChartCard";
import LoadingBlock from "../ui/LoadingBlock";
import type { SeriesRow } from "../types";

export default function ProbabilitiesCard({ series }: { series: SeriesRow[] }) {
  return (
    <ChartCard title="Model confidence" description="Class probabilities over time">
      {!series.length ? (
        <LoadingBlock />
      ) : (
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="p0" name="P(sell)" dot={false} />
              <Line type="monotone" dataKey="p1" name="P(hold)" dot={false} />
              <Line type="monotone" dataKey="p2" name="P(buy)" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartCard>
  );
}
