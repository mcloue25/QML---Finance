import React from "react";
import { Card, CardContent } from "../ui/card";
import Metric from "./ui/Metric";
import { fmtNum, fmtPct } from "./format";
import type { HeaderVM } from "./types";

export default function KpiRow({ header }: { header: HeaderVM | null }) {
  return (
    <div className="mt-6 grid gap-4 md:grid-cols-12">
      <Card className="md:col-span-12 rounded-2xl shadow-sm">
        {/* <CardContent className="grid gap-4 py-6 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-7">
          <Metric label="Annual return" value={fmtPct(header?.annualReturn)} hint="CAGR (approx)" />
          <Metric label="Annual vol" value={fmtPct(header?.vol)} />
          <Metric label="Sharpe" value={fmtNum(header?.sharpe, 2)} />
          <Metric label="Max drawdown" value={fmtNum(header?.mdd, 3)} hint="log" />
          <Metric label="Turnover / yr" value={fmtNum(header?.turnover, 2)} />
          <Metric label="Time in market" value={fmtPct(header?.tim)} />
          <Metric label="Hit rate" value={fmtPct(header?.hit)} />
        </CardContent> */}
        <CardContent className="grid gap-4 py-6 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-11">
          <Metric label="Annual return" value={fmtPct(header?.annualReturn)} hint="CAGR (approx)" />
          <Metric label="Annual vol" value={fmtPct(header?.vol)} />
          <Metric label="Sharpe" value={fmtNum(header?.sharpe, 2)} />
          <Metric label="Max drawdown" value={fmtNum(header?.mdd, 3)} hint="log" />
          <Metric label="Turnover / yr" value={fmtNum(header?.turnover, 2)} />
          <Metric label="Time in market" value={fmtPct(header?.tim)} />
          <Metric label="Hit rate" value={fmtPct(header?.hit)} />

          {/* NEW (wire these in HeaderVM if you want them typed) */}
          <Metric label="Total log return" value={fmtNum((header as any)?.totalLogReturn, 3)} />
          <Metric label="# Entries" value={fmtNum((header as any)?.nEntries, 0)} />
          <Metric label="Avg trade days" value={fmtNum((header as any)?.avgTradeDays, 2)} />
          <Metric label="Entry threshold" value={fmtNum((header as any)?.entryThreshold, 2)} />
        </CardContent>
      </Card>
    </div>
  );
}
