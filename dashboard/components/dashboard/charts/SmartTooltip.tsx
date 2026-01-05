import React from "react";
import { fmtDate, fmtVal } from "./chartTheme";

export default function SmartTooltip({
  active,
  label,
  payload,
}: any) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-xl border bg-background/95 p-3 shadow-sm">
      <div className="text-xs text-muted-foreground">{fmtDate(label)}</div>
      <div className="mt-1 space-y-1">
        {payload
          .filter((p: any) => p.value !== null && p.value !== undefined)
          .map((p: any) => (
            <div key={p.dataKey} className="flex items-center justify-between gap-6 text-sm">
              <div className="flex items-center gap-2">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: p.color }}
                />
                <span className="text-muted-foreground">{p.name}</span>
              </div>
              <span className="font-medium tabular-nums">{fmtVal(p.value)}</span>
            </div>
          ))}
      </div>
    </div>
  );
}
