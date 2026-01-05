export const CHART = {
  grid: "#e5e7eb",         // light gray
  axis: "#6b7280",         // muted text
  model: "#2563eb",        // blue
  baseline: "#9ca3af",     // gray
  drawdown: "#ef4444",     // red
  buy: "#16a34a",          // green (if you need it)
};

export const axisTick = { fontSize: 12, fill: CHART.axis };

export function fmtDate(d: string) {
  // expects YYYY-MM-DD (or ISO). Make it short and consistent.
  const dt = new Date(d);
  if (Number.isNaN(dt.getTime())) return d;
  return dt.toLocaleDateString(undefined, { year: "2-digit", month: "short" });
}

export function fmtVal(x: any) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  const v = Number(x);
  // for log-returns your scale might be “units”, so keep compact
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function fmtPct(x: any) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return `${(Number(x) * 100).toFixed(2)}%`;
}
