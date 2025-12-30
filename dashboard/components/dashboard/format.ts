export const fmtPct = (x: any) =>
  x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(2)}%`;

export const fmtNum = (x: any, d = 2) =>
  x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d);
