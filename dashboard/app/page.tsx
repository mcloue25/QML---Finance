// app/page.tsx (NO "use client")

import BacktestDashboard from "@/components/dashboard/BacktestDashboard";

export const metadata = {
  title: "Backtest Dashboard",
};

export default function Page() {
  return <BacktestDashboard />;
}
