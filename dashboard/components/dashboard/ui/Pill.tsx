import React from "react";

export default function Pill({
  children,
  variant = "outline",
}: {
  children: React.ReactNode;
  variant?: "outline" | "soft";
}) {
  const base =
    "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium";
  const styles =
    variant === "soft"
      ? "bg-muted text-foreground"
      : "border border-border text-foreground";
  return <span className={`${base} ${styles}`}>{children}</span>;
}
