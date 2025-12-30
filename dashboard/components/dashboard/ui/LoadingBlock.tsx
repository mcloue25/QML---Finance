import React from "react";

export default function LoadingBlock() {
  return (
    <div className="animate-pulse space-y-3">
      <div className="h-5 w-48 rounded bg-muted" />
      <div className="h-64 w-full rounded bg-muted" />
    </div>
  );
}
