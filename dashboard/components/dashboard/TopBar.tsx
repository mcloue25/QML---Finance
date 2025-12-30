import React from "react";
import { Button } from "../ui/button";
import { Download, SlidersHorizontal } from "lucide-react";
import Pill from "./ui/Pill";
import type { HeaderVM } from "./types";

export default function TopBar({
  header,
  runId,
}: {
  header: HeaderVM | null;
  runId: string | null;
}) {
  return (
    <div className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="text-lg font-semibold">ML Policy Backtests</div>
          {header ? <Pill variant="soft">{header.asset}</Pill> : null}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            className="rounded-2xl"
            onClick={() => window.location.reload()}
          >
            <SlidersHorizontal className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <Button variant="outline" className="rounded-2xl" disabled={!runId}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>
    </div>
  );
}
