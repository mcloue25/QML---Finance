import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({
    assets: [],
    horizons: [],
    model_types: ["xgb"],
    runs: [],
  });
}
