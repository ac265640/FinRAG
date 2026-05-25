/**
 * Next.js API Route: /api/available-filings
 *
 * Proxies GET requests to the FastAPI /api/v1/available-filings endpoint.
 * Adds the backend API key server-side so the browser never needs it.
 */

import { NextRequest } from "next/server";

const rawBackendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8002";
const BACKEND_URL = rawBackendUrl.endsWith("/") ? rawBackendUrl.slice(0, -1) : rawBackendUrl;
const BACKEND_API_KEY = process.env.BACKEND_API_KEY || process.env.NEXT_PUBLIC_API_KEY || "";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_req: NextRequest) {
  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/v1/available-filings`, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${BACKEND_API_KEY}`,
        "Accept": "application/json",
      },
    });

    if (!backendRes.ok) {
      return new Response(
        JSON.stringify({ available: {}, error: `Backend error: ${backendRes.status}` }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }

    const data = await backendRes.json();
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "max-age=30", // Cache for 30s — filings don't change mid-session
      },
    });
  } catch (err) {
    console.error("[proxy] Error fetching available-filings:", err);
    return new Response(
      JSON.stringify({ available: {}, error: String(err) }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
}
