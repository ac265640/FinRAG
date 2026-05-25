/**
 * Next.js API Route: /api/healthz
 *
 * Proxies GET requests to the FastAPI /healthz endpoint.
 * Bypasses client-side env variable requirements and CORS.
 */

import { NextRequest } from "next/server";

const rawBackendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8002";
const BACKEND_URL = rawBackendUrl.endsWith("/") ? rawBackendUrl.slice(0, -1) : rawBackendUrl;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_req: NextRequest) {
  try {
    const backendRes = await fetch(`${BACKEND_URL}/healthz`, {
      method: "GET",
      signal: AbortSignal.timeout(5000),
    });

    if (!backendRes.ok) {
      return new Response(
        JSON.stringify({ status: "error", error: `Backend status: ${backendRes.status}` }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }

    const data = await backendRes.json();
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("[proxy] Error fetching healthz:", err);
    return new Response(
      JSON.stringify({ status: "error", error: String(err) }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
}
