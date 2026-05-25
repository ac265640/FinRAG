/**
 * Next.js API Route: /api/query
 *
 * Proxies SSE streaming requests to the FastAPI backend.
 * This eliminates all CORS issues — the browser only ever talks
 * to localhost:3000 (same origin), and this server-side handler
 * forwards the request to the FastAPI backend.
 */

import { NextRequest } from "next/server";

const rawBackendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8002";
const BACKEND_URL = rawBackendUrl.endsWith("/") ? rawBackendUrl.slice(0, -1) : rawBackendUrl;
const BACKEND_API_KEY = process.env.BACKEND_API_KEY || process.env.NEXT_PUBLIC_API_KEY || "";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const backendRes = await fetch(`${BACKEND_URL}/api/v1/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${BACKEND_API_KEY}`,
        "Accept": "text/event-stream",
      },
      body: JSON.stringify(body),
      // @ts-expect-error Node.js fetch supports duplex
      duplex: "half",
    });

    if (!backendRes.ok) {
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendRes.status}` }),
        { status: backendRes.status, headers: { "Content-Type": "application/json" } }
      );
    }

    // Stream the SSE response body straight back to the browser
    return new Response(backendRes.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-store",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (err) {
    console.error("[proxy] Error forwarding to backend:", err);
    return new Response(
      JSON.stringify({ error: String(err) }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
