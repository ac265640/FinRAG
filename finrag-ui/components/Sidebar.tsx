"use client";

import { useEffect, useState } from "react";
import { getAvailableFilings } from "@/lib/api";
import { TICKERS, FILING_TYPES, EXAMPLE_QUERIES } from "@/lib/constants";
import type { QueryFilters } from "@/lib/types";
import FinRAGLogo from "./FinRAGLogo";

// ─── Icons ───────────────────────────────────────────────────────────────────

const ChartBarIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 20h18M8 20V10M12 20V4M16 20v-6" />
  </svg>
);
const ShieldIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);
const TrendingUpIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 7l-9 9-4-4L2 19M22 7h-6M22 7v6" />
  </svg>
);
const MessageIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
  </svg>
);
const AlertIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01" />
  </svg>
);
const ArrowUpRightIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M7 17L17 7M17 7H7M17 7v10" />
  </svg>
);

const QUERY_ICONS: Record<number, React.ReactNode> = {
  0: <ChartBarIcon />,
  1: <ShieldIcon />,
  2: <TrendingUpIcon />,
  3: <MessageIcon />,
  4: <AlertIcon />,
  5: <ArrowUpRightIcon />,
};

function formatFilingDate(dateStr: string): string {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr + "T00:00:00");
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return dateStr;
  }
}

// ─── Select wrapper ───────────────────────────────────────────────────────────

function GlassSelect({
  value, onChange, children, id,
}: {
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
  id?: string;
}) {
  return (
    <div className="relative">
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none text-sm rounded-xl px-3 py-2 pr-8 focus:outline-none transition-all"
        style={{
          background: "rgba(255,255,255,0.05)",
          border: "1px solid rgba(255,255,255,0.1)",
          color: "rgba(255,255,255,0.85)",
        }}
      >
        {children}
      </select>
      <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: "rgba(255,255,255,0.35)" }}>
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 4l4 4 4-4" />
        </svg>
      </div>
    </div>
  );
}

// ─── Section label ────────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <label className="text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.3)", letterSpacing: "0.1em" }}>
      {children}
    </label>
  );
}

// ─── Main Sidebar ─────────────────────────────────────────────────────────────

interface SidebarProps {
  filters: QueryFilters;
  onFiltersChange: (filters: QueryFilters) => void;
  onExampleSelect: (query: string) => void;
  isOnline: boolean;
}

export default function Sidebar({ filters, onFiltersChange, onExampleSelect, isOnline }: SidebarProps) {
  const [availableFilings, setAvailableFilings] = useState<Record<string, Record<string, string[]>>>({});
  const [loadingFilings, setLoadingFilings] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoadingFilings(true);
      const data = await getAvailableFilings();
      setAvailableFilings(data);
      setLoadingFilings(false);
    };
    load();
  }, []);

  const availableDates = availableFilings[filters.ticker]?.[filters.filing_type] ?? [];
  const hasData = availableDates.length > 0;
  const availableFormTypes = availableFilings[filters.ticker]
    ? Object.keys(availableFilings[filters.ticker])
    : [];

  useEffect(() => {
    if (loadingFilings) return;
    if (availableDates.length > 0) {
      if (!availableDates.includes(filters.fiscal_period)) {
        onFiltersChange({ ...filters, fiscal_period: availableDates[0] });
      }
    } else {
      onFiltersChange({ ...filters, fiscal_period: "" });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.ticker, filters.filing_type, loadingFilings]);

  return (
    <aside
      className="sidebar-glass w-[268px] min-w-[268px] h-full flex flex-col relative overflow-y-auto overflow-x-hidden"
      style={{ zIndex: 50 }}
    >
      {/* ── Brand / Logo ── */}
      <div className="px-5 pt-6 pb-4" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <FinRAGLogo size="sm" />
        <p className="text-xs mt-2" style={{ color: "rgba(255,255,255,0.3)" }}>AI Financial Research Assistant</p>
      </div>

      {/* ── Filters ── */}
      <div className="px-5 py-5 flex flex-col gap-5 flex-1">

        {/* Company */}
        <div className="flex flex-col gap-2">
          <SectionLabel>Company</SectionLabel>
          <GlassSelect
            id="company-select"
            value={filters.ticker}
            onChange={(v) => onFiltersChange({ ...filters, ticker: v, fiscal_period: "" })}
          >
            {TICKERS.map((t) => (
              <option key={t.value} value={t.value} style={{ background: "#111" }}>
                {t.value} — {t.fullName}
              </option>
            ))}
          </GlassSelect>
        </div>

        {/* Filing Type */}
        <div className="flex flex-col gap-2">
          <SectionLabel>Filing Type</SectionLabel>
          <div
            className="flex p-1 rounded-xl"
            style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)" }}
          >
            {FILING_TYPES.map((f) => {
              const active = filters.filing_type === f.value;
              const available = !loadingFilings && availableFormTypes.includes(f.value);
              return (
                <button
                  key={f.value}
                  onClick={() => {
                    if (available || active) {
                      onFiltersChange({ ...filters, filing_type: f.value, fiscal_period: "" });
                    }
                  }}
                  className="flex-1 text-xs font-medium py-1.5 rounded-lg transition-all"
                  style={{
                    background: active ? "hsl(var(--primary))" : "transparent",
                    color: active ? "#fff" : "rgba(255,255,255,0.45)",
                    opacity: !available && !active ? 0.4 : 1,
                    cursor: !available && !active ? "not-allowed" : "pointer",
                    boxShadow: active ? "0 2px 8px hsl(var(--primary) / 0.4)" : "none",
                  }}
                >
                  {f.value}
                </button>
              );
            })}
          </div>
        </div>

        {/* Filing Period */}
        <div className="flex flex-col gap-2">
          <SectionLabel>Filing Period</SectionLabel>
          {loadingFilings ? (
            <div
              className="h-[38px] rounded-xl animate-pulse"
              style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" }}
            />
          ) : hasData ? (
            <GlassSelect
              id="period-select"
              value={filters.fiscal_period}
              onChange={(v) => onFiltersChange({ ...filters, fiscal_period: v })}
            >
              {availableDates.map((d) => (
                <option key={d} value={d} style={{ background: "#111" }}>
                  {formatFilingDate(d)}
                </option>
              ))}
            </GlassSelect>
          ) : (
            <div
              className="text-xs py-2 px-3 rounded-xl"
              style={{
                color: "rgba(255,255,255,0.3)",
                border: "1px dashed rgba(255,255,255,0.1)",
                background: "rgba(255,255,255,0.02)",
              }}
            >
              No filings available.
            </div>
          )}
        </div>

        {/* Divider */}
        <div style={{ height: "1px", background: "rgba(255,255,255,0.06)" }} />

        {/* Suggested Prompts */}
        <div className="flex flex-col gap-3">
          <SectionLabel>Suggested Prompts</SectionLabel>
          <div className="flex flex-col gap-1">
            {EXAMPLE_QUERIES.map((eq, i) => (
              <button
                key={i}
                onClick={() => onExampleSelect(eq.query)}
                className="group flex items-start gap-3 w-full text-left px-2 py-2.5 rounded-xl transition-all"
                style={{ border: "1px solid transparent" }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.04)";
                  (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(255,255,255,0.07)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                  (e.currentTarget as HTMLButtonElement).style.borderColor = "transparent";
                }}
              >
                <div
                  className="flex-shrink-0 mt-0.5 transition-colors"
                  style={{ color: "hsl(var(--primary))", opacity: 0.6 }}
                >
                  {QUERY_ICONS[i]}
                </div>
                <div className="text-xs leading-relaxed" style={{ color: "rgba(255,255,255,0.45)" }}>
                  {eq.description}
                </div>
              </button>
            ))}
          </div>
        </div>

      </div>

      {/* ── Status Footer ── */}
      <div
        className="px-5 py-4 sticky bottom-0"
        style={{
          background: "rgba(0,0,0,0.4)",
          backdropFilter: "blur(16px)",
          borderTop: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <div className="flex items-center gap-2.5">
          <div className="relative flex h-2 w-2">
            {isOnline && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
            )}
            <span
              className="relative inline-flex rounded-full h-2 w-2"
              style={{ background: isOnline ? "#22c55e" : "#ef4444" }}
            />
          </div>
          <span className="text-xs" style={{ color: "rgba(255,255,255,0.35)" }}>
            {isOnline ? "System Online" : "System Offline"}
          </span>
        </div>
      </div>
    </aside>
  );
}
