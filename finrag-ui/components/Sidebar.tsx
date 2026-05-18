"use client";

import { useEffect, useState } from "react";
import { getAvailableFilings } from "@/lib/api";
import { TICKERS, FILING_TYPES, EXAMPLE_QUERIES } from "@/lib/constants";
import type { QueryFilters } from "@/lib/types";

// Icons
const ChartBarIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 20h18M8 20V10M12 20V4M16 20v-6" />
  </svg>
);
const ShieldIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);
const TrendingUpIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 7l-9 9-4-4L2 19M22 7h-6M22 7v6" />
  </svg>
);
const MessageIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
  </svg>
);
const AlertIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01" />
  </svg>
);
const ArrowUpRightIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
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
  }, [filters.ticker, filters.filing_type, loadingFilings]);

  return (
    <aside className="w-[280px] min-w-[280px] h-full bg-muted/30 border-r border-border flex flex-col relative overflow-y-auto overflow-x-hidden">
      
      {/* App Title */}
      <div className="p-6 pb-2">
        <h1 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
          FinRAG
        </h1>
        <p className="text-xs text-muted-foreground mt-1">AI Financial Research Assistant</p>
      </div>

      {/* Selectors Section */}
      <div className="p-6 flex-1 flex flex-col gap-6">
        
        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Company
          </label>
          <div className="relative">
            <select
              value={filters.ticker}
              onChange={(e) => onFiltersChange({ ...filters, ticker: e.target.value, fiscal_period: "" })}
              className="w-full appearance-none bg-background border border-border text-sm rounded-lg px-3 py-2 pr-8 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all shadow-sm"
            >
              {TICKERS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.value} — {t.fullName}
                </option>
              ))}
            </select>
            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-muted-foreground">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M2 4l4 4 4-4" />
              </svg>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Filing Type
          </label>
          <div className="flex p-1 bg-background border border-border rounded-lg shadow-sm">
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
                  className={`
                    flex-1 text-xs font-medium py-1.5 rounded-md transition-all
                    ${active ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}
                    ${!available && !active ? 'opacity-50 cursor-not-allowed' : ''}
                  `}
                >
                  {f.value}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Filing Period
          </label>
          {loadingFilings ? (
            <div className="h-[38px] bg-background border border-border rounded-lg animate-pulse" />
          ) : hasData ? (
            <div className="relative">
              <select
                value={filters.fiscal_period}
                onChange={(e) => onFiltersChange({ ...filters, fiscal_period: e.target.value })}
                className="w-full appearance-none bg-background border border-border text-sm rounded-lg px-3 py-2 pr-8 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all shadow-sm"
              >
                {availableDates.map((d) => (
                  <option key={d} value={d}>
                    {formatFilingDate(d)}
                  </option>
                ))}
              </select>
              <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-muted-foreground">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2 4l4 4 4-4" />
                </svg>
              </div>
            </div>
          ) : (
            <div className="text-xs text-muted-foreground py-2 px-3 border border-dashed border-border rounded-lg bg-background/50">
              No filings available.
            </div>
          )}
        </div>

        <hr className="border-border my-2" />

        <div className="flex flex-col gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Suggested Prompts
          </label>
          <div className="flex flex-col gap-1.5">
            {EXAMPLE_QUERIES.map((eq, i) => (
              <button
                key={i}
                onClick={() => onExampleSelect(eq.query)}
                className="group flex items-center gap-3 w-full text-left p-2 -mx-2 rounded-lg hover:bg-background hover:shadow-sm border border-transparent hover:border-border transition-all"
              >
                <div className="text-primary opacity-70 group-hover:opacity-100 transition-opacity flex-shrink-0">
                  {QUERY_ICONS[i]}
                </div>
                <div className="text-sm text-muted-foreground group-hover:text-foreground transition-colors leading-tight">
                  {eq.description}
                </div>
              </button>
            ))}
          </div>
        </div>

      </div>

      {/* Footer / Status */}
      <div className="p-4 border-t border-border bg-background/50 backdrop-blur sticky bottom-0">
        <div className="flex items-center gap-2">
          <div className="relative flex h-2.5 w-2.5">
            {isOnline && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            )}
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isOnline ? 'bg-green-500' : 'bg-red-500'}`}></span>
          </div>
          <span className="text-xs font-medium text-muted-foreground">
            {isOnline ? 'System Online' : 'System Offline'}
          </span>
        </div>
      </div>
      
    </aside>
  );
}
