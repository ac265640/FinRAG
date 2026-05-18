"use client";

import { useEffect } from "react";
import type { Citation } from "@/lib/types";

interface CitationDrawerProps {
  citation: Citation | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function CitationDrawer({ citation, isOpen, onClose }: CitationDrawerProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const edgarUrl = citation
    ? `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${citation.ticker}&type=${citation.filing_type}&dateb=&owner=include&count=10`
    : "#";

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        className={`fixed inset-0 bg-background/80 backdrop-blur-sm z-50 transition-opacity duration-300 ${isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
      />

      <div
        className={`
          fixed top-0 right-0 h-full w-full sm:w-[480px] bg-background/95 backdrop-blur-xl border-l border-primary/30 z-50 shadow-[-10px_0_30px_rgba(0,255,255,0.05)] flex flex-col
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
      >
        {citation && (
          <>
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-border bg-muted/20">
              <div>
                <h3 className="font-bold text-lg text-foreground flex items-center gap-2 font-mono">
                  &gt; {citation.ticker}
                  <span className="text-[10px] font-mono font-bold uppercase tracking-widest px-2 py-0.5 bg-primary/20 text-primary border border-primary/30 rounded">
                    {citation.filing_type}
                  </span>
                </h3>
                <p className="text-sm text-muted-foreground mt-1 line-clamp-1">
                  {citation.section} {citation.page ? `— p. ${citation.page}` : ""}
                </p>
              </div>
              
              <button
                onClick={onClose}
                className="p-2 -mr-2 rounded-full text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                aria-label="Close"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Source Document Excerpt
                </h4>
                
                <div className="relative group">
                  {/* Left quote border */}
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary rounded-l-md shadow-[0_0_10px_rgba(0,255,255,0.5)]" />
                  
                  <div className="pl-5 py-4 pr-4 bg-muted/50 rounded-r-xl rounded-l-sm border border-border/50 text-sm leading-relaxed text-foreground whitespace-pre-wrap font-mono">
                    {citation.text}
                  </div>
                </div>
              </div>

              {/* Metadata */}
              <div className="flex gap-2">
                {typeof (citation as any).relevance_score === "number" && (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-muted text-muted-foreground border border-border">
                    Relevance: {(((citation as any).relevance_score as number) * 100).toFixed(0)}%
                  </span>
                )}
                {citation.chunk_id && (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-mono font-medium bg-muted text-muted-foreground border border-border">
                    ID: {citation.chunk_id.slice(0, 8)}
                  </span>
                )}
              </div>

            </div>

            {/* Footer */}
            <div className="p-6 border-t border-border bg-muted/10">
              <a
                href={edgarUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between p-4 rounded-xl border border-border bg-background hover:bg-muted/50 transition-colors group"
              >
                <div>
                  <div className="font-bold text-foreground font-mono uppercase tracking-widest group-hover:text-primary transition-colors">
                    [ VIEW ON SEC EDGAR ]
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5 font-mono">
                    Open the original source filing in a new tab.
                  </div>
                </div>
                <div className="text-muted-foreground group-hover:text-primary transition-colors">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="7" y1="17" x2="17" y2="7"></line>
                    <polyline points="7 7 17 7 17 17"></polyline>
                  </svg>
                </div>
              </a>
            </div>
          </>
        )}
      </div>
    </>
  );
}
