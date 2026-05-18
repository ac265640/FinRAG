"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import PipelineProgress from "./PipelineProgress";
import CitationChip from "./CitationChip";
import DeclineState from "./DeclineState";
import type { Citation, QueryFilters, ChatMessage } from "@/lib/types";

// ─── Constants ─────────────────────────────────────────────────────────────

/**
 * Returns a summarization prompt tailored to the specific filing type.
 * 8-K filings cover a single event — asking for risk factors or guidance
 * will always produce a decline. Match the prompt to what the filing contains.
 */
function getSummarizePrompt(filingType: string): string {
  switch (filingType.toUpperCase()) {
    case "10-K":
      return "Provide a comprehensive summary of this annual report, covering: (1) total revenue and net income, (2) key business segment performance, (3) major risk factors disclosed, and (4) management's outlook and forward guidance.";
    case "10-Q":
      return "Provide a summary of this quarterly report, covering: (1) quarterly revenue and net income compared to the prior year period, (2) any significant changes in business performance or segment results, and (3) management commentary on the quarter and near-term outlook.";
    case "8-K":
      return "Summarize the key event or announcement described in this 8-K filing. What happened, what are the financial or operational implications, and what did management state about it?";
    default:
      return "Provide a concise summary of the key information disclosed in this filing, including any financial figures, significant events, and management statements.";
  }
}

// ─── Interfaces ─────────────────────────────────────────────────────────────

interface FinRAGStateProps {
  messages: ChatMessage[];
  isAnyLoading: boolean;
  submit: (query: string, filters: QueryFilters) => void;
  reset: () => void;
}

interface QueryPanelProps {
  filters: QueryFilters;
  onCitationClick: (citation: Citation) => void;
  pendingQuery?: string;
  onPendingQueryConsumed?: () => void;
  finragState: FinRAGStateProps;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function formatPeriodLabel(period: string): string {
  if (!period) return "—";
  if (/^\d{4}-\d{2}-\d{2}$/.test(period)) {
    try {
      const d = new Date(period + "T00:00:00");
      return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
    } catch { return period; }
  }
  return period;
}

function renderAnswerWithCitations(
  answer: string,
  citations: Citation[],
  onCitationClick: (c: Citation) => void,
  isLoading: boolean
): React.ReactNode {
  if (!answer) return null;
  const parts = answer.split(/(\[\d+\])/g);
  return (
    <span className={isLoading ? "cursor-blink" : ""}>
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          const idx = parseInt(match[1], 10);
          const citation = citations[idx - 1];
          if (citation) {
            return (
              <CitationChip
                key={i}
                citation={citation}
                index={idx}
                onClick={() => onCitationClick(citation)}
              />
            );
          }
          return <span key={i} className="text-primary font-medium">{part}</span>;
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

// ─── Single Chat Message Bubble ─────────────────────────────────────────────

function ChatMessageBubble({
  message,
  filters,
  onCitationClick,
}: {
  message: ChatMessage;
  filters: QueryFilters;
  onCitationClick: (c: Citation) => void;
}) {
  const periodLabel = formatPeriodLabel(filters.fiscal_period);

  return (
    <div className="w-full space-y-5 animate-fade-in">
      {/* User Query Bubble */}
      <div className="flex flex-col items-end w-full">
        <div className="bg-secondary/50 border border-border px-5 py-3.5 rounded-xl rounded-tr-sm max-w-[85%] backdrop-blur-sm">
          <p className="text-foreground font-mono text-sm">{message.query}</p>
        </div>
      </div>

      {/* Error */}
      {message.error && (
        <div className="p-4 border border-red-900/50 bg-red-900/10 text-red-400 rounded-xl">
          <h4 className="font-semibold mb-1 font-mono">Error processing request</h4>
          <p className="text-sm">{message.error}</p>
        </div>
      )}

      {/* Pipeline Progress */}
      {(message.isLoading || message.currentStage) && (
        <PipelineProgress
          currentStage={message.currentStage}
          isLoading={message.isLoading}
          hasFirstToken={message.answer.length > 0}
        />
      )}

      {/* Decline State */}
      {message.declined && (
        <DeclineState
          ticker={filters.ticker}
          filingType={filters.filing_type}
          period={periodLabel}
          declineReason={message.declineReason}
          onReset={() => {}}
        />
      )}

      {/* Answer Content */}
      {!message.declined && (message.answer || message.isLoading) && (
        <div className="flex flex-col items-start w-full gap-4">

          {/* Assistant Icon & Meta */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary flex-shrink-0">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a10 10 0 1 0 10 10H12V2z" />
                <path d="M22 12a10 10 0 0 0-10-10v10h10z" />
              </svg>
            </div>
            <span className="font-semibold text-foreground text-sm">FinRAG</span>
            {message.confidence !== null && !message.isLoading && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${
                message.confidence > 0.8 ? "bg-green-900/30 text-green-400 border border-green-900/50" :
                message.confidence > 0.5 ? "bg-yellow-900/30 text-yellow-400 border border-yellow-900/50" :
                "bg-red-900/30 text-red-400 border border-red-900/50"
              }`}>
                {(message.confidence * 100).toFixed(0)}% confidence
              </span>
            )}
          </div>

          {/* Text Content */}
          <div className="prose prose-sm dark:prose-invert max-w-none text-foreground leading-relaxed whitespace-pre-wrap pl-11">
            {renderAnswerWithCitations(message.answer, message.citations, onCitationClick, message.isLoading)}
            {message.isLoading && !message.answer && (
              <span className="inline-flex gap-1 pl-1">
                <span className="w-1.5 h-1.5 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-1.5 h-1.5 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-1.5 h-1.5 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </span>
            )}
          </div>

          {/* Sources */}
          {!message.isLoading && message.citations.length > 0 && (
            <div className="w-full mt-2 space-y-2 pl-11">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider font-mono">Sources</h4>
              <div className="flex flex-wrap gap-2">
                {message.citations.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => onCitationClick(c)}
                    className="flex items-center gap-2 px-3 py-2 text-xs border border-border bg-background hover:bg-muted/50 rounded-lg transition-colors text-left max-w-sm overflow-hidden"
                  >
                    <span className="text-primary font-mono flex-shrink-0">[{i + 1}]</span>
                    <span className="font-medium truncate">{c.section}</span>
                    {c.page > 0 && <span className="text-muted-foreground flex-shrink-0">p.{c.page}</span>}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Hero / Empty State ──────────────────────────────────────────────────────

function HeroEmptyState({ filters, onQuickSelect }: { filters: QueryFilters; onQuickSelect: (q: string) => void }) {
  return (
    <div className="animate-fade-in w-full flex flex-col items-center justify-center space-y-10 h-full pb-12">

      {/* Wordmark + Description */}
      <div className="text-center space-y-5 max-w-xl">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,255,255,0.15)]">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a10 10 0 1 0 10 10H12V2z" />
              <path d="M22 12a10 10 0 0 0-10-10v10h10z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold tracking-tight font-mono text-foreground">
            <span className="text-primary">&gt;</span> FinRAG
          </h1>
        </div>

        <p className="text-muted-foreground text-base leading-relaxed">
          A <span className="text-foreground font-medium">citation-enforced</span> financial research assistant over SEC filings.
          Ask questions about{" "}
          <span className="text-primary font-mono">10-K</span>,{" "}
          <span className="text-primary font-mono">10-Q</span>, and{" "}
          <span className="text-primary font-mono">8-K</span> filings — every answer is grounded in a specific paragraph from the actual document.
        </p>

        <p className="text-muted-foreground/70 text-sm leading-relaxed">
          When the evidence does not support a claim, FinRAG{" "}
          <span className="text-foreground/80 italic">refuses to answer</span> rather than hallucinate — because in financial research, accuracy is non-negotiable.
        </p>
      </div>

      {/* Divider */}
      <div className="flex items-center gap-3 w-full max-w-md">
        <div className="flex-1 h-px bg-border/50" />
        <span className="text-xs text-muted-foreground font-mono uppercase tracking-widest">Quick Start</span>
        <div className="flex-1 h-px bg-border/50" />
      </div>

      {/* Quick-start cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
        {[
          { q: "What was total revenue and net income?", icon: "📊", label: "Financials" },
          { q: "What AI-related risk factors were disclosed?", icon: "🛡️", label: "Risk Factors" },
          { q: "How did operating margin change YoY?", icon: "📈", label: "Margins" },
          { q: "What did management say about future guidance?", icon: "🔮", label: "Outlook" },
        ].map((item, i) => (
          <button
            key={i}
            onClick={() => onQuickSelect(item.q)}
            className="flex items-start gap-3 p-4 text-left border border-border rounded-xl bg-background hover:bg-muted/40 hover:border-primary/30 transition-all shadow-sm hover:shadow-[0_0_12px_rgba(0,255,255,0.05)] group"
          >
            <span className="text-xl">{item.icon}</span>
            <div>
              <div className="text-xs font-mono text-primary/70 mb-0.5 group-hover:text-primary transition-colors">{item.label}</div>
              <div className="text-sm font-mono text-foreground/80 leading-snug">{item.q}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Main QueryPanel ─────────────────────────────────────────────────────────

export default function QueryPanel({
  filters, onCitationClick, pendingQuery, onPendingQueryConsumed, finragState
}: QueryPanelProps) {
  const { messages, isAnyLoading, submit, reset } = finragState;

  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Consume pendingQuery from sidebar suggestions
  useEffect(() => {
    if (pendingQuery) {
      setQuery(pendingQuery);
      onPendingQueryConsumed?.();
      setTimeout(() => textareaRef.current?.focus(), 0);
    }
  }, [pendingQuery, onPendingQueryConsumed]);

  // Auto-scroll to bottom on new messages or streaming updates
  useEffect(() => {
    if (scrollRef.current && messages.length > 0) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = useCallback((queryOverride?: string) => {
    const q = (queryOverride ?? query).trim();
    if (!q || isAnyLoading) return;
    submit(q, filters);
    setQuery("");

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [query, isAnyLoading, submit, filters]);

  const handleSummarize = useCallback(() => {
    if (isAnyLoading) return;
    const prompt = getSummarizePrompt(filters.filing_type);
    submit(prompt, filters);
    setQuery("");
  }, [isAnyLoading, submit, filters]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleReset = () => {
    reset();
    setQuery("");
    textareaRef.current?.focus();
  };

  const hasMessages = messages.length > 0;

  return (
    <div className="flex-1 flex flex-col h-full relative bg-background">

      {/* Scrollable Content Area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 sm:px-8 pt-8 flex flex-col items-center"
      >
        <div className="w-full max-w-3xl mx-auto flex flex-col min-h-full">

          {/* Hero Empty State */}
          {!hasMessages && (
            <HeroEmptyState
              filters={filters}
              onQuickSelect={(q) => {
                setQuery(q);
                setTimeout(() => textareaRef.current?.focus(), 0);
              }}
            />
          )}

          {/* Sequential Chat History */}
          {hasMessages && (
            <div className="flex flex-col gap-8 pb-4">
              {messages.map((msg) => (
                <ChatMessageBubble
                  key={msg.id}
                  message={msg}
                  filters={filters}
                  onCitationClick={onCitationClick}
                />
              ))}
            </div>
          )}

        </div>

      </div>

      {/* Fixed Input Area (Flex Item) */}
      <div className="w-full flex-shrink-0 px-4 sm:px-6 pb-4 sm:pb-6 pt-4 bg-background border-t border-border/10">
        <div className="max-w-3xl mx-auto space-y-2">

          {/* Action Chips */}
          <div className="flex items-center gap-2 px-1">
            <button
              id="summarize-report-btn"
              onClick={handleSummarize}
              disabled={isAnyLoading}
              title="Summarize the selected report"
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg border border-primary/30 bg-primary/5 text-primary hover:bg-primary/15 hover:border-primary/60 hover:shadow-[0_0_10px_rgba(0,255,255,0.15)] disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
              Summarize Report
            </button>

            {hasMessages && (
              <button
                id="clear-chat-btn"
                onClick={handleReset}
                disabled={isAnyLoading}
                title="Clear chat history"
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg border border-border/50 bg-transparent text-muted-foreground hover:text-foreground hover:border-border hover:bg-muted/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all ml-auto"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6l-1 14H6L5 6" />
                  <path d="M10 11v6M14 11v6" />
                  <path d="M9 6V4h6v2" />
                </svg>
                Clear Chat
              </button>
            )}
          </div>

          {/* Textarea Input */}
          <div className="relative shadow-[0_0_20px_rgba(0,255,255,0.04)] border border-border rounded-xl bg-background/80 backdrop-blur-md overflow-hidden focus-within:ring-1 focus-within:ring-primary focus-within:shadow-[0_0_30px_rgba(0,255,255,0.12)] transition-all">
            <textarea
              ref={textareaRef}
              id="query-input"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 150) + "px";
              }}
              onKeyDown={handleKeyDown}
              placeholder={`> Ask about ${filters.ticker} ${filters.filing_type}...`}
              disabled={isAnyLoading}
              rows={1}
              className="w-full max-h-[150px] resize-none bg-transparent border-0 py-4 pl-4 pr-16 text-sm font-mono placeholder:text-muted-foreground focus:ring-0 disabled:opacity-50 outline-none"
            />

            <div className="absolute right-2 bottom-2">
              <button
                id="submit-query-btn"
                onClick={() => handleSubmit()}
                disabled={isAnyLoading || !query.trim()}
                aria-label="Submit query"
                className={`
                  p-2 rounded-lg flex items-center justify-center transition-all
                  ${query.trim() && !isAnyLoading
                    ? "bg-primary text-primary-foreground shadow-[0_0_10px_rgba(0,255,255,0.4)] hover:shadow-[0_0_20px_rgba(0,255,255,0.6)]"
                    : "bg-muted text-muted-foreground cursor-not-allowed"}
                `}
              >
                {isAnyLoading ? (
                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
