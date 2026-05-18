"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import PipelineProgress from "./PipelineProgress";
import CitationChip from "./CitationChip";
import DeclineState from "./DeclineState";
import FinRAGLogo from "./FinRAGLogo";
import type { Citation, QueryFilters, ChatMessage } from "@/lib/types";

// ─── Constants ──────────────────────────────────────────────────────────────

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
          return <span key={i} style={{ color: "hsl(var(--primary))" }} className="font-medium">{part}</span>;
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

// ─── Single Chat Message Bubble ──────────────────────────────────────────────

function ChatMessageBubble({
  message, filters, onCitationClick,
}: {
  message: ChatMessage;
  filters: QueryFilters;
  onCitationClick: (c: Citation) => void;
}) {
  const periodLabel = formatPeriodLabel(filters.fiscal_period);

  return (
    <div className="w-full space-y-5 animate-fade-in">
      {/* User Query */}
      <div className="flex flex-col items-end w-full">
        <div className="user-bubble max-w-[85%]">
          <p className="text-sm font-mono" style={{ color: "rgba(255,255,255,0.9)" }}>{message.query}</p>
        </div>
      </div>

      {/* Error */}
      {message.error && (
        <div className="p-4 rounded-xl" style={{ border: "1px solid rgba(239,68,68,0.3)", background: "rgba(239,68,68,0.06)", color: "#f87171" }}>
          <h4 className="font-semibold mb-1 font-mono text-sm">Error processing request</h4>
          <p className="text-xs">{message.error}</p>
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

          {/* Assistant header */}
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
              style={{
                background: "hsl(var(--primary) / 0.12)",
                border: "1px solid hsl(var(--primary) / 0.25)",
                color: "hsl(var(--primary))",
              }}
            >
              <svg width="14" height="14" viewBox="0 0 40 40" fill="none">
                <rect x="8" y="12" width="24" height="20" rx="10" fill="currentColor" opacity="0.15" stroke="currentColor" strokeWidth="2"/>
                <circle cx="15" cy="21" r="2.5" fill="currentColor"/>
                <circle cx="25" cy="21" r="2.5" fill="currentColor"/>
                <path d="M15 27 Q20 29.5 25 27" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none"/>
              </svg>
            </div>
            <span className="font-semibold text-sm" style={{ color: "rgba(255,255,255,0.85)" }}>FinRAG</span>
            {message.confidence !== null && !message.isLoading && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${
                message.confidence > 0.8 ? "bg-green-900/20 text-green-400 border border-green-900/40" :
                message.confidence > 0.5 ? "bg-yellow-900/20 text-yellow-400 border border-yellow-900/40" :
                "bg-red-900/20 text-red-400 border border-red-900/40"
              }`}>
                {(message.confidence * 100).toFixed(0)}% confidence
              </span>
            )}
          </div>

          {/* Text */}
          <div className="prose prose-sm dark:prose-invert max-w-none leading-relaxed whitespace-pre-wrap pl-11" style={{ color: "rgba(255,255,255,0.8)" }}>
            {renderAnswerWithCitations(message.answer, message.citations, onCitationClick, message.isLoading)}
            {message.isLoading && !message.answer && (
              <span className="inline-flex gap-1 pl-1">
                <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: "hsl(var(--primary) / 0.7)", animationDelay: "0ms" }} />
                <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: "hsl(var(--primary) / 0.7)", animationDelay: "150ms" }} />
                <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: "hsl(var(--primary) / 0.7)", animationDelay: "300ms" }} />
              </span>
            )}
          </div>

          {/* Sources */}
          {!message.isLoading && message.citations.length > 0 && (
            <div className="w-full mt-2 space-y-2 pl-11">
              <h4 className="text-xs font-semibold uppercase tracking-wider font-mono" style={{ color: "rgba(255,255,255,0.35)" }}>Sources</h4>
              <div className="flex flex-wrap gap-2">
                {message.citations.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => onCitationClick(c)}
                    className="source-chip"
                  >
                    <span className="font-mono flex-shrink-0" style={{ color: "hsl(var(--primary))" }}>[{i + 1}]</span>
                    <span className="font-medium truncate max-w-[140px]">{c.section}</span>
                    {c.page > 0 && <span style={{ color: "rgba(255,255,255,0.4)" }} className="flex-shrink-0">p.{c.page}</span>}
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

// ─── Hero / Empty State ───────────────────────────────────────────────────────

function HeroEmptyState({ filters, onQuickSelect }: { filters: QueryFilters; onQuickSelect: (q: string) => void }) {
  return (
    <div className="animate-fade-in w-full flex flex-col items-center space-y-8">

      {/* Logo */}
      <div className="flex flex-col items-center space-y-3">
        <FinRAGLogo size="lg" />
        <p className="text-sm text-center max-w-sm" style={{ color: "rgba(255,255,255,0.4)", lineHeight: 1.6 }}>
          Citation-enforced financial research over SEC EDGAR filings.
          Every answer is grounded in a specific paragraph.
        </p>
      </div>

      {/* Quick-start cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full max-w-xl">
        {[
          { q: "What was total revenue and net income?", icon: "📊", label: "Financials" },
          { q: "What AI-related risk factors were disclosed?", icon: "🛡️", label: "Risk Factors" },
          { q: "How did operating margin change year-over-year?", icon: "📈", label: "Margins" },
          { q: "What did management say about future guidance?", icon: "🔮", label: "Outlook" },
        ].map((item, i) => (
          <button
            key={i}
            onClick={() => onQuickSelect(item.q)}
            className="flex items-start gap-3 p-3.5 text-left rounded-2xl transition-all group"
            style={{
              border: "1px solid rgba(255,255,255,0.07)",
              background: "rgba(255,255,255,0.03)",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.06)";
              (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(255,255,255,0.12)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.03)";
              (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(255,255,255,0.07)";
            }}
          >
            <span className="text-lg leading-none mt-0.5">{item.icon}</span>
            <div>
              <div className="text-xs font-mono mb-0.5 transition-colors" style={{ color: "hsl(var(--primary))" }}>{item.label}</div>
              <div className="text-xs leading-snug" style={{ color: "rgba(255,255,255,0.55)" }}>{item.q}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Grok-style Input Section ─────────────────────────────────────────────────

function InputSection({
  query, setQuery, textareaRef, handleSubmit, handleKeyDown,
  handleSummarize, handleReset, isAnyLoading, hasMessages, filters,
}: {
  query: string;
  setQuery: (v: string) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  handleSubmit: () => void;
  handleKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  handleSummarize: () => void;
  handleReset: () => void;
  isAnyLoading: boolean;
  hasMessages: boolean;
  filters: QueryFilters;
}) {
  return (
    <div className="w-full max-w-2xl mx-auto space-y-3">

      {/* Action chips row */}
      <div className="flex items-center gap-2 px-1">
        <button
          id="summarize-report-btn"
          onClick={handleSummarize}
          disabled={isAnyLoading}
          title="Summarize the selected report"
          className="action-chip action-chip-primary"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
            className="action-chip ml-auto"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6l-1 14H6L5 6" />
              <path d="M10 11v6M14 11v6" />
              <path d="M9 6V4h6v2" />
            </svg>
            Clear
          </button>
        )}
      </div>

      {/* Pill input bar */}
      <div className="grok-input-bar flex items-end gap-2 px-4 py-3">
        {/* Expand/attach icon */}
        <button
          className="flex-shrink-0 mb-0.5 transition-colors"
          style={{ color: "rgba(255,255,255,0.3)" }}
          onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "rgba(255,255,255,0.7)")}
          onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "rgba(255,255,255,0.3)")}
          tabIndex={-1}
          aria-label="Attach file"
          title="Attach file (coming soon)"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          id="query-input"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
          }}
          onKeyDown={handleKeyDown}
          placeholder={`Ask about ${filters.ticker} ${filters.filing_type}…`}
          disabled={isAnyLoading}
          rows={1}
          className="grok-textarea flex-1 max-h-[140px] text-sm leading-relaxed py-0.5"
          style={{ fontFamily: "var(--font-sans)" }}
        />

        {/* Submit button */}
        <button
          id="submit-query-btn"
          onClick={() => handleSubmit()}
          disabled={isAnyLoading || !query.trim()}
          aria-label="Submit query"
          className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all ${
            query.trim() && !isAnyLoading ? "submit-btn-active" : "submit-btn-inactive"
          }`}
        >
          {isAnyLoading ? (
            <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          )}
        </button>
      </div>

      {/* Hint */}
      <p className="text-center text-xs px-2" style={{ color: "rgba(255,255,255,0.2)" }}>
        Press <kbd className="px-1.5 py-0.5 rounded text-xs" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)" }}>Enter</kbd> to submit · <kbd className="px-1.5 py-0.5 rounded text-xs" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)" }}>Shift+Enter</kbd> for new line
      </p>
    </div>
  );
}

// ─── Main QueryPanel ──────────────────────────────────────────────────────────

export default function QueryPanel({
  filters, onCitationClick, pendingQuery, onPendingQueryConsumed, finragState,
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

  // Auto-scroll to bottom on new messages
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
    if (textareaRef.current) textareaRef.current.style.height = "auto";
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

  const inputSectionProps = {
    query, setQuery, textareaRef, handleSubmit, handleKeyDown,
    handleSummarize, handleReset, isAnyLoading, hasMessages, filters,
  };

  // ── Empty / Hero state: full-height centering, Grok-style ──
  if (!hasMessages) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center h-full relative px-4 sm:px-6 pb-8" style={{ zIndex: 1 }}>
        <div className="w-full flex flex-col items-center gap-10 max-w-2xl">
          <HeroEmptyState
            filters={filters}
            onQuickSelect={(q) => {
              setQuery(q);
              setTimeout(() => textareaRef.current?.focus(), 0);
            }}
          />
          <InputSection {...inputSectionProps} />
        </div>
      </div>
    );
  }

  // ── Active chat state: scrollable history + fixed input ──
  return (
    <div className="flex-1 flex flex-col h-full relative" style={{ zIndex: 1 }}>

      {/* Scrollable chat history */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 sm:px-8 pt-6 pb-2 flex flex-col items-center"
      >
        <div className="w-full max-w-3xl mx-auto flex flex-col gap-8 pb-4">
          {messages.map((msg) => (
            <ChatMessageBubble
              key={msg.id}
              message={msg}
              filters={filters}
              onCitationClick={onCitationClick}
            />
          ))}
        </div>
      </div>

      {/* Fixed bottom input */}
      <div
        className="w-full flex-shrink-0 px-4 sm:px-6 pb-5 pt-3"
        style={{ borderTop: "1px solid rgba(255,255,255,0.05)", background: "rgba(0,0,0,0.4)", backdropFilter: "blur(16px)" }}
      >
        <InputSection {...inputSectionProps} />
      </div>
    </div>
  );
}
