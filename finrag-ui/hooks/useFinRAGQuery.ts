"use client";

import { useState, useCallback, useRef } from "react";
import { streamQuery } from "@/lib/api";
import type { Citation, QueryFilters, QueryResponse, PipelineStage, ChatMessage } from "@/lib/types";

// ─── Hook Return Type ──────────────────────────────────────────────────────

interface UseFinRAGQueryReturn {
  messages: ChatMessage[];
  isAnyLoading: boolean;
  submit: (query: string, filters: QueryFilters) => void;
  reset: () => void;
}

// ─── Hook ──────────────────────────────────────────────────────────────────

export function useFinRAGQuery(): UseFinRAGQueryReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // Track the running answer for the current streaming message by ID
  const activeAnswerRef = useRef<Record<string, string>>({});

  const submit = useCallback((query: string, filters: QueryFilters) => {
    if (!query.trim()) return;

    // Generate a stable ID for this message turn
    const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    activeAnswerRef.current[id] = "";

    // Append a new loading message to the history
    const newMessage: ChatMessage = {
      id,
      query,
      answer: "",
      citations: [],
      confidence: null,
      declined: false,
      declineReason: null,
      error: null,
      isLoading: true,
      currentStage: null,
      hasResult: false,
    };

    setMessages((prev) => [...prev, newMessage]);

    // Helper: update only the message matching this ID
    const updateMessage = (updater: (msg: ChatMessage) => ChatMessage) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? updater(m) : m))
      );
    };

    streamQuery(query, filters, {
      onStage: (stage: PipelineStage) => {
        updateMessage((m) => ({ ...m, currentStage: stage }));
      },

      onToken: (token: string) => {
        activeAnswerRef.current[id] = (activeAnswerRef.current[id] ?? "") + token;
        const snapshot = activeAnswerRef.current[id];
        updateMessage((m) => ({ ...m, answer: snapshot }));
      },

      onCitation: (citation: Citation) => {
        updateMessage((m) => {
          // Deduplicate by chunk_id
          const alreadyHas = m.citations.some((c) => c.chunk_id === citation.chunk_id);
          if (alreadyHas) return m;
          return { ...m, citations: [...m.citations, citation] };
        });
      },

      onComplete: (response: QueryResponse) => {
        updateMessage((m) => ({
          ...m,
          isLoading: false,
          currentStage: null,
          answer: response.answer || m.answer,
          citations: response.citations?.length ? response.citations : m.citations,
          confidence: response.confidence,
          declined: response.declined,
          declineReason: response.decline_reason,
          hasResult: true,
          error: null,
        }));
        // Clean up the running ref to avoid memory growth
        delete activeAnswerRef.current[id];
      },

      onDecline: (reason: string) => {
        updateMessage((m) => ({
          ...m,
          declined: true,
          declineReason: reason,
        }));
      },

      onError: (error: string) => {
        updateMessage((m) => ({
          ...m,
          isLoading: false,
          currentStage: null,
          error,
          hasResult: false,
        }));
        delete activeAnswerRef.current[id];
      },
    });
  }, []);

  const reset = useCallback(() => {
    activeAnswerRef.current = {};
    setMessages([]);
  }, []);

  const isAnyLoading = messages.some((m) => m.isLoading);

  return { messages, isAnyLoading, submit, reset };
}
