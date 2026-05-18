// All TypeScript interfaces for the FinRAG frontend.
// These match the FastAPI backend response shapes exactly.

export interface QueryFilters {
  ticker: string;
  filing_type: string;
  fiscal_period: string;
}

export interface Citation {
  chunk_id: string;
  ticker: string;
  filing_type: string;
  section: string;
  page: number;
  text: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence: number;
  declined: boolean;
  decline_reason: string | null;
}

export interface StreamChunk {
  type: "stage" | "token" | "citation" | "complete" | "error";
  data: string | Citation | QueryResponse;
}

export type PipelineStage =
  | "Encoding query..."
  | "Retrieving candidate chunks..."
  | "Reranking with cross-encoder..."
  | "Enforcing citations..."
  | "Generating grounded answer..."
  | "Complete";

export type ConfidenceLevel = "high" | "medium" | "low" | "declined";

export interface TickerOption {
  value: string;
  label: string;
  fullName: string;
}

export interface FilingTypeOption {
  value: string;
  label: string;
  description: string;
}

export interface ExampleQuery {
  query: string;
  description: string;
}

/** A single turn in the sequential chat history. */
export interface ChatMessage {
  id: string;
  query: string;
  answer: string;
  citations: Citation[];
  confidence: number | null;
  declined: boolean;
  declineReason: string | null;
  error: string | null;
  isLoading: boolean;
  currentStage: PipelineStage | null;
  hasResult: boolean;
}
