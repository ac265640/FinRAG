import sys
from finrag.vectorstore.chroma_store import ChromaStore
from finrag.retrieval.reranker import CrossEncoderReranker

query = "Summarize the key event or announcement described in this 8-K filing. What happened, what are the financial or operational implications, and what did management state about it?"
where = {"$and": [{"ticker": "TSLA"}, {"form_type": "8-K"}, {"filing_date": "2026-01-28"}]}

chroma = ChromaStore()
candidates = chroma.query(query, n_results=10, where=where)

reranker = CrossEncoderReranker()
res = reranker.rerank(query, candidates)
print("Candidates found:", len(candidates))
print("RERANK RESULT:", len(res))
for r in res:
    print(f"{r['reranker_score']:.6f} | {r['text'][:100]}")
