import sys
from finrag.retrieval.reranker import CrossEncoderReranker
query = "Summarize the key event or announcement described in this 8-K filing. What happened, what are the financial or operational implications, and what did management state about it?"
chunk = "Tesla reported Q1 earnings with a 50% increase in revenue. Management stated they are very happy. This 8-K exhibit outlines the financial results."
reranker = CrossEncoderReranker()
res = reranker.rerank(query, [{"text": chunk}])
print("RERANK RESULT:")
print(res)
