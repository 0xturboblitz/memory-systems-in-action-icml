# Results (n=200)

| Variant | Accuracy | Cost | Time |
|---------|----------|------|------|
| oracle | 71.50% | $5.36 | 14:38 |
| bm25 | 57.50% | $9.72 | 16:59 |
| builtin_mcp | 58.50% | $10.15 | 19:39 |
| reranker | 57.50% | $9.59 | 37:40 |
| bge | (running) | - | ~50min |
| stella_v5 | (running ~28%) | - | ~70min/batch |
| hybrid | (pending) | - | - |

**Notes:**
- Oracle has perfect retrieval (gold evidence)
- BM25 and builtin_mcp perform similarly (~58%)
- Reranker (BM25 + cross-encoder) doesn't improve over BM25
- BGE still running, ~48min elapsed
