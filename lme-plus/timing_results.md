# Timing Results

## GPU Results (Tesla T4, n=5)

| Method | Total Time | Per-Q Time | Fixed Cost | Variable/Q |
|--------|------------|------------|------------|------------|
| Stella | 81.6s | 16.3s | ~25s (model load) | ~11s (embed + LLM) |
| Hybrid | 34.1s | 6.8s | ~25s (model load) | ~2s (BM25 + LLM) |

## Estimated Scaling

| Method | n=50 | n=100 | n=200 |
|--------|------|-------|-------|
| Stella | ~10 min | ~20 min | ~40 min |
| Hybrid | ~4 min | ~7 min | ~13 min |
| BM25 | ~2 min | ~4 min | ~8 min |

## Old CPU Results (1 sample each)

| Variant | Total Time | Env Setup | Inference | API Cost | Tokens |
|---------|-----------|-----------|-----------|----------|--------|
| oracle | 18.1s | ~16s | 2.2s | $0.01 | 4,009 |
| builtin_mcp | 14.9s | ~13s | 2.2s | $0.04 | 15,983 |
| bm25 | 17.4s | ~15s | 2.4s | $0.04 | 15,983 |
| bge | 58.7s | ~55s | 3.8s | $0.03 | 10,761 |
| reranker | 24.2s | ~18s | 5.9s | $0.03 | 13,062 |
| hybrid | 150.4s | ~107s | 43.3s | $0.11 | 44,296 |

**Notes:**
- GPU speeds up model loading significantly
- Hybrid reuses Stella but filters with BM25 first (fewer chunks to embed at search time)
- BM25 has no model overhead - fastest for iteration
