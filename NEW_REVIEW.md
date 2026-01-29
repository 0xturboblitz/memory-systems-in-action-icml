# ICML Review: "Do Memory Tools Help Agents? The Surprising Failure of Dense Retrieval"

## 1. Summary

This paper introduces LME+, an agentic adaptation of the LongMemEval benchmark, where a ReAct agent actively queries memory tools to answer questions. The authors evaluate five memory approaches: Oracle (perfect retrieval), BM25 keyword search, Stella V5 dense embeddings, filesystem access, and a hybrid combining keyword + embedding reranking. The key finding is that BM25 (62%) vastly outperforms dense embeddings (26%) in this agentic setting—contrary to static benchmark results where Stella V5 achieves 71% on LME. The hybrid approach (42%) performs worse than BM25 alone, suggesting embeddings actively demote correct results when used for reranking.

---

## 2. Claims and Evidence

**Main claims:**
1. Dense embeddings catastrophically fail in agentic settings (26% vs 62% for BM25)
2. Hybrid reranking makes performance worse
3. The retrieval-agent gap (28pp between Oracle and BM25) shows retrieval is the bottleneck

**Assessment:**

The core empirical claims are supported by the experimental tables. However, several evidence concerns arise:

- **Wilson confidence intervals overlap significantly.** BM25 [47.2, 75.4] and Hybrid [28.2, 56.8] overlap; BM25 and Stella [14.6, 40.3] overlap slightly. While effect sizes are large (Cohen's h = 0.76), the statistical power is limited.

- **The 45-point drop claim (71% static → 26% agentic) is not directly verified.** The paper cites 71% from LongMemEval but doesn't replicate the static result themselves. The experimental setups may differ (different subsets, different prompts, different configurations). This comparison across papers is suggestive but not definitive.

- **Hybrid implementation choice.** The paper uses cascade reranking (BM25 → top-10 → embed rerank → top-5). Recent literature on hybrid retrieval shows that Reciprocal Rank Fusion (RRF) and convex combination often outperform cascade approaches. The authors acknowledge this in Discussion but the main claims generalize from a suboptimal hybrid implementation.

---

## 3. Relation to Prior Work

**What's cited well:**
- LongMemEval, MemGPT, A-MEM, LoCoMo, MemoryAgentBench
- Dense vs sparse retrieval fundamentals (BM25, DPR, Stella)

**Missing or underexplored:**
- **RAG-Fusion** (arXiv 2402.03367) systematically studies RRF for hybrid search—relevant to why cascade reranking may fail.
- **ARES** and **RAGBench** provide evaluation frameworks for RAG systems that could inform methodology.
- **The original LongMemEval optimizations** (session decomposition, fact-augmented key expansion, time-aware query expansion) that achieve strong results with embeddings are not tested. The paper dismisses embeddings entirely without exploring these mitigations.

**Methodology alignment:**
The paper uses a reasonable ReAct evaluation framework. The LongMemEval paper tested multiple LLMs and retrieval configurations more extensively.

---

## 4. Strengths

**S1. Novel and timely research question.** The paper addresses an important gap: static retrieval benchmarks may not predict agentic performance. This is a genuinely underexplored question with practical implications for memory system design.

**S2. Surprising and actionable negative result.** The finding that dense embeddings underperform BM25 by 36pp is counterintuitive and, if robust, could change how practitioners build agent memory systems. Negative results are valuable.

**S3. Clear experimental design.** The Oracle condition establishes a clean upper bound (90%), isolating retrieval quality from agent architecture. The experimental protocol is well-documented.

**S4. Honest limitations section.** The paper acknowledges sample size, single embedding model, single agent architecture, and single LLM limitations. This transparency is commendable.

**S5. Reproducibility commitment.** Code and data promised to be committed with full provenance. Cost transparency ($8.02 total) is helpful.

---

## 5. Weaknesses

**W1. (Critical) Hybrid implementation may be suboptimal.** The cascade reranking approach (BM25 → top-10 → embed rerank → top-5) is not the standard for hybrid retrieval. Recent work on RAG-Fusion and industry practice shows RRF or convex fusion often outperform cascade approaches by 26-31% NDCG. The paper's claim that "hybrid makes it worse" may be an artifact of the specific fusion strategy, not a fundamental property.

**W2. (Critical) Missing comparison to LongMemEval's own optimizations.** The original LongMemEval paper proposes session decomposition, fact-augmented key expansion, and time-aware query expansion specifically to improve embedding-based retrieval. These optimizations are not tested, yet the paper concludes embeddings fail. This is an unfair comparison—like evaluating vanilla BM25 against optimized embeddings and declaring embeddings win.

**W3. (Major) Single embedding model tested.** Only Stella V5 is evaluated. Jasper (distilled from Stella), BGE, E5, and Jina embeddings may behave differently. The conclusion that "dense embeddings fail" is overgeneralized from a single model.

**W4. (Major) No analysis of WHY embeddings fail.** The paper proposes four hypotheses (H1-H4) but doesn't test any of them empirically. A proper ablation would:
- Test chunking strategies (H1)
- Compare conversation-tuned vs document-tuned embeddings (H2)
- Measure rank correlation between embedding scores and answer presence (H4)

**W5. (Moderate) Potential implementation differences in the 71% baseline.** The paper cites Stella V5 achieving 71% on static LME but doesn't verify this themselves. Implementation details (prompt templates, s2p vs s2s prefixes, max length settings) could differ. The 45-point drop may partly be methodological, not fundamental.

**W6. (Minor) Agent architecture not explored.** Only ReAct with 5 iterations tested. Other architectures (Self-RAG, reflection-based approaches) may interact differently with retrieval quality.

---

## 6. Questions for Authors

**Q1.** Have you tested Reciprocal Rank Fusion (RRF) instead of cascade reranking for the hybrid approach? This would directly address whether the hybrid failure is fundamental or implementation-specific. *(Critical for evaluating W1)*

**Q2.** Can you replicate the 71% Stella V5 result on static LME with your exact configuration (prompt templates, prefixes, max length)? What's the actual gap with identical settings? *(Important for validating the central claim)*

**Q3.** Did you explore the LongMemEval paper's proposed optimizations (fact-augmented key expansion, time-aware query expansion) for embeddings? If not, why? *(Critical for addressing W2)*

**Q4.** What is the rank correlation between embedding cosine similarity and whether the gold answer appears in the session? This would empirically test H4. *(Would strengthen the analysis)*

**Q5.** For the 5 questions where Stella V5 succeeded uniquely (mentioned in Discussion), can you characterize what made embeddings work there? *(Would refine the "when embeddings help" discussion)*

---

## 7. Minor Issues

- Line 109: BM25 parameters ($k_1=1.2$, $b=0.75$) are stated but not justified. Were these tuned?
- The "Reproducibility Validation" section (discovering a formatting bug) is procedurally honest but the -6pp change for Filesystem suggests results have moderate sensitivity to implementation details.
- Figure reference in text but figures start at Figure 1 (cost_accuracy_tradeoff).
- The paper uses "vastly outperformed" which is strong language given overlapping confidence intervals.

---

## 8. Overall Recommendation

**Overall: 3 (Weak Accept)**

This paper tackles an important and timely research question: whether static retrieval performance predicts agentic memory performance. The finding that BM25 outperforms Stella V5 by 36pp is surprising and potentially impactful. The experimental design is clean, limitations are honestly stated, and the negative result provides value.

However, the claims are stronger than the evidence supports:
- The hybrid failure may be an artifact of cascade reranking (W1)
- The LongMemEval paper's embedding optimizations weren't tested (W2)
- A single embedding model is generalized to "dense embeddings fail" (W3)
- The hypotheses for failure aren't empirically validated (W4)

If the authors can address Q1 (test RRF fusion) and Q2 (verify the static baseline) in revision, this would strengthen the paper significantly. The core observation is valuable, but the current evidence doesn't fully support the strong conclusions.

---

## 9. Confidence Score

**Confidence: 3 (Fairly confident)**

I'm familiar with the retrieval and RAG literature but less familiar with the specific memory systems literature (MemGPT, A-MEM, etc.). I may have missed nuances in the LongMemEval evaluation protocol. The core methodological concerns (hybrid implementation, single model) are well-grounded in retrieval literature.

---

# Actionable Improvements (Priority Order)

1. **Test RRF or convex fusion for hybrid** — Directly addresses the most critical weakness. If RRF hybrid still underperforms BM25, the claim is much stronger.

2. **Implement LongMemEval's embedding optimizations** — Session decomposition + fact-augmented keys could close the gap. If BM25 still wins after these, that's a much stronger finding.

3. **Add at least one more embedding model** — BGE-large or E5-large would show generalizability.

4. **Empirically test H4** — Plot embedding rank vs. gold answer presence. Simple analysis that would validate your hypothesis.

5. **Verify the 71% static baseline yourself** — Run Stella V5 on your questions with static top-5 retrieval. Report both numbers from identical setup.

6. **Consider softening the claims** — "BM25 outperforms Stella V5 with cascade reranking on this subset" is defensible. "Dense embeddings catastrophically fail" is overclaiming.

---

## References

- LongMemEval (ICLR 2025): https://arxiv.org/abs/2410.10813
- RAG-Fusion: https://arxiv.org/abs/2402.03367
- RRF for Hybrid Search: https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search
- Jasper embedding model: https://arxiv.org/html/2412.19048v2
- A-MEM: https://arxiv.org/abs/2502.12110
- MemoryAgentBench: https://arxiv.org/abs/2507.05257
