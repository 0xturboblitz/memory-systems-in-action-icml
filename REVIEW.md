# ICML Review: "Do Memory Tools Help Agents? The Surprising Failure of Dense Retrieval"

## 1. Summary

This paper introduces LME+, an agentic adaptation of the LongMemEval benchmark for evaluating memory systems in agent settings. Using a ReAct agent with GPT-4o, the authors evaluate five memory approaches (Oracle, keyword search, Stella V5 dense embeddings, filesystem, and hybrid) across 50 questions. The main finding is that keyword search (62%) dramatically outperforms dense embeddings (26%) and hybrid approaches (42%) in the agentic setting, contrary to static benchmark results where embeddings achieve 71%. The authors attribute this to retrieval quality being the primary bottleneck, evidenced by a 28-point gap between Oracle (90%) and the best practical method.

---

## 2. Claims and Evidence

**Main claims:**
1. Dense embeddings catastrophically fail in agentic settings (26% vs 62% for keyword)
2. Hybrid approaches perform worse than keyword search alone
3. A 28-point retrieval gap exists between Oracle and practical methods
4. Static benchmark performance doesn't translate to agentic settings

**Evaluation:**

The claims are supported by the presented experiments, but with significant caveats:

- **Statistical power is insufficient**: With n=50, the 95% CI for keyword (47-75%) and dense (15-40%) do overlap slightly. While the effect size is substantial (Cohen's h=0.76), the authors correctly acknowledge this limitation but may overclaim certainty. The title "surprising failure" and "catastrophically fail" language is stronger than the statistical evidence supports.

- **Implementation concerns**: The "keyword search" is described as frequency counting without IDF weighting, stop word removal, or length normalization (line 109). This is not BM25—BM25 includes TF-IDF normalization and length penalties. The paper conflates these throughout (calling it "BM25-style" in conclusions). This mischaracterization is problematic.

- **Baseline comparison gap**: The paper claims Stella V5 achieves 71% on static LME, suggesting a 45-point drop. However, static LME uses top-5 retrieval while LME+ uses top-3. This methodological difference isn't controlled for and could partially explain the gap.

---

## 3. Relation to Prior Work

**Cited work is appropriate but incomplete:**

**Missing critical citations:**

1. **A-MEM: Agentic Memory for LLM Agents** (NeurIPS 2025) - Directly addresses agentic memory with dynamic organization using Zettelkasten principles. This is highly relevant concurrent work.

2. **Memory in the Age of AI Agents** (Survey, Dec 2025) - Comprehensive taxonomy distinguishing factual, experiential, and working memory. The paper would benefit from positioning within this framework.

3. **MemoryAgentBench** - A unified benchmark specifically for evaluating agentic memory capabilities including retrieval, update, and conflict resolution. This is a direct competitor benchmark.

4. **Letta Evals** (Oct 2025) - Reports 74% with filesystem on LoCoMo, directly relevant to the filesystem baseline in this paper.

5. **Hybrid retrieval fusion literature** - The paper cites Bruch et al. but implements cascade reranking rather than RRF/convex fusion. Recent work shows RRF outperforms cascade approaches—testing this would strengthen claims.

**Methodological alignment concerns:**

The LongMemEval benchmark (ICLR 2025) specifically proposes optimizations including "session decomposition for value granularity" and "time-aware query expansion." The paper doesn't implement these optimizations for any method, which could explain poor performance across all approaches.

---

## 4. Strengths

**S1. Important and timely question.** The gap between static retrieval benchmarks and agentic performance is under-studied and practically significant. This addresses a real need in the community.

**S2. Transparent methodology and limitations.** Section 6 honestly discusses sample size limitations, implementation choices, and generalizability concerns. The reproducibility validation (Section 4.3) demonstrates methodological care.

**S3. Clear negative result with practical implications.** The finding that adding embedding reranking hurts performance (hybrid < keyword) is surprising and actionable. If robust, this would change how practitioners design memory systems.

**S4. Good experimental design structure.** The Oracle upper bound establishes what's achievable, the filesystem baseline tests whether search is necessary, and the ablation between methods is logical.

**S5. Cost-accuracy analysis.** Reporting cost/token usage alongside accuracy provides practical guidance beyond just accuracy metrics.

---

## 5. Weaknesses

**W1. (Critical) Small sample size undermines claims.** n=50 (10% of available questions) yields confidence intervals of ±13pp. The headline claim "keyword vastly outperforms dense" (36-point gap) could shrink substantially with more data. Adjacent work like τ-bench evaluates on hundreds of tasks. Scaling to at least 200 questions would strengthen claims significantly.

**W2. (Critical) Keyword search mischaracterization.** The implementation (line 107-109) is simple frequency counting, not BM25. BM25 includes:
- IDF weighting (reduces weight of common terms)
- Length normalization (penalizes long documents)
- Saturation term (diminishing returns for repeated terms)

The paper acknowledges "proper BM25 typically improves performance by 5-15%" but still calls it "BM25" in tables and conclusions. This misrepresentation affects how readers will interpret and apply these findings.

**W3. (Major) Single embedding model tested.** Only Stella V5 is evaluated. Recent benchmarks show substantial variance between embedding models. Testing BGE-M3, E5-Mistral, or OpenAI embeddings would clarify whether this is a Stella-specific issue or a general dense retrieval problem.

**W4. (Major) Top-k mismatch between static and agentic settings.** Static LME uses top-5 retrieval (71% accuracy cited); LME+ uses top-3. This 40% reduction in retrieved context isn't controlled for. Running LME+ with top-5 retrieval would isolate the static-vs-agentic effect from the k parameter effect.

**W5. (Moderate) Hybrid implementation may be suboptimal.** The cascade approach (keyword→embed rerank) is just one fusion strategy. Bruch et al. shows convex combination fusion often outperforms RRF and cascade approaches. Testing RRF with k=60 (community standard) would strengthen the hybrid failure claim.

**W6. (Moderate) Single agent architecture.** Only ReAct with GPT-4o and 5 iterations is tested. AgentArch shows significant performance variance across agent architectures. Testing with Claude or a different architecture would address generalizability.

**W7. (Minor) Hypotheses (H1-H4) are not empirically tested.** Section 4.2 proposes four hypotheses for why dense embeddings fail but doesn't test them. Simple experiments (e.g., chunking sessions to test H1, using conversation-trained embeddings to test H2) would strengthen the analysis.

---

## 6. Questions for Authors

**Q1.** Have you tested with top-5 retrieval (matching static LME) instead of top-3? How much of the 45-point performance drop (71%→26%) is attributable to this parameter choice vs. the static-to-agentic shift? *(Critical for interpreting main claims)*

**Q2.** What is the retrieval accuracy (Recall@3) for each method before the agent processes results? This would help isolate retrieval failures from agent reasoning failures. *(Important for identifying bottleneck)*

**Q3.** For the 5 questions where Stella succeeded uniquely, what distinguishes them from the 13 overlap questions? Is there a pattern (semantic vs. lexical query type) that could guide when to use each method? *(Would add nuance to recommendations)*

**Q4.** Did you consider RRF fusion instead of cascade reranking for the hybrid? Recent literature suggests RRF may be more robust. *(Affects hybrid failure interpretation)*

**Q5.** The filesystem baseline achieved 32%—what retrieval strategy did the agent develop? Did it search by keywords, browse sequentially, or use another approach? *(Interesting ablation on agent behavior)*

---

## 7. Minor Issues / Typos

- Line 108: The formula shows simple term counting, not BM25. Relabel as "keyword frequency" or "TF-only" for accuracy.
- Line 117: "top-10 candidates, rerank by embeddings to top-3" - unclear if this is hard cutoff or score-based.
- Table 1: Consider adding Recall@k for each method alongside accuracy.
- Abstract: "vastly outperformed" is informal for a scientific paper; consider "substantially outperformed."

---

## 8. Overall Recommendation

**Overall: 3 (Weak Accept)**

This paper tackles an important and timely question—whether static retrieval performance translates to agentic settings—and provides evidence for a negative answer. The surprising finding that keyword search outperforms dense embeddings is potentially valuable to the community.

However, several issues limit confidence in the conclusions:
1. The small sample size (n=50) yields wide confidence intervals that partially overlap between methods
2. The keyword baseline is mischaracterized as BM25 when it lacks core BM25 components
3. Only one embedding model is tested, limiting generalizability
4. The top-k mismatch between static (k=5) and agentic (k=3) settings confounds interpretation

If the authors can address Q1 (top-k matching) and scale to at least 100-200 questions while testing one additional embedding model, this would become a solid contribution. In its current form, the paper provides suggestive evidence that warrants publication but falls short of the strong claims made in the title and abstract.

**Recommendation for revision:**
- Scale to ≥150 questions
- Test with k=5 to match static LME
- Test one additional embedding model (e.g., BGE-M3)
- Rename "BM25" to "term frequency" throughout
- Test RRF fusion for hybrid

---

## 9. Confidence Score

**Confidence: 4 (Confident)**

I am familiar with the RAG, agent memory, and retrieval literature. I have verified the methodology against recent benchmarks and identified the key missing citations. I may have missed some nuances specific to LongMemEval implementation details.

---

## Priority Action Items for Improvement

1. **Scale evaluation to 150-200 questions** - This is the single most impactful change. Your CIs are too wide for the claims.

2. **Fix BM25 mischaracterization** - Either implement actual BM25 (with IDF, length norm) or rename throughout to "term frequency matching."

3. **Control for top-k** - Run with k=5 to match static LME and isolate the agentic effect.

4. **Test one more embedding model** - BGE-M3 or E5-Mistral would address the single-model limitation.

5. **Add Recall@k metrics** - Show retrieval accuracy separate from end-to-end accuracy.

6. **Cite recent agentic memory work** - A-MEM, MemoryAgentBench, and the Memory in AI Agents survey should be discussed.
