# ICML Review: "Do Memory Tools Help Agents? The Surprising Failure of Dense Retrieval"

## 1. Summary

This paper introduces LME+, an agentic adaptation of the LongMemEval benchmark, to evaluate memory systems when used by ReAct agents rather than in static retrieval settings. The authors compare five memory approaches across 50 questions: Oracle (perfect retrieval), frequency-based keyword search, dense embeddings (Stella V5), filesystem access, and a hybrid (keyword + embedding reranking). The key finding is that keyword search (62%) dramatically outperforms dense embeddings (26%) in the agentic setting—a reversal of their static benchmark performance where Stella V5 achieves ~71%. The authors identify a 28-point gap between Oracle (90%) and the best practical method, attributing this to retrieval quality rather than agent architecture.

## 2. Claims and Evidence Assessment

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Dense embeddings fail in agentic settings | 26% vs 62% (keyword) | **Partially supported** but confounded—see weaknesses |
| Keyword search outperforms dense retrieval | 36pp gap with CI overlap | Effect size large (Cohen's h=0.76), but n=50 limits statistical power |
| Hybrid makes it worse | 42% vs 62% keyword alone | Supported, though cascade implementation may be suboptimal |
| Retrieval is the bottleneck | 28pp Oracle-MCP gap | **Well supported**—this is the paper's strongest contribution |

## 3. Relation to Prior Work

**Strengths:**
- Appropriately cites LongMemEval, LoCoMo, MemGPT, and relevant memory surveys
- Correctly notes that Letta's filesystem success was on structured extraction (LoCoMo), not conversational QA

**Critical gaps identified from literature search:**

1. **Missing: Emergence AI's RAG work** — [Emergence AI achieved SOTA on LongMemEval](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) with a simple RAG approach, even surpassing oracle performance. This directly challenges the paper's framing that retrieval is the ceiling.

2. **Missing: Supermemory** — [Supermemory achieved 71.43% on Multi-Session reasoning](https://supermemory.ai/research), which is relevant context for evaluating the 62% keyword result.

3. **Missing: Zep's temporal knowledge graph approach** — [Zep achieved 71.2% on LongMemEval](https://blog.getzep.com/) using knowledge graphs, suggesting alternative architectures beyond dense/sparse retrieval.

4. **Missing: Hybrid retrieval literature** — The paper cites hybrid retrieval but doesn't implement proper fusion (RRF, learned ensembles). [Recent 2024 surveys](https://arxiv.org/html/2501.09136v1) show hybrid approaches with proper fusion achieve 10-50% gains, making the cascade implementation a strawman.

5. **Missing: Stella V5's known limitation** — [Stella V5 was trained on max_length=512](https://huggingface.co/NovaSearch/stella_en_1.5B_v5), making it unsuitable for long sessions (~5k tokens). This is a **critical confound** that explains the failure mode without supporting the broader claim about dense retrieval.

## 4. Strengths

1. **Important research question**: Testing whether static benchmark gains translate to agentic settings is practically valuable for memory system designers.

2. **Clean experimental design**: The adapter abstraction with consistent `search_memory` interface enables fair comparison. The ReAct agent implementation is standard.

3. **Surprising and honest negative result**: The finding that dense embeddings fail is counterintuitive and useful, even if the explanation needs refinement.

4. **Good reproducibility**: Code, data, and provenance committed to git. Cost transparency ($8.02 total) is helpful.

5. **Appropriate statistical reporting**: Wilson confidence intervals with acknowledgment of wide margins (±13pp) is honest.

## 5. Weaknesses

### Major Issues

~~**W1: Critical confound invalidates main claim (Severity: High)**~~

The paper claims "dense embeddings fail in agentic settings," but Stella V5's documented max_length of 512 tokens makes it fundamentally unsuitable for embedding entire sessions (~5k tokens). The failure is likely due to **implementation mismatch**, not an inherent problem with dense retrieval in agentic contexts. The paper should have:
- Chunked sessions before embedding (standard practice)
- Used a long-context embedding model (e.g., GTE-Qwen2, Jina v3)
- At minimum, acknowledged this limitation prominently

**W2: Keyword search is not BM25 (Severity: High)** — **FIXED**

~~The paper conflates "keyword search" with BM25 throughout, but Section 3.2 reveals it's actually simple frequency counting without IDF weighting, stemming, or stop word removal. This is a much weaker baseline than BM25.~~

**Resolution:** The `builtin_mcp.py` adapter has been updated to use proper BM25 scoring with:
- IDF (Inverse Document Frequency) weighting
- Document length normalization (b=0.75)
- Term frequency saturation (k1=1.2)

The implementation now matches the standard BM25 formula and pre-indexes sessions on environment setup for efficiency.

**W3: Sample size severely limits conclusions (Severity: Medium-High)**

n=50 with ±13pp confidence intervals means:
- Keyword (62%) CI: [47%, 75%]
- Stella (26%) CI: [15%, 40%]

These overlap substantially. While the effect size is large, claiming "dense embeddings catastrophically fail" from a single embedding model on 10% of the dataset is overclaiming.

~~**W4: Hybrid implementation is a strawman (Severity: Medium)**~~

The cascade approach (keyword → embedding rerank) is not how modern hybrid retrieval works. Standard approaches use Reciprocal Rank Fusion or learned ensemble weighting. The claim that "hybrid makes it worse" only applies to this specific, suboptimal implementation.

**W5: Missing obvious ablations (Severity: Medium)**

- What happens with chunked sessions for Stella V5?
- What happens with proper BM25 instead of frequency counting?
- What about other embedding models (BGE, E5, GTE)?
- What about varying top-k (1, 5, 10)?

### Minor Issues — **ALL FIXED**

- ~~Section 3.2 calls it "MCP (Keyword)" but MCP is the protocol, not the retrieval method—confusing terminology~~ **FIXED** — Now "Keyword (BM25)"
- ~~Temperature 0.7 is high for factual QA; 0.0 would reduce variance~~ **FIXED** — Now temperature 0.0 in code and paper
- ~~README.md has junk text at the bottom ("asdfasdfasdf")~~ **N/A** — Not found
- ~~Paper says "500 questions" in abstract but only 442 are available per Section 3.3~~ **FIXED** — README now says "50 questions sampled from 442 available"

## 6. Questions for Authors

1. **Q1 (Critical)**: Did you evaluate Stella V5's retrieval recall@3 on the same questions? If it's failing to retrieve the correct session, the agentic layer cannot succeed regardless of architecture. This would help disentangle retrieval failure from agent failure.

2. ~~**Q2 (Critical)**: Why was the proper BM25 implementation (`bm25.py` with IDF weighting) not used for the main experiments? The frequency-counting "keyword search" is much weaker.~~ **RESOLVED** — Now uses proper BM25.

3. **Q3 (Important)**: What is the average session length in tokens? If sessions exceed 512 tokens (Stella's training length), the embedding degradation is expected and doesn't generalize to dense retrieval broadly.

4. **Q4 (Important)**: Have you tried chunking sessions into smaller segments for dense retrieval? This is standard practice and would address the length mismatch.

5. **Q5 (Clarification)**: The paper mentions 50/500 questions, but Section 3.3 says 442 available. Which is correct, and what is the sampling procedure?

## 7. Minor Issues

- Typo: "lesystem" → "filesystem" (p.5)
- Figure 1 caption mentions "higher sophistication" which is subjective
- ~~Reference [15] (Zhou et al., 2024) on hybrid retrieval has a future arXiv ID (2507)—verify this exists~~ **FIXED** — Citation was hallucinated; replaced with Bruch et al. (2022) arXiv:2210.11934

## 8. Overall Recommendation

**Score: 2 (Weak Reject)**

**Rationale**: The paper asks an important question (do static retrieval gains transfer to agentic settings?) and provides some useful empirical data. However, the main claim that "dense embeddings fail" is **critically confounded** by using an embedding model outside its documented operating parameters (512 vs ~5k tokens). This is not a fair evaluation of dense retrieval.

The keyword search implementation also isn't proper BM25, so the comparison is between a degraded dense retriever and a degraded sparse retriever—both operating suboptimally. The sample size (n=50) further limits the conclusions.

The paper's strongest contribution—identifying the 28-point Oracle gap as a retrieval bottleneck—is valuable but not sufficiently novel given existing work showing similar patterns.

**Path to acceptance**:
1. Fix the Stella V5 evaluation (use chunking or long-context model)
2. Use proper BM25 instead of frequency counting
3. Test at least one additional embedding model
4. Increase sample size to at least 200
5. Implement proper hybrid fusion (RRF)

## 9. Confidence Score

**Confidence: 4 (Confident)**

I have expertise in retrieval systems, RAG pipelines, and agent architectures. I've verified the code implementation and consulted recent literature. My main uncertainty is whether the hybrid approach was intentionally simplified or if it represents a misunderstanding of standard practice.

---

## How to Improve This Paper

### High Priority (Required for Resubmission)

1. **Fix the embedding evaluation**: Either chunk sessions to ~256-512 tokens before embedding, or use a long-context embedding model like GTE-Qwen2-7B-instruct (supports 8k tokens) or Jina-embeddings-v3. This is the single biggest issue.

2. ~~**Use proper BM25**: Your `bm25.py` already implements this correctly—use it instead of frequency counting.~~ **DONE** — `builtin_mcp.py` now uses proper BM25.

3. **Scale up**: 200+ questions minimum. The current n=50 yields ±13pp margins that undermine the claims.

4. **Report retrieval metrics separately**: Show Recall@3 for each retrieval method before the agent even runs. This disentangles retrieval quality from agent quality.

### Medium Priority (Strengthens Paper)

5. **Test multiple embedding models**: BGE-large-en-v1.5, E5-large-v2, and GTE-Qwen2 at minimum.

6. **Implement proper hybrid**: Use Reciprocal Rank Fusion with tuned α parameter, not cascade reranking.

7. **Add ablations**: Vary top-k (1, 3, 5, 10), chunk sizes, and iteration limits.

8. **Cite recent SOTA**: Emergence AI and Supermemory results on LongMemEval provide important context.

### Low Priority (Polish)

9. Fix temperature to 0.0 for reproducibility
10. Clean up README
11. Clarify MCP terminology vs retrieval method
12. ~~Verify reference [15] arXiv ID~~ **DONE** — Replaced hallucinated citation

---

**Sources consulted:**
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
- [Emergence AI SOTA on LongMemEval](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag)
- [Supermemory Research](https://supermemory.ai/research)
- [Stella V5 HuggingFace](https://huggingface.co/NovaSearch/stella_en_1.5B_v5)
- [Agentic RAG Survey](https://arxiv.org/html/2501.09136v1)
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
- [BM25 vs Dense Retrieval Guides](https://medium.com/@siddharth_58896/rag-techniques-bm25-vs-dense-retrievers-a-complete-practical-guide-b1302ee35b7b)
