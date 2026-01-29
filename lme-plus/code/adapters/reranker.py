"""
Reranker Adapter: Two-stage BM25 → Cross-Encoder pipeline with chunking.

Stage 1: BM25 retrieves top-20 candidates (high recall)
Stage 2: Cross-encoder reranks chunked sessions to top-3 (high precision)

Cross-encoder models have ~512 token limits, so sessions are chunked
and the best-scoring chunk is used for each session.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sentence_transformers import CrossEncoder

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


class RerankerAdapter:
    """
    Two-stage retrieval pipeline:
    1. BM25 for initial retrieval (fast, sparse)
    2. Cross-encoder for reranking (accurate, dense)

    Uses MS-MARCO trained cross-encoder which is standard for passage reranking.
    """
    def __init__(self, data_dir: Path,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 bm25_k1: float = 1.2,
                 bm25_b: float = 0.75,
                 first_stage_k: int = 20):
        self.data_dir = data_dir
        self.cross_encoder_model = cross_encoder_model
        self.k1 = bm25_k1
        self.b = bm25_b
        self.first_stage_k = first_stage_k

        self.env_dir = None
        self.cross_encoder = None

        # BM25 index (computed on set_environment)
        self.session_data = []
        self.session_texts = []  # Full text for reranking
        self.session_tokens = []  # Tokenized for BM25
        self.doc_freqs = Counter()
        self.avgdl = 0
        self.N = 0

    def _load_cross_encoder(self):
        """Lazy load cross-encoder model"""
        if self.cross_encoder is None:
            print(f"Loading cross-encoder model: {self.cross_encoder_model}...")
            self.cross_encoder = CrossEncoder(self.cross_encoder_model)
            print("Cross-encoder loaded.")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into ~400-token chunks with overlap."""
        words = text.split()
        words_per_chunk = int(CHUNK_SIZE * 1.3)
        overlap_words = int(CHUNK_OVERLAP * 1.3)

        if len(words) <= words_per_chunk:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunks.append(" ".join(words[start:end]))
            start = end - overlap_words
            if start >= len(words) - overlap_words:
                break
        return chunks

    def set_environment(self, env_dir: Path):
        """Set the current question environment and build BM25 index"""
        self.env_dir = env_dir
        self._load_cross_encoder()

        # Load all sessions
        chat_history_dir = self.env_dir / "chat_history"
        session_files = sorted(chat_history_dir.glob("*.json"))

        self.session_data = []
        self.session_texts = []
        self.session_tokens = []
        self.doc_freqs = Counter()

        total_tokens = 0

        for session_file in session_files:
            with open(session_file) as f:
                data = json.load(f)
                self.session_data.append(data)

                # Create text representation
                text = self._session_to_text(data)
                self.session_texts.append(text)

                # Tokenize for BM25
                tokens = self._tokenize(text)
                self.session_tokens.append(tokens)

                # Document frequency
                unique_terms = set(tokens)
                for term in unique_terms:
                    self.doc_freqs[term] += 1

                total_tokens += len(tokens)

        self.N = len(session_files)
        self.avgdl = total_tokens / self.N if self.N > 0 else 1

        print(f"Indexed {self.N} sessions for BM25+reranking")

    def _session_to_text(self, session_data: Dict) -> str:
        """Convert session to text"""
        parts = []
        for turn in session_data.get("turns", []):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _idf(self, term: str) -> float:
        """IDF with BM25 formula"""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """BM25 score"""
        doc_len = len(doc_tokens)
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score

    def _bm25_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Stage 1: BM25 retrieval"""
        query_tokens = self._tokenize(query)

        scored = []
        for idx, doc_tokens in enumerate(self.session_tokens):
            score = self._bm25_score(query_tokens, doc_tokens)
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _cross_encoder_rerank(self, query: str, candidates: List[Tuple[int, float]],
                              top_k: int) -> List[Tuple[int, float, float]]:
        """
        Stage 2: Cross-encoder reranking with chunking.
        Chunks each candidate session and takes max score per session.

        Returns: List of (idx, bm25_score, cross_encoder_score)
        """
        if not candidates:
            return []

        # Chunk each candidate and prepare query-chunk pairs
        all_pairs = []
        chunk_to_candidate = []
        for cand_idx, (session_idx, _) in enumerate(candidates):
            chunks = self._chunk_text(self.session_texts[session_idx])
            for chunk in chunks:
                all_pairs.append((query, chunk))
                chunk_to_candidate.append(cand_idx)

        # Get cross-encoder scores for all chunks
        chunk_scores = self.cross_encoder.predict(all_pairs, show_progress_bar=False)

        # Get best chunk score for each candidate
        candidate_ce_scores = [float('-inf')] * len(candidates)
        for chunk_idx, score in enumerate(chunk_scores):
            cand_idx = chunk_to_candidate[chunk_idx]
            candidate_ce_scores[cand_idx] = max(candidate_ce_scores[cand_idx], float(score))

        # Combine with original indices and BM25 scores
        reranked = [
            (session_idx, bm25_score, candidate_ce_scores[cand_idx])
            for cand_idx, (session_idx, bm25_score) in enumerate(candidates)
        ]

        # Sort by cross-encoder score (descending)
        reranked.sort(key=lambda x: x[2], reverse=True)

        return reranked[:top_k]

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI function calling tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search conversation history using two-stage retrieval (BM25 + neural reranking). Returns top matching sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant conversations"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top sessions to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def execute_tool(self, function_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool call"""
        if function_name == "search_memory":
            top_k = args.get("top_k", 5)
            return self._search_memory(args["query"], top_k)
        else:
            return f"Unknown function: {function_name}"

    def _search_memory(self, query: str, top_k: int = 5) -> str:
        """Two-stage retrieval: BM25 → cross-encoder"""
        if not self.env_dir or len(self.session_tokens) == 0:
            return "Error: Environment not set"

        # Stage 1: BM25 retrieval
        bm25_candidates = self._bm25_retrieve(query, self.first_stage_k)

        if not bm25_candidates:
            return f"No sessions found matching: {query}"

        # Stage 2: Cross-encoder reranking
        reranked = self._cross_encoder_rerank(query, bm25_candidates, top_k)

        # Format results
        lines = [
            f"Found {len(bm25_candidates)} BM25 candidates, reranked to top {len(reranked)}:",
            ""
        ]

        for idx, bm25_score, ce_score in reranked:
            lines.append("=" * 60)
            lines.append(f"[BM25: {bm25_score:.2f} | Reranker: {ce_score:.3f}]")
            lines.append(self._format_session(self.session_data[idx]))

        return "\n".join(lines)

    def _format_session(self, session_data: Dict) -> str:
        """Format session data for LLM context"""
        lines = [
            f"Session ID: {session_data.get('session_id', 'unknown')}",
            f"Date: {session_data.get('date', 'unknown')}",
            ""
        ]

        for turn in session_data.get("turns", []):
            role = turn.get("role", "unknown").title()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
            lines.append("")

        return "\n".join(lines)
