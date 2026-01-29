"""
Hybrid Adapter: Reciprocal Rank Fusion (RRF) of keyword + embedding search.

Standard hybrid retrieval: run both retrievers in parallel, fuse with RRF.
"""
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer

RRF_K = 60  # Standard RRF constant


class HybridAdapter:
    """
    Hybrid adapter using Reciprocal Rank Fusion (RRF) of keyword + embedding search.
    Both retrievers run in parallel, results fused with RRF(k=60).
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.env_dir = None
        self.model = None
        self.session_data = []
        self.session_texts = []
        self.session_embeddings = None

    def _load_model(self):
        """Lazy load Stella V5 model"""
        if self.model is None:
            print("Loading Stella V5 for hybrid retrieval...")
            self.model = SentenceTransformer('dunzhang/stella_en_1.5B_v5')
            print("Model ready.")

    def set_environment(self, env_dir: Path):
        """Set the current question environment and pre-compute embeddings"""
        self.env_dir = env_dir
        self._load_model()

        chat_history_dir = self.env_dir / "chat_history"
        session_files = sorted(chat_history_dir.glob("*.json"))

        self.session_data = []
        self.session_texts = []
        for session_file in session_files:
            with open(session_file) as f:
                data = json.load(f)
                self.session_data.append(data)
                text_parts = [turn.get("content", "") for turn in data.get("turns", [])]
                self.session_texts.append(" ".join(text_parts))

        # Pre-compute embeddings for dense retrieval
        if self.session_texts:
            self.session_embeddings = self.model.encode(
                self.session_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI function calling tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search conversation history using RRF hybrid (keyword + embedding fusion). Returns top matching sessions.",
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
        """
        Hybrid retrieval with Reciprocal Rank Fusion (RRF):
        1. Keyword search → ranking
        2. Embedding search → ranking
        3. Fuse with RRF: score(d) = 1/(k+rank_kw) + 1/(k+rank_emb)
        """
        if not self.env_dir or len(self.session_data) == 0:
            return "Error: Environment not set"

        n = len(self.session_data)

        # Keyword ranking
        keywords = set(query.lower().split())
        kw_scores = []
        for idx, text in enumerate(self.session_texts):
            score = sum(text.lower().count(kw) for kw in keywords)
            kw_scores.append((score, idx))
        kw_scores.sort(reverse=True, key=lambda x: x[0])
        kw_rank = {idx: rank for rank, (_, idx) in enumerate(kw_scores)}

        # Embedding ranking
        query_emb = self.model.encode(
            query, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        emb_scores = np.dot(self.session_embeddings, query_emb)
        emb_rank = {idx: rank for rank, idx in enumerate(np.argsort(-emb_scores))}

        # RRF fusion
        rrf_scores = []
        for idx in range(n):
            rrf = 1.0 / (RRF_K + kw_rank[idx]) + 1.0 / (RRF_K + emb_rank[idx])
            rrf_scores.append((rrf, idx))
        rrf_scores.sort(reverse=True, key=lambda x: x[0])

        # Return top-k
        top_indices = [idx for _, idx in rrf_scores[:top_k]]
        lines = [f"Found {n} session(s). Showing top {len(top_indices)} (RRF hybrid):", ""]
        for idx in top_indices:
            lines.append("=" * 60)
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
