"""
BGE Adapter: Dense retrieval using BGE embeddings with chunking.

BGE-large-en-v1.5 has max_length=512 tokens, so sessions are chunked
to avoid truncation.
"""
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


class BGEAdapter:
    """
    BGE (BAAI General Embedding) adapter using dense retrieval with chunking.
    Sessions are split into ~400-token chunks before embedding.
    """
    def __init__(self, data_dir: Path, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.env_dir = None
        self.model = None
        self.chunk_embeddings = []
        self.chunk_to_session = []
        self.session_data = []

    def _load_model(self):
        """Lazy load BGE model"""
        if self.model is None:
            print(f"Loading {self.model_name} model (this may take a minute)...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded.")

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
        """Set the current question environment and pre-compute chunked embeddings"""
        self.env_dir = env_dir
        self._load_model()

        # Load all sessions
        chat_history_dir = self.env_dir / "chat_history"
        session_files = sorted(chat_history_dir.glob("*.json"))

        self.session_data = []
        all_chunks = []
        self.chunk_to_session = []

        for session_idx, session_file in enumerate(session_files):
            with open(session_file) as f:
                data = json.load(f)
                self.session_data.append(data)

                # Create text representation
                text_parts = []
                for turn in data.get("turns", []):
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    text_parts.append(f"{role}: {content}")

                session_text = "\n".join(text_parts)

                # Chunk the session
                chunks = self._chunk_text(session_text)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    self.chunk_to_session.append(session_idx)

        # Pre-compute embeddings for all chunks
        print(f"Embedding {len(all_chunks)} chunks from {len(self.session_data)} sessions with BGE...")
        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print("Embeddings ready.")

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI function calling tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search conversation history using semantic similarity. Returns top matching sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find semantically relevant conversations"
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
        """Dense retrieval: embed query and find nearest chunks, return parent sessions"""
        if not self.env_dir or len(self.chunk_embeddings) == 0:
            return "Error: Environment not set"

        # BGE recommends instruction prefix for queries
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"

        # Embed query
        query_embedding = self.model.encode(
            query_with_instruction,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Compute cosine similarity with all chunks
        similarities = np.dot(self.chunk_embeddings, query_embedding)

        # Get top chunks and map to unique sessions
        sorted_indices = np.argsort(similarities)[::-1]
        seen_sessions = set()
        top_session_indices = []
        top_similarities = []

        for chunk_idx in sorted_indices:
            session_idx = self.chunk_to_session[chunk_idx]
            if session_idx not in seen_sessions:
                seen_sessions.add(session_idx)
                top_session_indices.append(session_idx)
                top_similarities.append(similarities[chunk_idx])
                if len(top_session_indices) >= top_k:
                    break

        # Format results
        lines = [f"Found top {len(top_session_indices)} semantically similar session(s):", ""]
        for idx, sim in zip(top_session_indices, top_similarities):
            lines.append("=" * 60)
            lines.append(f"[Similarity: {sim:.3f}]")
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
