"""
Built-in MCP Adapter: Simulates keyword-based memory tool with BM25 ranking
"""
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


class BuiltinMCPAdapter:
    """
    Simulates a built-in MCP memory tool with BM25-based search.
    Uses proper BM25 scoring with IDF weighting and length normalization.
    """
    def __init__(self, data_dir: Path, enable_filesystem: bool = False, k1: float = 1.2, b: float = 0.75):
        self.data_dir = data_dir
        self.env_dir = None
        self.enable_filesystem = enable_filesystem
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization

        # Computed on set_environment
        self.session_data = []
        self.session_tokens = []
        self.doc_freqs = Counter()
        self.avgdl = 0
        self.N = 0

    def set_environment(self, env_dir: Path):
        """Set the current question environment and compute IDF statistics"""
        self.env_dir = env_dir
        self._index_sessions()

    def _index_sessions(self):
        """Index all sessions and compute IDF statistics for BM25"""
        if not self.env_dir:
            return

        chat_history_dir = self.env_dir / "chat_history"
        session_files = sorted(chat_history_dir.glob("*.json"))

        self.session_data = []
        self.session_tokens = []
        self.doc_freqs = Counter()
        total_tokens = 0

        for session_file in session_files:
            with open(session_file) as f:
                data = json.load(f)
                self.session_data.append(data)

                # Tokenize session content
                text = self._session_to_text(data)
                tokens = self._tokenize(text)
                self.session_tokens.append(tokens)

                # Count document frequencies (unique terms per doc)
                unique_terms = set(tokens)
                for term in unique_terms:
                    self.doc_freqs[term] += 1

                total_tokens += len(tokens)

        self.N = len(session_files)
        self.avgdl = total_tokens / self.N if self.N > 0 else 1

    def _session_to_text(self, session_data: Dict) -> str:
        """Convert session to searchable text"""
        parts = []
        for turn in session_data.get("turns", []):
            content = turn.get("content", "")
            parts.append(content)
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenization: lowercase, split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _idf(self, term: str) -> float:
        """Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)"""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute BM25 score for a document given query"""
        doc_len = len(doc_tokens)
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI function calling tool definitions"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search conversation history using keywords. Returns top matching sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query with keywords to find relevant conversations"
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

        # Optionally add filesystem tools
        if self.enable_filesystem:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_session",
                    "description": "Read a specific session by index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_index": {
                                "type": "integer",
                                "description": "Index of the session (0 to num_sessions-1)"
                            }
                        },
                        "required": ["session_index"]
                    }
                }
            })

        return tools

    def execute_tool(self, function_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool call"""
        if function_name == "search_memory":
            top_k = args.get("top_k", 5)
            return self._search_memory(args["query"], top_k)
        elif function_name == "read_session" and self.enable_filesystem:
            return self._read_session(args["session_index"])
        else:
            return f"Unknown function: {function_name}"

    def _search_memory(self, query: str, top_k: int = 5) -> str:
        """
        BM25-based search with IDF weighting and length normalization
        """
        if not self.env_dir or len(self.session_tokens) == 0:
            return "Error: Environment not set"

        query_tokens = self._tokenize(query)

        # Score all sessions using BM25
        scored_sessions = []
        for idx, doc_tokens in enumerate(self.session_tokens):
            score = self._bm25_score(query_tokens, doc_tokens)
            if score > 0:
                scored_sessions.append((score, idx, self.session_data[idx]))

        if not scored_sessions:
            return f"No sessions found matching: {query}"

        # Sort by score (descending) and take top-k
        scored_sessions.sort(reverse=True, key=lambda x: x[0])
        top_sessions = scored_sessions[:top_k]

        # Format results
        lines = [f"Found {len(scored_sessions)} matching session(s). Showing top {len(top_sessions)}:", ""]
        for score, idx, session_data in top_sessions:
            lines.append("=" * 60)
            lines.append(f"[BM25 Score: {score:.2f}]")
            lines.append(self._format_session(session_data))

        return "\n".join(lines)

    def _read_session(self, session_index: int) -> str:
        """Read a specific session by index"""
        if not self.env_dir:
            return "Error: Environment not set"

        chat_history_dir = self.env_dir / "chat_history"
        session_files = sorted(chat_history_dir.glob("*.json"))

        if session_index < 0 or session_index >= len(session_files):
            return f"Error: Invalid session index. Valid range: 0-{len(session_files)-1}"

        with open(session_files[session_index]) as f:
            session_data = json.load(f)

        return self._format_session(session_data)

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
