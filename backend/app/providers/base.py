from typing import Any, Dict, List, Optional

class AIProvider:
    def propose_boundaries(
        self, paras_meta: List[Dict[str, Any]],
        target_tokens: int, min_tokens: int, max_tokens: int
    ) -> List[int]:
        raise NotImplementedError

    def propose_boundaries_context(
        self, paras_meta: List[Dict[str, Any]], full_text: str,
        target_tokens: int, min_tokens: int, max_tokens: int
    ) -> List[int]:
        return self.propose_boundaries(paras_meta, target_tokens, min_tokens, max_tokens)

    def propose_boundaries_context_full(
        self, paras_meta: List[Dict[str, Any]], full_text: str,
        target_tokens: int, min_tokens: int, max_tokens: int
    ) -> List[int]:
        return self.propose_boundaries_context(paras_meta, full_text, target_tokens, min_tokens, max_tokens)

    def title_from_text(self, text: str) -> str:
        words = " ".join((text or "").strip().split()[:10])
        return words or "Section"

    def generate_outline(self, corpus_text: str, num_lessons: int,
                         title: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_lesson_detail(self, corpus_text: str, lesson_title: str, num_questions: int) -> Dict[str, Any]:
        raise NotImplementedError

    def batch_embed_texts(self, inputs: List[str]) -> List[List[float]]:
        return [[] for _ in (inputs or [])]

    def embed_text(self, text: str) -> List[float]:
        return []

    def generate_chunk_metadata(self, chunks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        return []

    def group_chunks_semantic(self, items: List[Dict[str, Any]], desired_N: Optional[int] = None) -> Dict[str, Any]:
        return {"groups": []}
