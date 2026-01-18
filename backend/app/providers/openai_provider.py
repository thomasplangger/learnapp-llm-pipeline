                                  
from __future__ import annotations

import os
import re
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
import logging
from .base import AIProvider


class OpenAIProvider(AIProvider):
                                                        
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set and no api_key provided.")
        self.client = OpenAI(api_key=key)

                               
        for name in ("openai", "openai._base_client", "httpx"):
            try:
                logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass

                                                                     
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                                                          
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.default_para_trim = int(os.getenv("CHUNK_PARA_TRIM", "400"))                                             
        self.default_window_max_paras = int(os.getenv("CHUNK_MAX_PARAS_PER_WINDOW", "500"))

                                                               
        self._dbg_on = os.getenv("CHUNK_DEBUG", "0") == "1"
        self._dbg_root = Path(os.getenv("CHUNK_DEBUG_DIR", "debug"))
        self._dbg_run = None
        self._dbg_call_idx = 0

                                                             
    MAX_MODEL_TOKENS = 128_000
    SAFE_BUDGET = 110_000                                             

                                                              
    def _dbg_dir(self) -> Optional[Path]:
        """Create and return the per-run debug directory (or None if disabled)."""
        if not self._dbg_on:
            return None
        if self._dbg_run is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._dbg_run = f"{stamp}"
        d = self._dbg_root / self._dbg_run
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _dbg_write(self, relpath: str, obj: Any) -> None:
        """Write JSON or text to debug dir if enabled."""
        if not self._dbg_on:
            return
        base = self._dbg_dir()
        if not base:
            return
        p = base / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            if isinstance(obj, (dict, list)):
                p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                p.write_text(str(obj), encoding="utf-8")
        except Exception as e:
                                                            
            pass

    def _llm_json(
        self,
        system: str,
        user_obj: dict,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 900,
        label: str | None = None,
    ) -> dict:
                                                       
        self._dbg_call_idx += 1
        ts = datetime.now().strftime("%H-%M")
        safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", label or "llm_call")
        call_dir = f"{safe_label}_{ts}_{self._dbg_call_idx:03d}"
        self._dbg_write(f"{call_dir}/request.messages.json", [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ])
        self._dbg_write(f"{call_dir}/request.meta.json", {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
        )

                                                           
        content = resp.choices[0].message.content or "{}"
                                
        try:
            raw_dump = resp.model_dump()                                  
        except Exception:
            raw_dump = {"id": getattr(resp, "id", None), "choices": [{"message": {"content": content}}]}
        self._dbg_write(f"{call_dir}/response.raw.json", raw_dump)
        self._dbg_write(f"{call_dir}/response.text.txt", content)

                      
        try:
            parsed = json.loads(content)
        except Exception as ex:
            b, e = content.find("{"), content.rfind("}")
            if b != -1 and e != -1 and e > b:
                try:
                    parsed = json.loads(content[b:e+1])
                except Exception as ex2:
                    parsed = {}
                    self._dbg_write(f"{call_dir}/parse_error.txt", f"substring parse failed: {ex2}")
            else:
                parsed = {}
                self._dbg_write(f"{call_dir}/parse_error.txt", f"json.loads failed: {ex}")
        self._dbg_write(f"{call_dir}/response.parsed.json", parsed)

        return parsed

                                                             
    def propose_boundaries_with_metadata(
        self,
        paragraphs: List[Dict[str, Any]],
        desired_chunks: Optional[int] = None,
        strict_count: bool = False,
        mods: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to return both boundaries and metadata for each chunk.
        Compatible with old boundary-only expectations.
        """
        mods = mods or {}
        M = len(paragraphs)
        if M == 0:
            return {"boundaries": [0], "metadata": []}

        if mods.get("prechunking") and not mods.get("_skip_prechunk"):
            blocks = self._prechunk_blocks(paragraphs, mods)
            if len(blocks) > 1:
                merged = self._segment_blocks(paragraphs, blocks, mods, desired_chunks, strict_count)
                if merged:
                    return merged

        hierarchical_override = False
        obj: Optional[Dict[str, Any]] = None
        if mods.get("hierarchical") and not mods.get("_skip_hierarchical"):
            try:
                obj = self._hierarchical_segment(paragraphs, mods)
                hierarchical_override = True
            except Exception:
                obj = None

        similarity_threshold = None
        try:
            if mods.get("similarity_threshold") is not None:
                similarity_threshold = float(mods["similarity_threshold"])
                if not (0.0 < similarity_threshold < 1.0):
                    similarity_threshold = None
        except Exception:
            similarity_threshold = None

        heading_boost = None
        try:
            hs = mods.get("heading_strength") if mods else None
            if hs is not None:
                heading_boost = max(1.0, float(hs))
        except Exception:
            heading_boost = None

        items = []
        for i, p in enumerate(paragraphs):
            level = (p.get("level") or "P").upper()
            txt = (p.get("text") or "")[:800]
            entry = {
                "i": i,
                "level": level,
                "text": txt,
            }
            if heading_boost and level in {"H1", "H2", "H3"}:
                entry["heading_weight"] = heading_boost
                entry["text"] = f"[HEADING x{heading_boost:.2f}] {txt}"
            items.append(entry)

        targets = self._estimate_chunk_targets(M)
        min_paras = max(1, targets.get("min_paras", 1))
        max_paras = max(min_paras + 1, targets.get("max_paras", min_paras + 4))
        if mods:
            if mods.get("merge_small"):
                try:
                    mod_min = int(mods["merge_small"])
                    if mod_min > 0:
                        min_paras = max(1, mod_min)
                except Exception:
                    pass
            if mods.get("split_large"):
                try:
                    mod_max = int(mods["split_large"])
                    if mod_max > 0:
                        max_paras = max(min_paras + 1, mod_max)
                except Exception:
                    pass
        approx_chunks = max(1, targets.get("approx_chunks", 1))

        if not hierarchical_override:
            system = (
                "You are a precise document segmenter.\n"
                "You receive JSON describing paragraphs indexed 0..M-1, with heading level (H1/H2/H3/P) plus a text snippet.\n"
                "Your task is to output STRICT JSON with keys 'boundaries' and 'metadata' that segments the document cleanly.\n"
                "- 'boundaries' is an array of paragraph indices. ALWAYS include 0 and M.\n"
                f"- Favor chunk sizes between {min_paras} and {max_paras} paragraphs and avoid 1-2 paragraph chunks unless the content is standalone.\n"
                f"- Keep the total number of chunks near {approx_chunks} without going dramatically above or below it.\n"
                "- Prefer to start chunks at major headings (H1/H2, H3 if it clearly signals a new topic).\n"
                "- Do NOT create a new chunk for every minor bullet; only split when the topic meaningfully shifts.\n"
                "- Consider any 'suggested_breaks' indices (based on low cosine similarity between paragraphs) as likely boundaries, unless context clearly indicates otherwise.\n"
                "- Some paragraph entries may include 'heading_weight' > 1 and a '[HEADING x..]' prefix, indicating boosted headings that should be favored as boundaries when reasonable.\n"
                "- 'metadata' describes each chunk between consecutive boundaries and MUST contain chunk_index, title (<=90 chars), topic (2-5 words), summary (2-6 sentences), and 5-12 keywords.\n"
                "- Respond with valid JSON only, no explanations."
            )

            suggested_breaks: List[int] = []
            if similarity_threshold:
                suggested_breaks = self._similarity_breaks(paragraphs, similarity_threshold)

            mods_payload = {k: v for k, v in mods.items() if not str(k).startswith("_")}
            user = {
                "M": M,
                "MIN_P": min_paras,
                "MAX_P": max_paras,
                "APPROX_CHUNKS": approx_chunks,
                "desired_chunks": desired_chunks,
                "strict_count": bool(strict_count),
                "paragraphs": items,
                "chunk_targets": targets,
                "mods": mods_payload or None,
                "similarity_threshold": similarity_threshold,
                "suggested_breaks": suggested_breaks or None,
            }

            obj = self._llm_json(system, user, model=self.model, temperature=0.0, max_tokens=12000, label="llm_call_boundaries_meta")

                                                              
        raw_bounds = obj.get("boundaries") or []
        if not hierarchical_override and similarity_threshold:
            suggested_breaks = self._similarity_breaks(paragraphs, similarity_threshold) if similarity_threshold else []
            if suggested_breaks:
                raw_bounds = list(raw_bounds or [])
                raw_bounds.extend(suggested_breaks)
        if not isinstance(raw_bounds, list):
                                                                        
            raise ValueError("LLM returned invalid boundaries JSON (empty or wrong type)")
        base_bounds = self._normalize_boundaries(raw_bounds, M)

                                       
        metas = obj.get("metadata") or []
        clean_meta = []
        for i in range(len(base_bounds) - 1):
            md = metas[i] if i < len(metas) else {}
            title = (md.get("title") or "").strip()[:90]
            topic = (md.get("topic") or "").strip()
            summary = (md.get("summary") or "").strip()
            keywords = [str(k).strip() for k in md.get("keywords", []) if str(k).strip()]
            clean_meta.append({
                "chunk_index": i,
                "title": title or None,
                "topic": topic or None,
                "summary": summary or None,
                "keywords": keywords[:12],
            })

        bounds = self._postprocess_boundaries(base_bounds, paragraphs, min_paras, max_paras)
        clean_meta = self._reflow_metadata(base_bounds, bounds, clean_meta)

                                                                            
        embed_inputs: List[str] = []
        for m in clean_meta:
            embed_inputs.append("\n".join(filter(None, [
                m.get("title"), m.get("summary"), ", ".join(m.get("keywords", []))
            ])))

        vectors: List[List[float]] = []
        try:
            vectors = self.batch_embed_texts(embed_inputs)
        except Exception:
                                
            vectors = [self.embed_text(x) for x in embed_inputs]

        for m, vec in zip(clean_meta, vectors):
            m["embedding"] = vec or []
                                         
            conf = 0.6
            if m.get("summary") and len(m.get("keywords", [])) >= 5:
                conf += 0.2
            if m.get("title"):
                conf += 0.1
            m["confidence"] = min(conf, 0.95)

        return {"boundaries": bounds, "metadata": clean_meta}

    @staticmethod
    def _estimate_chunk_targets(num_paragraphs: int) -> Dict[str, int]:
        """Estimate chunk sizing heuristics from the document itself."""
        if num_paragraphs <= 0:
            return {"approx_chunks": 0, "min_paras": 1, "max_paras": 1, "avg_paras": 1}
        if num_paragraphs <= 8:
            avg = max(2, num_paragraphs)
            min_paras = 1
            max_paras = max(2, num_paragraphs)
        elif num_paragraphs <= 25:
            avg = 6
            min_paras = 3
            max_paras = 10
        elif num_paragraphs <= 60:
            avg = 10
            min_paras = 4
            max_paras = 15
        elif num_paragraphs <= 120:
            avg = 12
            min_paras = 5
            max_paras = 18
        elif num_paragraphs <= 240:
            avg = 14
            min_paras = 6
            max_paras = 20
        else:
            avg = 30
            min_paras = 20
            max_paras = 40
        approx_chunks = max(1, round(num_paragraphs / avg))
        return {
            "approx_chunks": approx_chunks,
            "min_paras": min_paras,
            "max_paras": max_paras,
            "avg_paras": avg,
        }

    @staticmethod
    def _normalize_boundaries(boundaries: List[Any], total: int) -> List[int]:
        """Keep boundaries sorted, unique, and clipped to [0, total]."""
        vals: List[int] = []
        for b in boundaries:
            if isinstance(b, (int, float)):
                vals.append(int(b))
        vals.extend([0, total])
        clipped = sorted(set(max(0, min(total, v)) for v in vals))
        if clipped[0] != 0:
            clipped.insert(0, 0)
        if clipped[-1] != total:
            clipped.append(total)
        return clipped

    def _postprocess_boundaries(
        self,
        boundaries: List[int],
        paragraphs: List[Dict[str, Any]],
        min_paras: int,
        max_paras: int,
    ) -> List[int]:
        """Snap to headings, merge tiny chunks, and optionally split large ones."""
        total = len(paragraphs)
        processed = self._normalize_boundaries(boundaries, total)
        processed = self._snap_boundaries_to_headings(processed, paragraphs, window=2)
        processed = self._normalize_boundaries(processed, total)
        processed = self._merge_small_chunks(processed, min_paras)
        processed = self._split_large_chunks(processed, paragraphs, max_paras)
        processed = self._merge_small_chunks(processed, min_paras)
        return processed

    def _snap_boundaries_to_headings(
        self,
        boundaries: List[int],
        paragraphs: List[Dict[str, Any]],
        window: int = 2,
    ) -> List[int]:
        if not boundaries or not paragraphs:
            return boundaries
        headings = {
            idx: str(paragraphs[idx].get("level") or "P").upper()
            for idx in range(len(paragraphs))
            if str(paragraphs[idx].get("level") or "P").upper() in {"H1", "H2", "H3"}
        }
        if not headings:
            return boundaries
        snapped = list(boundaries)
        total = len(paragraphs)
        for i in range(1, len(snapped) - 1):
            original = snapped[i]
            candidates = [original]
            for offset in range(1, window + 1):
                candidates.extend([original - offset, original + offset])
            best = None
            best_score = None
            for cand in candidates:
                if cand <= 0 or cand >= total:
                    continue
                if cand not in headings:
                    continue
                score = (abs(cand - original), self._heading_priority(headings[cand]))
                if best is None or score < best_score:
                    best = cand
                    best_score = score
            if best is not None:
                snapped[i] = best
        return snapped

    @staticmethod
    def _merge_small_chunks(boundaries: List[int], min_size: int) -> List[int]:
        if min_size <= 1 or len(boundaries) <= 2:
            return boundaries
        bounds = list(boundaries)
        changed = True
        while changed and len(bounds) > 2:
            changed = False
            for idx in range(len(bounds) - 1):
                size = bounds[idx + 1] - bounds[idx]
                if size >= min_size or size <= 0:
                    continue
                if idx == 0:
                    remove_idx = 1
                elif idx == len(bounds) - 2:
                    remove_idx = idx
                else:
                    prev_size = bounds[idx] - bounds[idx - 1]
                    next_size = bounds[idx + 2] - bounds[idx + 1]
                    remove_idx = idx if prev_size <= next_size else idx + 1
                remove_idx = max(1, min(len(bounds) - 1, remove_idx))
                if remove_idx in (0, len(bounds) - 1):
                    continue
                bounds.pop(remove_idx)
                changed = True
                break
        return bounds

    def _split_large_chunks(
        self,
        boundaries: List[int],
        paragraphs: List[Dict[str, Any]],
        max_size: int,
    ) -> List[int]:
        if max_size <= 0 or len(boundaries) <= 2:
            return boundaries
        bounds = list(boundaries)
        heading_indices = [
            idx for idx, para in enumerate(paragraphs)
            if str(para.get("level") or "P").upper() in {"H1", "H2", "H3"}
        ]
        if not heading_indices:
            return bounds
        changed = True
        while changed:
            changed = False
            for i in range(len(bounds) - 1):
                size = bounds[i + 1] - bounds[i]
                if size <= max_size * 1.4 and size <= max_size + 4:
                    continue
                candidates = [h for h in heading_indices if bounds[i] < h < bounds[i + 1]]
                if not candidates:
                    continue
                mid = bounds[i] + size // 2
                best = min(
                    candidates,
                    key=lambda idx: (abs(idx - mid), self._heading_priority(str(paragraphs[idx].get("level") or "P").upper()))
                )
                if best in bounds or best <= bounds[i] or best >= bounds[i + 1]:
                    continue
                bounds.insert(i + 1, best)
                changed = True
                break
        return bounds

    @staticmethod
    def _heading_priority(level: str) -> int:
        order = {"H1": 0, "H2": 1, "H3": 2}
        return order.get(level, 3)

    def _reflow_metadata(
        self,
        old_bounds: List[int],
        new_bounds: List[int],
        metadata: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        spans_old = list(zip(old_bounds[:-1], old_bounds[1:]))
        spans_new = list(zip(new_bounds[:-1], new_bounds[1:]))
        out: List[Dict[str, Any]] = []
        for idx, (start, end) in enumerate(spans_new):
            overlaps: List[Dict[str, Any]] = []
            for j, (o_start, o_end) in enumerate(spans_old):
                if o_end <= start or o_start >= end:
                    continue
                if j < len(metadata):
                    overlaps.append(metadata[j])
            combined = self._combine_metadata_entries(overlaps)
            combined["chunk_index"] = idx
            out.append(combined)
        return out

    @staticmethod
    def _combine_metadata_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not entries:
            return {
                "chunk_index": 0,
                "title": None,
                "topic": None,
                "summary": None,
                "keywords": [],
            }
        if len(entries) == 1:
            single = dict(entries[0])
            single["keywords"] = list(entries[0].get("keywords", []))
            return single
        titles = [e.get("title") for e in entries if e.get("title")]
        topics = [e.get("topic") for e in entries if e.get("topic")]
        summaries = [e.get("summary") for e in entries if e.get("summary")]
        keywords: List[str] = []
        for entry in entries:
            for kw in entry.get("keywords", []):
                if kw and kw not in keywords:
                    keywords.append(kw)
        return {
            "chunk_index": 0,
            "title": " / ".join(dict.fromkeys(titles))[:90] if titles else None,
            "topic": topics[0] if topics else None,
            "summary": " ".join(summaries) if summaries else None,
            "keywords": keywords[:12],
        }

    def _hierarchical_segment(
        self,
        paragraphs: List[Dict[str, Any]],
        mods: Dict[str, Any],
    ) -> Dict[str, Any]:
        blocks = self._hierarchical_blocks(paragraphs)
        if len(blocks) <= 1:
            raise ValueError("Not enough hierarchical structure")
        combined_bounds: List[int] = [0]
        combined_meta: List[Dict[str, Any]] = []
        for start, end in blocks:
            block = paragraphs[start:end]
            if not block:
                continue
            block_mods = dict(mods or {})
            block_mods.pop("hierarchical", None)
            block_mods["_skip_hierarchical"] = True
            sub = self.propose_boundaries_with_metadata(
                block,
                desired_chunks=None,
                strict_count=False,
                mods=block_mods,
            )
            sub_bounds = sub.get("boundaries") or []
            if len(sub_bounds) < 2:
                continue
            for b in sub_bounds[1:]:
                shifted = start + int(b)
                if shifted <= combined_bounds[-1]:
                    continue
                combined_bounds.append(shifted)
            combined_meta.extend(sub.get("metadata") or [])
        if combined_bounds[-1] != len(paragraphs):
            combined_bounds.append(len(paragraphs))
        for idx, meta in enumerate(combined_meta):
            meta["chunk_index"] = idx
        return {"boundaries": combined_bounds, "metadata": combined_meta}

    def _hierarchical_blocks(self, paragraphs: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        primary = self._heading_sections(paragraphs, "H1")
        if not primary:
            primary = [(0, len(paragraphs))]
        blocks: List[Tuple[int, int]] = []
        for start, end in primary:
            local = self._heading_sections(paragraphs[start:end], "H2")
            if len(local) == 1 and local[0] == (0, end - start):
                blocks.append((start, end))
            else:
                for ls, le in local:
                    if ls == le:
                        continue
                    blocks.append((start + ls, start + le))
        if not blocks:
            blocks = [(0, len(paragraphs))]
        return blocks

    def _heading_sections(self, paragraphs: List[Dict[str, Any]], level: str) -> List[Tuple[int, int]]:
        level = (level or "P").upper()
        anchors = sorted(
            set(
                int(i)
                for i, p in enumerate(paragraphs)
                if (p.get("level") or "P").upper() == level
            )
        )
        if not anchors:
            return [(0, len(paragraphs))]
        starts: List[int] = []
        if anchors[0] != 0:
            starts.append(0)
        starts.extend(anchors)
        sections: List[Tuple[int, int]] = []
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(paragraphs)
            if start < end:
                sections.append((start, end))
        if not sections:
            sections = [(0, len(paragraphs))]
        return sections

    def _segment_blocks(
        self,
        paragraphs: List[Dict[str, Any]],
        blocks: List[Tuple[int, int]],
        mods: Dict[str, Any],
        desired_chunks: Optional[int],
        strict_count: bool,
    ) -> Optional[Dict[str, Any]]:
        if not blocks:
            return None
        merged_bounds: List[int] = [0]
        merged_meta: List[Dict[str, Any]] = []
        total_len = len(paragraphs)
        for start, end in blocks:
            if end <= start:
                continue
            block_mods = dict(mods or {})
            block_mods["_skip_prechunk"] = True
            sub = self.propose_boundaries_with_metadata(
                paragraphs[start:end],
                desired_chunks=desired_chunks,
                strict_count=strict_count,
                mods=block_mods,
            )
            sub_bounds = sub.get("boundaries") or []
            if not isinstance(sub_bounds, list) or len(sub_bounds) < 2:
                continue
            for b in sub_bounds[1:]:
                shifted = start + int(b)
                if 0 < shifted < total_len:
                    merged_bounds.append(shifted)
            for meta in sub.get("metadata") or []:
                merged_meta.append(dict(meta))
        merged_bounds = sorted(set(max(0, min(total_len, int(b))) for b in merged_bounds))
        if not merged_bounds or merged_bounds[0] != 0:
            merged_bounds.insert(0, 0)
        if merged_bounds[-1] != total_len:
            merged_bounds.append(total_len)
        for idx, meta in enumerate(merged_meta):
            meta["chunk_index"] = idx
        return {"boundaries": merged_bounds, "metadata": merged_meta}

    def _prechunk_blocks(self, paragraphs: List[Dict[str, Any]], mods: Dict[str, Any]) -> List[Tuple[int, int]]:
        total = len(paragraphs)
        if total == 0:
            return [(0, 0)]
        try:
            min_pages = int(mods.get("prechunking_min_pages") or 30)
        except Exception:
            min_pages = 30
        paras_per_page = max(4, int(os.getenv("PRECHUNK_PARAS_PER_PAGE", "6")))
        threshold = max(80, min_pages * paras_per_page)

        blocks: List[Tuple[int, int]] = []

        def split_block(start: int, end: int) -> None:
            length = end - start
            if length <= threshold:
                blocks.append((start, end))
                return
            split = self._find_prechunk_split(paragraphs, start, end)
            if split <= start or split >= end:
                split = max(start + 1, min(end - 1, (start + end) // 2))
            split_block(start, split)
            split_block(split, end)

        split_block(0, total)
        blocks = sorted((s, e) for s, e in blocks if e > s)
        merged: List[Tuple[int, int]] = []
        last_start, last_end = None, None
        for s, e in blocks:
            if last_start is None:
                last_start, last_end = s, e
                continue
            if s == last_end:
                last_end = e
            else:
                merged.append((last_start, last_end))
                last_start, last_end = s, e
        if last_start is not None:
            merged.append((last_start, last_end))
        return merged or [(0, total)]

    def _find_prechunk_split(self, paragraphs: List[Dict[str, Any]], start: int, end: int) -> int:
        mid = (start + end) // 2
        span = max(5, (end - start) // 10)
        for level in ("H1", "H2"):
            idx = self._nearest_heading(paragraphs, mid, start, end, level, span)
            if idx is not None:
                return idx
        return max(start + 1, min(end - 1, mid))

    def _nearest_heading(
        self,
        paragraphs: List[Dict[str, Any]],
        target: int,
        start: int,
        end: int,
        level: str,
        window: int,
    ) -> Optional[int]:
        best_idx: Optional[int] = None
        best_delta: Optional[int] = None
        level = (level or "P").upper()
        for idx in range(max(start, target - window), min(end, target + window) + 1):
            lvl = (paragraphs[idx].get("level") or "P").upper()
            if lvl != level:
                continue
            delta = abs(idx - target)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        return best_idx

    def _similarity_breaks(self, paragraphs: List[Dict[str, Any]], threshold: float) -> List[int]:
        if not paragraphs or not (0.0 < threshold < 1.0):
            return []
        texts = []
        for p in paragraphs:
            txt = (p.get("text") or "").strip()
            if not txt:
                txt = " "
            texts.append(txt[:512])
        try:
            vectors = self.batch_embed_texts(texts)
        except Exception:
            vectors = [[] for _ in texts]
        breaks: List[int] = []
        prev_vec = None
        prev_norm = None
        for idx, vec in enumerate(vectors):
            if not vec:
                prev_vec = None
                prev_norm = None
                continue
            norm = math.sqrt(sum((float(v) ** 2) for v in vec)) or 1.0
            if prev_vec is not None and prev_norm:
                dot = sum(float(a) * float(b) for a, b in zip(prev_vec, vec))
                cos = dot / (prev_norm * norm)
                if cos < threshold:
                    breaks.append(idx)
            prev_vec = vec
            prev_norm = norm
        return sorted(set(int(b) for b in breaks if 0 < b < len(paragraphs)))

                                                                     
    def embed_text(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text using the configured
        embedding model. Returns an empty list on error to avoid breaking flows.
        """
        try:
            if not (text or "").strip():
                return []
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text],
            )
            vec = resp.data[0].embedding if getattr(resp, "data", None) else []
                                                                    
            return list(vec) if isinstance(vec, (list, tuple)) else []
        except Exception:
            return []

    def batch_embed_texts(self, inputs: List[str]) -> List[List[float]]:
        """Efficiently embed many texts in a small number of API calls."""
        if not inputs:
            return []
        out: List[List[float]] = []
                                                                  
        try:
            batch_sz = int(os.getenv("EMBED_BATCH", "64"))
        except Exception:
            batch_sz = 64
        for i in range(0, len(inputs), max(1, batch_sz)):
            chunk = inputs[i:i+batch_sz]
            try:
                resp = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=chunk,
                )
                vecs = [list(d.embedding) for d in getattr(resp, "data", [])]
            except Exception:
                                                   
                vecs = [self.embed_text(t) for t in chunk]
                                                          
            if len(vecs) != len(chunk):
                vecs = vecs + [[] for _ in range(len(chunk) - len(vecs))]
            out.extend(vecs)
        return out

                                                                         
    def choose_lo_count(self, topics: List[str], default_min: int = 3, default_max: int = 10) -> int:
        """Suggest an LO count from a list of chunk topics/keywords using the LLM.
        Falls back to sqrt heuristic.
        """
        try:
            m = max(default_min, min(default_max, int(math.sqrt(max(1, len(topics))))))
            system = (
                "You suggest how many learning objectives to create for a course.\n"
                "Return JSON: {\"n\": int}.\n"
                "Rules: choose between 3 and 10, balancing coverage and cohesion."
            )
            user = {"topics": topics[:200], "suggested": m}
            resp = self._llm_json(system, user, model=self.model, temperature=0.0, max_tokens=20, label="llm_call_lo_count")
            n = int(resp.get("n", m))
            return max(default_min, min(default_max, n))
        except Exception:
            return max(default_min, min(default_max, int(math.sqrt(max(1, len(topics))))))

    def summarize_learning_objective(self, items: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create a concise LO summary from representative chunk items.
        items: list of {'title','summary','keywords'} strings.
        Returns {title, summary, objectives, keywords}.
        """
        try:
            system = (
                "You condense a group of course chunks into one Learning Objective.\n"
                "Return JSON only with keys: title, summary, objectives, keywords.\n"
                "- title: <= 90 chars\n- summary: 3-5 sentences\n- objectives: 3-5 bullet phrases (imperative)\n- keywords: 5-12 tags"
            )
            user = {"items": items[:12]}
            obj = self._llm_json(system, user, model=self.model, temperature=0.2, max_tokens=500, label="llm_call_lo_summarize")
            title = (obj.get("title") or "").strip()[:90]
            objectives = obj.get("objectives") or []
            objectives = [str(o).strip() for o in objectives if str(o).strip()][:5]
            keywords = obj.get("keywords") or []
            keywords = [str(k).strip() for k in keywords if str(k).strip()][:12]
            return {
                "title": title or None,
                "summary": (obj.get("summary") or "").strip(),
                "objectives": objectives,
                "keywords": keywords,
            }
        except Exception:
                              
            title = items[0].get("title")[:90] if items else "Learning Objective"
            return {"title": title or "Learning Objective", "summary": "", "objectives": [], "keywords": []}

    def group_chunks_semantic(self, items: List[Dict[str, Any]], desired_N: Optional[int] = None) -> Dict[str, Any]:
        """
        Group chunks into learning objectives based on topical fit using the LLM.
        items: list of {index, title, topic, keywords, summary}
        Returns {groups: [{title, summary, keywords, members:[indices]}]}
        """
        try:
                                            
            comp: List[Dict[str, Any]] = []
            for it in items[:160]:
                comp.append({
                    "i": int(it.get("index", len(comp))),
                    "title": (it.get("title") or "")[:90],
                    "topic": (it.get("topic") or "")[:60],
                    "keywords": [str(k)[:24] for k in (it.get("keywords") or [])][:10],
                    "summary": (it.get("summary") or "").split("\n")[0][:220],
                })

            want = int(desired_N) if desired_N else None
            all_indices = [it["i"] for it in comp]
            system = (
                "You group course chunks into coherent Learning Objectives based on topic similarity.\n"
                "Return STRICT JSON with key 'groups' as an array. Each group has: title, summary, keywords (5-12), members (chunk indices).\n"
                "Rules:\n- Use members as the provided indices (i).\n"
                "- Assign EVERY chunk index exactly once (no omissions, no duplicates).\n"
                "- Do not duplicate a chunk across groups.\n- Avoid groups of size 1 unless necessary.\n"
                "- Titles <= 90 chars; concise summary (2-5 sentences).\n"
                "- You MUST use all indices in this list: {all_indices}.\n"
                "- No explanations outside JSON."
            )
            if want is not None:
                user = {"N": want, "chunks": comp, "instruction": f"Group into exactly {want} objectives."}
            else:
                user = {"chunks": comp, "instruction": "Choose a sensible number of objectives (between 3 and 10)."}

            obj = self._llm_json(system, user, model=self.model, temperature=0.2, max_tokens=12000, label="llm_call_los_group")
            groups = obj.get("groups") or []
                      
            clean = []
            used = set()
            for g in groups:
                members = g.get("members") or []
                mem = []
                for x in members:
                    try:
                        xi = int(x)
                    except Exception:
                        continue
                    if 0 <= xi < len(comp) and xi not in used:
                        used.add(xi)
                        mem.append(xi)
                if not mem:
                    continue
                title = (g.get("title") or "").strip()[:90]
                summary = (g.get("summary") or "").strip()
                keywords = [str(k).strip() for k in (g.get("keywords") or []) if str(k).strip()][:12]
                clean.append({
                    "title": title or None,
                    "summary": summary,
                    "keywords": keywords,
                    "members": mem,
                })
                                                     
            missing = [i for i in all_indices if i not in used]
            if missing and clean:
                groups_comp = []
                for gi, g in enumerate(clean[:12]):
                    groups_comp.append({
                        "gi": gi,
                        "title": g.get("title") or "",
                        "summary": (g.get("summary") or "")[:200],
                        "keywords": (g.get("keywords") or [])[:8],
                    })
                miss_items = [comp[i] for i in missing]
                system2 = (
                    "Assign each unassigned chunk to one of the existing groups.\n"
                    "Return STRICT JSON with key 'assignments' as an array of objects: {i, group}.\n"
                    "Rules:\n- Use the given chunk indices.\n- Use group IDs from the provided list.\n- Do not omit any chunk.\n- No explanations."
                )
                user2 = {"groups": groups_comp, "chunks": miss_items}
                try:
                    obj2 = self._llm_json(system2, user2, model=self.model, temperature=0.0, max_tokens=2000, label="llm_call_los_group_fill")
                    assigns = obj2.get("assignments") or []
                    for a in assigns:
                        try:
                            ci = int(a.get("i"))
                            gi = int(a.get("group"))
                        except Exception:
                            continue
                        if ci in missing and 0 <= gi < len(clean):
                            clean[gi]["members"].append(ci)
                            used.add(ci)
                                                              
                    still = [i for i in missing if i not in used]
                    if still:
                        smallest = min(range(len(clean)), key=lambda k: len(clean[k]["members"]))
                        clean[smallest]["members"].extend(still)
                except Exception:
                    pass
            return {"groups": clean}
        except Exception:
            return {"groups": []}

    def generate_chunk_metadata(self, chunks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Generate metadata for a list of chunks using one or a few LLM calls (batched).
        chunks: list of {index:int, text:str} (text should be short preview).
        Returns a list aligned by index: [{chunk_index, title, topic, summary, keywords, confidence, embedding}].
        """
        if not chunks:
            return []

                                              
        try:
            meta_batch = int(os.getenv("METADATA_BATCH", "24"))
        except Exception:
            meta_batch = 24

        cleaned_total: List[Dict[str, Any]] = []
        for start in range(0, len(chunks), max(1, meta_batch)):
            part = chunks[start:start+meta_batch]
            items = []
            for k, c in enumerate(part):
                items.append({
                    "i": int(c.get("index", start + k)),
                    "text": (c.get("text") or "")[:800],
                })
            system = (
                "For each provided chunk preview, return metadata.\n"
                "Respond JSON only with key 'metadata' as an array (same order),\n"
                "each having: chunk_index, title (<=90), topic (2-5 words), summary (2-6 sentences), keywords (5-12)."
            )
            user = {"chunks": items}
            obj = self._llm_json(system, user, model=self.model, temperature=0.2, max_tokens=8000)
            metas = obj.get("metadata") or []

            partial_clean: List[Dict[str, Any]] = []
            for i in range(len(items)):
                md = metas[i] if i < len(metas) else {}
                title = (md.get("title") or "").strip()[:90]
                topic = (md.get("topic") or "").strip()
                summary = (md.get("summary") or "").strip()
                keywords = [str(k).strip() for k in md.get("keywords", []) if str(k).strip()][:12]
                m = {
                    "chunk_index": start + i,
                    "title": title or None,
                    "topic": topic or None,
                    "summary": summary or None,
                    "keywords": keywords,
                }
                partial_clean.append(m)

                                    
            embed_inputs = ["\n".join(filter(None, [m.get("title"), m.get("summary"), ", ".join(m.get("keywords", []))])) for m in partial_clean]
            try:
                vecs = self.batch_embed_texts(embed_inputs)
            except Exception:
                vecs = [self.embed_text(x) for x in embed_inputs]
            for m, v in zip(partial_clean, vecs):
                m["embedding"] = v or []
                conf = 0.6
                if m.get("summary") and len(m.get("keywords", [])) >= 5:
                    conf += 0.2
                if m.get("title"):
                    conf += 0.1
                m["confidence"] = min(conf, 0.95)

            cleaned_total.extend(partial_clean)

                                                                                   
        if len(cleaned_total) < len(chunks):
            cleaned_total.extend({"chunk_index": i} for i in range(len(cleaned_total), len(chunks)))

        return cleaned_total
