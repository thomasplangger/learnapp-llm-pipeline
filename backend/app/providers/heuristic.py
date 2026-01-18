# backend/app/providers/heuristic.py
import re
from typing import Any, Dict, List, Optional
from .base import AIProvider

def estimate_tokens_from_chars(n: int) -> int:
    return max(1, n // 4)

_STOP = set("""a an and are as at be by for from has have in into is it of on or that the to with was were will would can could should may might this those these there here such which when while where who whom whose how why not than then""".split())

_sentence_re = re.compile(r"(?<=[.!?])\s+")
_blank_block_re = re.compile(r"\n\s*\n+")
_heading_re = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+|[A-Z][A-Z0-9 _-]{3,}|[A-Z][^\n]{0,60}$)")

def _split_page_blocks(text: str) -> List[tuple[int,int]]:
    spans = []
    last = 0
    for m in re.finditer(r"\f", text):
        if m.start() > last:
            spans.append((last, m.start()))
        last = m.end()
    if last < len(text):
        spans.append((last, len(text)))
    return spans if spans else [(0, len(text))]

def _split_blank_blocks(text: str, s: int, e: int) -> List[tuple[int,int]]:
    seg = text[s:e]
    spans, last = [], 0
    for m in _blank_block_re.finditer(seg):
        if m.start() > last:
            spans.append((s + last, s + m.start()))
        last = m.end()
    if s + last < e:
        spans.append((s + last, e))
    return spans if spans else [(s, e)]

def _split_long_by_sentences(text: str, s: int, e: int, max_chars: int = 1800) -> List[tuple[int,int]]:
    seg = text[s:e]
    if len(seg) <= max_chars:
        return [(s, e)]
    parts = _sentence_re.split(seg)
    spans = []
    acc = ""
    start_off = 0
    for piece in parts:
        acc = (acc + " " + piece).strip() if acc else piece
        if len(acc) >= max_chars:
            end_pos = start_off + len(acc)
            spans.append((s + start_off, s + end_pos))
            start_off = end_pos + 1
            acc = ""
    if acc:
        spans.append((s + start_off, s + start_off + len(acc)))
    cleaned, last = [], s
    for (a,b) in spans:
        if a < last: a = last
        if b <= a: continue
        cleaned.append((a,b))
        last = b
    return cleaned or [(s,e)]

def split_into_paragraphs_with_spans(text: str, max_para_chars: int = 1800) -> List[Dict[str, Any]]:
    if not text:
        return []
    page_spans = _split_page_blocks(text)
    blocks = []
    for (ps, pe) in page_spans:
        blocks.extend(_split_blank_blocks(text, ps, pe))
    refined = []
    for (bs, be) in blocks:
        refined.extend(_split_long_by_sentences(text, bs, be, max_chars=max_para_chars))

    # simple list bullet detector (works for "1. ", "- ", "* ", "• ")
    _list_bullet = re.compile(r"^\s*(?:[-*•]\s+|\d+\.\s+)")
    _all_caps = re.compile(r'^[^a-z]{3,}$')
    _numbered_header = re.compile(r'^\s*(?:\d+(?:\.\d+)*[.)]?\s+)')

    meta = []
    for i, (s, e) in enumerate(refined):
        para = text[s:e]
        lines = para.splitlines()
        first_line = (lines[0] if lines else para)[:200]
        preview = para.strip().replace("\n", " ")[:200]

        is_list = bool(_list_bullet.match(first_line.strip()))
        heading_like = bool(_heading_re.match(first_line.strip()))
        # light level: H2 for obvious headers, else P. (We keep it light; the LLM will refine.)
        if heading_like and (_all_caps.match(first_line.strip()) or _numbered_header.match(first_line.strip())):
            level = "H2"
        elif heading_like:
            level = "H3"
        else:
            level = "P"

        meta.append({
            "index": i, "start": s, "end": e, "length": e - s,
            "first_line": first_line.strip()[:120],
            "preview": preview,
            "heading_like": heading_like,
            "is_list": is_list,
            "level": level,
            "token_est": estimate_tokens_from_chars(e - s),
        })
    return meta


def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter / uni if uni else 0.0

_STOP = set("""a an and are as at be by for from has have in into is it of on or that the to with was were will would can could should may might this those these there here such which when while where who whom whose how why not than then""".split())

def _terms(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}

def _terms_for_title(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    freq = {}
    for w in words:
        if len(w) <= 2 or w in _STOP: continue
        freq[w] = freq.get(w, 0) + 1
    tops = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:5]
    return [w for (w,_) in tops]


class HeuristicProvider(AIProvider):

    def title_from_text(self, text: str) -> str:
        first_line = (text or "").strip().splitlines()[0:1]
        if first_line:
            head = first_line[0].strip()
            if head and len(head) <= 80:
                return head
        kws = _terms_for_title(text)
        if kws:
            return " / ".join([k.capitalize() for k in kws[:3]])
        return super().title_from_text(text)

    def propose_boundaries(self, paras_meta, target_tokens, min_tokens, max_tokens) -> List[int]:
        if not paras_meta:
            return [0, 1]
        boundaries = [0]
        acc = 0
        last = 0
        for i, p in enumerate(paras_meta):
            t = int(p["token_est"])
            will_overflow = acc + t > max_tokens
            close_enough = acc >= min_tokens
            heading = p.get("heading_like", False)
            if i > last and (will_overflow or (close_enough and heading)):
                boundaries.append(i)
                last = i
                acc = t
            else:
                acc += t
        if boundaries[-1] != len(paras_meta):
            boundaries.append(len(paras_meta))
        return boundaries

    def propose_boundaries_context(self, paras_meta, full_text, target_tokens, min_tokens, max_tokens) -> List[int]:
        if not paras_meta:
            return [0,1]
        boundaries = [0]
        acc = 0
        last = 0
        prev_terms = None
        for i, p in enumerate(paras_meta):
            s, e = p["start"], p["end"]
            t = int(p["token_est"])
            terms = _terms(full_text[s:e])
            sim = _jaccard(prev_terms, terms) if prev_terms is not None else 1.0
            break_for_size = acc + t > max_tokens
            ok_size = acc >= min_tokens
            break_for_topic = (sim < 0.2) and ok_size
            heading = p.get("heading_like", False) and ok_size
            if i > last and (break_for_size or break_for_topic or heading):
                boundaries.append(i)
                last = i
                acc = t
            else:
                acc += t
            prev_terms = terms
        if boundaries[-1] != len(paras_meta):
            boundaries.append(len(paras_meta))
        return boundaries


    def _pick_headings(self, text: str) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines()]
        heads = []
        for ln in lines:
            if _heading_re.match(ln):
                if 3 <= len(ln) <= 120: heads.append(ln[:80])
            if len(heads) >= 50: break
        return heads or ["General"]

    def generate_outline(self, corpus_text, num_lessons, title=None, description=None):
        course_title = (title or (self._pick_headings(corpus_text)[0] + " (Course)")).strip()
        course_description = (description or "Auto-generated outline (heuristic).").strip()
        paras = split_into_paragraphs_with_spans(corpus_text)
        lessons, heads = [], self._pick_headings(corpus_text)
        step = max(1, len(paras)//max(1,num_lessons)) if paras else 1
        for i in range(num_lessons):
            if paras:
                p_idx = min(len(paras)-1, i*step)
                s, e = paras[p_idx]["start"], paras[p_idx]["end"]
                summary = " ".join(corpus_text[s:e].strip().split())[:180]
            else:
                summary = corpus_text[:180]
            section = heads[min(i, len(heads)-1)] if heads else "General"
            lessons.append({"title": f"Lesson {i+1}", "summary": summary or "Overview", "section_title": section})
        return {"course_title": course_title, "course_description": course_description, "lessons": lessons}

    def generate_lesson_detail(self, corpus_text, lesson_title, num_questions):
        paras = split_into_paragraphs_with_spans(corpus_text)
        content_parts = []
        for i in range(min(3, len(paras))):
            s, e = paras[i]["start"], paras[i]["end"]
            content_parts.append(corpus_text[s:e].strip())
        content = "\n\n".join(content_parts) or corpus_text[:800]
        sentences = _sentence_re.split(content)
        key_points = [s.strip()[:120] for s in sentences[:3] if s.strip()]
        questions = []
        for i, s in enumerate(sentences[:num_questions]):
            stem = (s.strip() or f"{lesson_title} concept").rstrip(".")
            options = [f"A) {stem}", f"B) Not {stem}", f"C) {stem} (variant)", "D) None"]
            questions.append({"question": stem + "?", "options": options, "correct_answer": "A", "explanation": "Heuristic placeholder."})
        return {"content": content or f"Notes for {lesson_title}.", "key_points": key_points or [lesson_title], "questions": questions}
