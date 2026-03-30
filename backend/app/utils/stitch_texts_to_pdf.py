import argparse, json, csv, re, os, math, random, time
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

                                
_TIKTOKEN_OK = False
try:
    import tiktoken                
    _TIKTOKEN_OK = True
except Exception:
    _TIKTOKEN_OK = False

try:
    from pypdf import PdfReader                         
    _PDF_OK = True
except Exception:
    try:
        from PyPDF2 import PdfReader            
        _PDF_OK = True
    except Exception:
        _PDF_OK = False


                                                                               

def sanitize_text(s: str) -> str:
    if not s: return s
    s = s.replace("\u2011", "-").replace("\u2012", "-")                                           
    s = s.replace("\u2013", "-").replace("\u2014", "-")                                
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201A", ",")           
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u2212", "-")                                                             
    s = s.replace("\u2026", "…")                                                                           
    s = s.replace("\u202f", " ")                                                         
    return s

_inline_bold = re.compile(r"(\*\*|__)(?P<txt>.+?)\1")
_inline_ital = re.compile(
    r"(?<!\*)\*(?!\*)(?P<txt>[^*]+?)(?<!\*)\*(?!\*)|_(?P<txt2>[^_]+?)_"
)
_inline_code = re.compile(r"`(?P<txt>[^`]+?)`")

def md_inline_to_xml(s: str, mono_font: Optional[str]) -> str:
    """
    Convert simple inline Markdown to ReportLab mini-XML (<b>, <i>, <font>).
    IMPORTANT: escape &, <, > BEFORE inserting our tags to avoid malformed XML
    when the text contains math like <f, φ>.
    """
    s = sanitize_text(s or "")

                                
    s = (s
         .replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;"))

                                                                       
    def bold_sub(m):  return f"<b>{m.group('txt')}</b>"
    def ital_sub(m):  return f"<i>{m.group('txt') or m.group('txt2')}</i>"

                                                               
    def code_sub(m):
        face = mono_font or "Courier"
        return f"<font face='{face}'>{m.group('txt')}</font>"

    s = _inline_bold.sub(bold_sub, s)
    s = _inline_ital.sub(ital_sub, s)
    s = _inline_code.sub(code_sub, s)
    return s

def pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Return the exact number of pages in a PDF, or None if unavailable."""
    if not _PDF_OK:
        return None
    try:
        with open(pdf_path, "rb") as fh:
            reader = PdfReader(fh)
            return len(reader.pages)
    except Exception:
        return None

                                                             

def parse_markdown_basic(md_text: str) -> List[Dict[str, Any]]:
    """
    Parse a small, stable subset:
      #/##/### headings; -, 1./1)/"1 " lists; fenced code ```lang; paragraphs.
    Emits blocks: heading/paragraph/list/code
    """
    lines = md_text.splitlines()
    blocks: List[Dict[str, Any]] = []
    para_buf: List[str] = []
    list_buf: List[str] = []
    list_numbered: Optional[bool] = None
    in_code = False
    code_lang = ""
    code_buf: List[str] = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            blocks.append({"type": "paragraph", "text": " ".join(para_buf).strip()})
            para_buf = []

    def flush_list():
        nonlocal list_buf, list_numbered
        if list_buf:
            blocks.append({"type": "list", "numbered": bool(list_numbered), "items": list_buf[:]})
            list_buf = []; list_numbered = None

    for raw in lines:
        line = raw.rstrip("\n")

                     
        m_f = re.match(r"^```(\w+)?\s*$", line)
        if m_f:
            if in_code:
                blocks.append({"type": "code", "lang": code_lang, "text": "\n".join(code_buf)})
                code_buf, code_lang, in_code = [], "", False
            else:
                flush_list(); flush_para()
                in_code, code_lang = True, (m_f.group(1) or "")
            continue
        if in_code:
            code_buf.append(line); continue

                  
        if line.startswith("# "):
            flush_para(); flush_list()
            blocks.append({"type": "heading", "level": 1, "text": line[2:].strip()}); continue
        if line.startswith("## "):
            flush_para(); flush_list()
            blocks.append({"type": "heading", "level": 2, "text": line[3:].strip()}); continue
        if line.startswith("### "):
            flush_para(); flush_list()
            blocks.append({"type": "heading", "level": 3, "text": line[4:].strip()}); continue

               
        if re.match(r"^\s*-\s+", line):
            flush_para()
            item = re.sub(r"^\s*-\s+", "", line).strip()
            if list_numbered is None: list_numbered = False
            elif list_numbered is True: flush_list(); list_numbered = False
            list_buf.append(item); continue

        if re.match(r"^\s*\d+[.)]?\s+", line):                  
            flush_para()
            item = re.sub(r"^\s*\d+[.)]?\s+", "", line).strip()
            if list_numbered is None: list_numbered = True
            elif list_numbered is False: flush_list(); list_numbered = True
            list_buf.append(item); continue

               
        if not line.strip():
            flush_list(); flush_para(); continue

                   
        if list_buf: flush_list()
        para_buf.append(line.strip())

    flush_list(); flush_para()
    if in_code:
        blocks.append({"type": "code", "lang": code_lang, "text": "\n".join(code_buf)})
    return blocks


                                                            

def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))

def count_tokens(text: str, model_name: str = "cl100k_base", approx_words_per_token: float = 0.75) -> int:
    if _TIKTOKEN_OK:
        try:
            enc = tiktoken.get_encoding(model_name)
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, int(count_words(text) / approx_words_per_token))


                                                                         

def closest_subset(weights: List[int], target: int, prefer_more_items: bool = True) -> List[int]:
    """Dynamic frontier with pruning; fine for N up to ~60."""
    frontier: List[Tuple[int, Tuple[int, ...]]] = [(0, tuple())]
    for idx, w in enumerate(weights):
        nf = frontier[:] + [(s + w, sel + (idx,)) for (s, sel) in frontier]
        nf.sort(key=lambda x: x[0])
        pruned: List[Tuple[int, Tuple[int, ...]]] = []
        last, bestlen = None, -1
        for s, sel in nf:
            if last is None or s != last:
                pruned.append((s, sel)); last = s; bestlen = len(sel)
            else:
                if prefer_more_items and len(sel) > bestlen:
                    pruned[-1] = (s, sel); bestlen = len(sel)
        k = max(1, len(pruned) // 4000)
        frontier = pruned[::k] if k > 1 else pruned

    best, best_key = None, None
    for s, sel in frontier:
        diff = abs(s - target)
        overshoot = 1 if s > target else 0
        key = (diff, -len(sel), overshoot)
        if best_key is None or key < best_key:
            best_key, best = key, sel
    return list(best or ())

def improve_pages_selection(items, chosen_idx, target_pages):
    """Greedy nudge using per-item measured pages."""
    chosen = set(chosen_idx)
    current = sum(items[i]["pages"] for i in chosen)
    improved = True
    while improved:
        improved = False
        best = (abs(current - target_pages), None, None)
        for i in range(len(items)):           
            if i in chosen: continue
            new = current + items[i]["pages"]; d = abs(new - target_pages)
            if d < best[0]: best = (d, "add", i)
        for i in list(chosen):              
            new = current - items[i]["pages"]; d = abs(new - target_pages)
            if d < best[0]: best = (d, "del", i)
        if best[1] == "add":
            chosen.add(best[2]); current += items[best[2]]["pages"]; improved = True
        elif best[1] == "del" and len(chosen) > 1:
            chosen.remove(best[2]); current -= items[best[2]]["pages"]; improved = True
    return sorted(chosen)


                                                 

def _try_register_font(font_path: Path, font_name: str) -> bool:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
        return True
    except Exception:
        return False

def register_unicode_font(preferred: Optional[str] = None) -> str:
    candidates: List[Tuple[str, Path]] = []
    if preferred:
        p = Path(preferred).expanduser()
        if p.is_file(): candidates.append(("CustomTTF", p))
    possible = [
        ("DejaVuSans", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")),
        ("DejaVuSans", Path("C:/Windows/Fonts/DejaVuSans.ttf")),
        ("NotoSans",   Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")),
        ("NotoSans",   Path("C:/Windows/Fonts/NotoSans-Regular.ttf")),
        ("SegoeUI",    Path("C:/Windows/Fonts/segoeui.ttf")),
        ("ArialUnicode", Path("C:/Windows/Fonts/arialuni.ttf")),
        ("Arial", Path("C:/Windows/Fonts/arial.ttf")),
        ("Arial", Path("/System/Library/Fonts/Supplemental/Arial.ttf")),
    ]
    for name, p in possible:
        if p.is_file(): candidates.append((name, p))
    for name, p in candidates:
        if _try_register_font(p, name): return name
    return "Helvetica"

def register_mono_font() -> str:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return "Courier"
    candidates = [
        ("DejaVuSansMono", Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")),
        ("DejaVuSansMono", Path("C:/Windows/Fonts/DejaVuSansMono.ttf")),
        ("Consolas",       Path("C:/Windows/Fonts/consola.ttf")),
        ("CascadiaCode",   Path("C:/Windows/Fonts/CascadiaCode.ttf")),
        ("Menlo",          Path("/System/Library/Fonts/Menlo.ttc")),
    ]
    for name, p in candidates:
        if p.is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, str(p)))
                return name
            except Exception:
                pass
    return "Courier"


                                                                        

def _make_styles(font_name: str, mono_name: Optional[str]=None):
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=font_name, fontSize=18, leading=22, spaceAfter=8)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=font_name, fontSize=14, leading=18, spaceBefore=8, spaceAfter=6)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=font_name, fontSize=12, leading=16, spaceBefore=6, spaceAfter=4)
    Body = ParagraphStyle("Body", parent=styles["BodyText"], fontName=font_name, fontSize=10.5, leading=14, alignment=TA_LEFT, spaceAfter=6)
    Code = ParagraphStyle("Code", parent=styles["BodyText"], fontName=(mono_name or "Courier"),
                          fontSize=9.5, leading=12, spaceBefore=6, spaceAfter=6)
    return H1, H2, H3, Body, Code

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", sanitize_text(s or "")).strip().lower()

def build_flowables_for_item(title: str, blocks: List[Dict[str, Any]], styles, mono_font: Optional[str]) -> List[Any]:
    from reportlab.platypus import Paragraph, ListFlowable, ListItem, Preformatted
    H1, H2, H3, Body, Code = styles
    flow = []

                                                                                   
    add_title = True
    if blocks and blocks[0].get("type") == "heading" and blocks[0].get("level") == 1:
        if _norm(blocks[0].get("text")) in {_norm(title), _norm(title).rstrip(":")}:
            add_title = False
    if add_title:
        flow.append(Paragraph(md_inline_to_xml(title, mono_font), H1))

                                                                        
    def _strip_leading_num(s: str) -> str:
        return re.sub(r"^\s*\d+[.)]?\s+", "", s).strip()

    for block in blocks:
        t = block["type"]
        if t == "heading":
            lvl = block.get("level", 2)
            txt = md_inline_to_xml(block["text"], mono_font)
            flow.append(Paragraph(txt, H1 if lvl==1 else H2 if lvl==2 else H3))

        elif t == "paragraph":
            flow.append(Paragraph(md_inline_to_xml(block["text"], mono_font), Body))

        elif t == "list":
            numbered = bool(block.get("numbered"))
            items: ListItem = []
            if numbered:
                                                                                        
                li_objs = []
                for idx, raw in enumerate(block["items"], start=1):
                    cleaned = _strip_leading_num(raw)
                    para = Paragraph(md_inline_to_xml(cleaned, mono_font), Body)
                    li_objs.append(ListItem(para, value=idx))                              
                flow.append(
                    ListFlowable(
                        li_objs,
                        bulletType="1",            
                        start=1,
                        leftIndent=18,
                        spaceBefore=2,
                        spaceAfter=2,
                    )
                )
            else:
                                                
                li_objs = []
                for raw in block["items"]:
                    para = Paragraph(md_inline_to_xml(raw, mono_font), Body)
                    li_objs.append(ListItem(para))
                flow.append(
                    ListFlowable(
                        li_objs,
                        bulletType="bullet",
                        leftIndent=18,
                        spaceBefore=2,
                        spaceAfter=2,
                    )
                )

        elif t == "code":
            flow.append(Preformatted(block["text"], Code))

    return flow



                                                     

def measure_pages_for_item(flowables: List[Any], page_margins_cm: float, pagesize_name: str = "A4") -> int:
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.units import cm
    pagesize = A4 if pagesize_name.upper()=="A4" else LETTER
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=pagesize,
                            leftMargin=page_margins_cm*cm, rightMargin=page_margins_cm*cm,
                            topMargin=page_margins_cm*cm, bottomMargin=page_margins_cm*cm)
    doc.build(flowables)
    try: return int(doc.canv.getPageNumber())
    except Exception: return 1

def measure_pages_of_selection(items, idx_list, styles, margins_cm, pagesize_name) -> int:
    from reportlab.platypus import SimpleDocTemplate, PageBreak
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.units import cm
    pagesize = A4 if pagesize_name.upper()=="A4" else LETTER
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=pagesize,
                            leftMargin=margins_cm*cm, rightMargin=margins_cm*cm,
                            topMargin=margins_cm*cm, bottomMargin=margins_cm*cm)
    story = []
    for i, idx in enumerate(idx_list):
        it = items[idx]
        story.extend(build_flowables_for_item(it["title"], it["blocks"], styles, it.get("_mono")))
        if i < len(idx_list)-1:
            from reportlab.platypus import PageBreak as PB
            story.append(PB())
    doc.build(story)
    try: return int(doc.canv.getPageNumber())
    except Exception: return 1

def refine_selection_fulldoc(items, initial_idx, target_pages, styles, margins_cm, pagesize_name) -> List[int]:
    """Greedy refine using full-document page measurement."""
    chosen = list(initial_idx)
    best_pages = measure_pages_of_selection(items, chosen, styles, margins_cm, pagesize_name)
    best_diff = abs(best_pages - target_pages)
    improved = True
    while improved:
        improved = False
        current_set = set(chosen)
                         
        best_add = None; best_add_diff = best_diff
        for i in range(len(items)):
            if i in current_set: continue
            test_idx = chosen + [i]
            pages = measure_pages_of_selection(items, test_idx, styles, margins_cm, pagesize_name)
            d = abs(pages - target_pages)
            if d < best_add_diff:
                best_add_diff, best_add = d, i
                                           
        best_del = None; best_del_diff = best_diff
        if len(chosen) > 1:
            for i in list(chosen):
                test_idx = [j for j in chosen if j != i]
                pages = measure_pages_of_selection(items, test_idx, styles, margins_cm, pagesize_name)
                d = abs(pages - target_pages)
                if d < best_del_diff:
                    best_del_diff, best_del = d, i

                                            
        if best_add is not None and best_add_diff < best_diff and best_add_diff <= best_del_diff:
            chosen.append(best_add); best_diff = best_add_diff; improved = True
        elif best_del is not None and best_del_diff < best_diff:
            chosen.remove(best_del); best_diff = best_del_diff; improved = True

    return chosen


                                                                          

def _extract_metrics_from_new_meta(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    m = meta.get("metrics") or {}
    return (m.get("words_estimated") or meta.get("words"),
            m.get("tokens_estimated") or meta.get("tokens"),
            m.get("pages_estimated") or meta.get("pages"))

def _infer_topic_length(base: Path, md_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    From <base>/<topic>/<length>/<file>, extract ('topic','length'); else (None,None)
    """
    try:
        rel = md_path.relative_to(base)
    except Exception:
        return None, None
    parts = list(rel.parts)
    topic = parts[0] if len(parts) >= 3 else None
    length = parts[1] if len(parts) >= 3 else None
    if length not in {"short","medium","long"}:
        length = None
    return topic, length

def load_corpus(indir: Path, topics_filter: Optional[List[str]] = None,
                lengths_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load *.md + matching *.json (JSON must have 'filename').
    Supports:
      - Flat:   <indir>/*.md|json
      - Nested: <indir>/<topic>/<length>/*.md|json
    Applies optional filters by topics & lengths.
    """
    base = Path(indir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No such corpus directory: {base}")

                   
    md_files = list(base.rglob("*.md"))
    json_files = list(base.rglob("*.json"))
    md_map: Dict[str, Path] = {p.name: p for p in md_files}

    corpus: List[Dict[str, Any]] = []
    for jf in json_files:
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        fname = meta.get("filename")
        if not fname or fname not in md_map:
            continue
        md_path = md_map[fname]
        sibling_pdf = md_path.with_suffix(".pdf")

        topic, length = _infer_topic_length(base, md_path)

                                         
        if topics_filter:
            tset = {x.lower() for x in topics_filter}
            if not topic or topic.lower() not in tset:
                continue

                                          
        if lengths_filter:
            lset = {x.lower() for x in lengths_filter}
            if not length or length.lower() not in lset:
                continue

        text = md_path.read_text(encoding="utf-8")
        blocks = parse_markdown_basic(text)
        words_meta, tokens_meta, pages_meta = _extract_metrics_from_new_meta(meta)
        title = meta.get("title") or Path(fname).stem

        corpus.append({
            "filename": fname,
            "path": str(md_path),
            "title": title,
            "text": text,
            "blocks": blocks,
            "meta": meta,
            "words_meta": words_meta,
            "tokens_meta": tokens_meta,
            "pages_meta": pages_meta,
            "topic": topic,
            "length": length,
            "sibling_pdf": str(sibling_pdf) if sibling_pdf.exists() else None,
        })

    if not corpus:
        raise RuntimeError(f"No matched {{markdown,json}} pairs under {base}. Each JSON must include a matching 'filename'.")
    return corpus


                                                             

def next_run_folder(base_out: Path) -> Path:
    base_out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base_out.iterdir() if p.is_dir() and re.match(r"^run_\d{3}$", p.name)]
    idx = 1
    if existing:
        nums = [int(p.name.split("_")[1]) for p in existing]
        idx = max(nums) + 1
    run = base_out / f"run_{idx:03d}"
    run.mkdir(parents=True, exist_ok=True)
    return run


                                                     

def render_pdf(selections: List[Dict[str, Any]], out_pdf: Path, font_name: str, mono_font: str,
               page_margins_cm: float, pagesize_name: str = "A4") -> None:
    from reportlab.platypus import SimpleDocTemplate, PageBreak
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.units import cm
    pagesize = A4 if pagesize_name.upper()=="A4" else LETTER
    styles = _make_styles(font_name, mono_font)
    story = []
    for i, sel in enumerate(selections):
        story.extend(build_flowables_for_item(sel.get("title") or sel["filename"], sel["blocks"], styles, mono_font))
        if i < len(selections) - 1:
            story.append(PageBreak())
    doc = SimpleDocTemplate(str(out_pdf), pagesize=pagesize,
                            leftMargin=page_margins_cm*cm, rightMargin=page_margins_cm*cm,
                            topMargin=page_margins_cm*cm, bottomMargin=page_margins_cm*cm)
    doc.build(story)


                                                                           

def run_stitch(
    indir: Path,
    outdir: Path,
    target_metric: str,
    target_value: int,
    topics: Optional[List[str]] = None,
    lengths: Optional[List[str]] = None,
    approx_words_per_token: float = 0.75,
    approx_words_per_page: int = 550,
    seed: Optional[int] = None,
    page_margins_cm: float = 2.0,
    pagesize: str = "A4",
    preview_html: bool = False,
    font_ttf: Optional[str] = None,
    selection_mode: str = "mixed"
) -> Dict[str, Any]:
    """
    Core function used by CLI and can be imported from backend.
    Returns the bundle metadata dict.
    """
    if seed is None:
        seed = int(time.time() * 1000) & 0xFFFFFFFF
    random.seed(seed)

    run_dir = next_run_folder(outdir)

           
    font_name = register_unicode_font(font_ttf)
    mono_font = register_mono_font()

                                
    corpus = load_corpus(indir, topics_filter=(topics or None),
                         lengths_filter=(lengths or None))

                          
    styles = _make_styles(font_name, mono_font)

                                                                     
    items: List[Dict[str, Any]] = []
    for doc in corpus:
        text = doc["text"]
        words = int(doc["words_meta"]) if doc["words_meta"] is not None else count_words(text)
        tokens = int(doc["tokens_meta"]) if doc["tokens_meta"] is not None else count_tokens(text, approx_words_per_token=approx_words_per_token)
                                                                              
        pages_meta = doc.get("pages_meta")
        pages_pdf = pdf_page_count(Path(doc["sibling_pdf"])) if doc.get("sibling_pdf") else None
        pages = None
        if pages_pdf is not None:
            pages = pages_pdf
        elif pages_meta is not None:
            pages = int(pages_meta)
        if pages is None:
                                                                                      
            pages = max(1, math.ceil(words / max(1, approx_words_per_page)))
        items.append({
            "filename": doc["filename"],
            "title": doc.get("title") or doc["filename"],
            "blocks": doc["blocks"],
            "text": text,
            "words": words,
            "tokens": tokens,
            "pages": pages,
            "meta": doc["meta"],
            "_mono": mono_font,
            "topic": doc.get("topic"),
            "length": doc.get("length"),
            "sibling_pdf": doc.get("sibling_pdf"),
        })

                                  
    perm = list(range(len(items))); random.shuffle(perm)
    items = [items[i] for i in perm]

                                                      
    def jitter(v: int) -> int:
        mag = max(1, int(0.01 * max(1, v)))
        return v + random.randint(-mag, mag)

                       
    chosen_idx: List[int] = []
    metric_name, target_val = None, None
    if target_metric == "num_texts":
        N = max(1, min(target_value, len(items)))
        idxs = list(range(len(items)))
        idxs.sort(key=lambda i: items[i]["words"], reverse=True)
        chosen_idx = idxs[:N]
        metric_name, target_val = "num_texts", N
    elif target_metric == "total_tokens":
        weights = [jitter(it["tokens"]) for it in items]
        target_val = target_value; metric_name = "total_tokens"
        chosen_idx = closest_subset(weights, target_val, prefer_more_items=True)
    elif target_metric == "total_words":
        weights = [jitter(it["words"]) for it in items]
        target_val = target_value; metric_name = "total_words"
        chosen_idx = closest_subset(weights, target_val, prefer_more_items=True)
    else:               
        weights = [jitter(it["pages"]) for it in items]
        target_val = target_value; metric_name = "total_pages"
        chosen_idx = closest_subset(weights, target_val, prefer_more_items=True)
        chosen_idx = improve_pages_selection(items, chosen_idx, target_value)

    if not chosen_idx:
                                          
        closest = min(range(len(items)), key=lambda i: abs(items[i]["words"] - (target_value or 1)))
        chosen_idx = [closest]

                                                                   

    selected = [items[i] for i in chosen_idx]

                                                       
    total_words = sum(it["words"] for it in selected)
    total_tokens = sum(it["tokens"] for it in selected)
    total_pages = sum(it["pages"] for it in selected)

                  
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if metric_name == "num_texts":
        prefix = f"bundle_texts-{len(selected)}_{ts}"
    elif metric_name == "total_tokens":
        prefix = f"bundle_tokens-{target_val}_{ts}"
    elif metric_name == "total_words":
        prefix = f"bundle_words-{target_val}_{ts}"
    else:
        prefix = f"bundle_pages-{target_value}_{ts}"

    out_pdf = run_dir / f"{prefix}.pdf"
    out_json = run_dir / f"{prefix}.json"
    out_csv = run_dir / f"{prefix}.csv"
    out_html = run_dir / f"{prefix}.preview.html" if preview_html else None

                                                                                     
    from reportlab.platypus import SimpleDocTemplate, PageBreak
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.units import cm
    pagesize_obj = A4 if pagesize.upper()=="A4" else LETTER
    buf = BytesIO()
    doc_tmp = SimpleDocTemplate(buf, pagesize=pagesize_obj,
                                leftMargin=page_margins_cm*cm, rightMargin=page_margins_cm*cm,
                                topMargin=page_margins_cm*cm, bottomMargin=page_margins_cm*cm)
    styles_final = _make_styles(font_name, mono_font)
    story_tmp = []
    for i, it in enumerate(selected):
        story_tmp.extend(build_flowables_for_item(it["title"], it["blocks"], styles_final, mono_font))
        if i < len(selected) - 1:
            story_tmp.append(PageBreak())
    doc_tmp.build(story_tmp)
    try:
        total_pages = int(doc_tmp.canv.getPageNumber())
    except Exception:
        pass

                
    render_pdf(
        selections=[{"filename": it["filename"], "title": it["title"], "blocks": it["blocks"]} for it in selected],
        out_pdf=out_pdf, font_name=font_name, mono_font=mono_font,
        page_margins_cm=page_margins_cm, pagesize_name=pagesize
    )

    final_pages = pdf_page_count(out_pdf)
    if final_pages is not None:
        total_pages = int(final_pages)

                           
    if out_html:
        def html_escape(s: str) -> str:
            return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts = ["<html><head><meta charset='utf-8'><style>"
                 "body{font-family:system-ui,Segoe UI,Roboto,Arial}"
                 "h1{font-size:20px}h2{font-size:16px}h3{font-size:14px}"
                 "p{line-height:1.45}ul,ol{margin-left:1.2em}"
                 "pre{background:#f6f8fa;padding:10px;border-radius:6px;overflow:auto}"
                 "</style></head><body>"]
        for it in selected:
            parts.append(f"<h1>{html_escape(it['title'])}</h1>")
            for b in it["blocks"]:
                if b["type"] == "heading":
                    tag = "h1" if b.get("level",2)==1 else ("h2" if b.get("level",2)==2 else "h3")
                    parts.append(f"<{tag}>{html_escape(b['text'])}</{tag}>")
                elif b["type"] == "paragraph":
                    parts.append(f"<p>{html_escape(b['text'])}</p>")
                elif b["type"] == "list":
                    tag = "ol" if b.get("numbered") else "ul"
                    parts.append(f"<{tag}>")
                    for li in b["items"]:
                        parts.append(f"<li>{html_escape(li)}</li>")
                    parts.append(f"</{tag}>")
                elif b["type"] == "code":
                    parts.append(f"<pre>{html_escape(b['text'])}</pre>")
            parts.append("<hr/>")
        parts.append("</body></html>")
        out_html.write_text("\n".join(parts), encoding="utf-8")

                       
    sel_mode = "topics" if (topics or lengths) else selection_mode or "mixed"
    topics_used = sorted({it["topic"] for it in selected if it.get("topic")})
    lengths_used = sorted({it["length"] for it in selected if it.get("length")})

    bundle = {
        "schema_version": "1.5.2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator": {
            "tool": "stitch_texts_to_pdf.py",
            "reportlab": True,
            "tiktoken_used": _TIKTOKEN_OK,
            "font": font_name,
            "mono_font": mono_font
        },
                               
        "topics": topics_used,                                           
        "lengths": lengths_used,                           
        "inputs": {
            "indir": str(indir.resolve()),
            "out_run_dir": str(run_dir.resolve()),
            "target_metric": metric_name,
            "target_value": (target_value if metric_name != "num_texts" else len(selected)),
            "margins_cm": page_margins_cm,
            "pagesize": pagesize,
            "seed": seed,
            "selection_mode": sel_mode,
            "requested_topics": topics or [],
            "requested_lengths": lengths or []
        },
        "selection": {
            "count": len(selected),
            "filenames": [it["filename"] for it in selected],
            "topics_used": topics_used,
            "lengths_used": lengths_used,
        },
        "totals": {
            "words": total_words,
            "tokens": total_tokens,
            "pages_measured": total_pages
        },
        "files": {
            "combined_pdf": out_pdf.name,
            "summary_csv": out_csv.name,
            "preview_html": out_html.name if out_html else None
        },
        "items": [{
            "filename": it["filename"],
            "title": it["title"],
            "words": it["words"],
            "tokens": it["tokens"],
            "pages_measured": it["pages"],
            "topic": it.get("topic"),
            "length": it.get("length"),
            "source_meta": it["meta"]
        } for it in selected]
    }
    out_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    with (run_dir / out_csv.name).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","title","words","tokens","pages_measured","topic","length"])
        for it in selected:
            w.writerow([it["filename"], it["title"], it["words"], it["tokens"], it["pages"], it.get("topic") or "", it.get("length") or ""])

    return bundle


                                               

def main():
    ap = argparse.ArgumentParser(description="Stitch pre-made texts into a PDF close to requested size.")
    ap.add_argument("--indir", type=str, default=str(Path("backend/generated").resolve()),
                    help="Corpus root (default: backend/generated). Supports flat or <topic>/<length>/")
    ap.add_argument("--outdir", type=str, default=str(Path("backend/testdata").resolve()),
                    help="Base output folder (default: backend/testdata)")

                          
    ap.add_argument("--topics", type=str, default="", help="Comma-separated topics to include (default: all)")
    ap.add_argument("--lengths", type=str, default="short,medium,long", help="Comma-separated lengths to include")

    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--num-texts", type=int)
    target.add_argument("--total-words", type=int)
    target.add_argument("--total-tokens", type=int)
    target.add_argument("--total-pages", type=int)

    ap.add_argument("--approx-words-per-token", type=float, default=0.75)
    ap.add_argument("--approx-words-per-page", type=int, default=550)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--page-margins-cm", type=float, default=2.0)
    ap.add_argument("--pagesize", type=str, default="A4", choices=["A4","LETTER"])
    ap.add_argument("--preview-html", action="store_true")
    ap.add_argument("--font-ttf", type=str, default=None)
    ap.add_argument("--selection-mode", type=str, default="mixed", choices=["mixed","topics"])

    args = ap.parse_args()

    topics = [t.strip() for t in (args.topics or "").split(",") if t.strip()] or None
    lengths = [l.strip().lower() for l in (args.lengths or "").split(",") if l.strip()] or None

    if args.num_texts is not None:
        metric, value = "num_texts", args.num_texts
    elif args.total_tokens is not None:
        metric, value = "total_tokens", args.total_tokens
    elif args.total_words is not None:
        metric, value = "total_words", args.total_words
    else:
        metric, value = "total_pages", args.total_pages

    bundle = run_stitch(
        indir=Path(args.indir),
        outdir=Path(args.outdir),
        target_metric=metric,
        target_value=value,
        topics=topics,
        lengths=lengths,
        approx_words_per_token=args.approx_words_per_token,
        approx_words_per_page=args.approx_words_per_page,
        seed=args.seed,
        page_margins_cm=args.page_margins_cm,
        pagesize=args.pagesize,
        preview_html=args.preview_html,
        font_ttf=args.font_ttf,
        selection_mode=args.selection_mode
    )

    print(f"✅ PDF:   {Path(bundle['files']['combined_pdf']).name} (in {bundle['inputs']['out_run_dir']})")
    print(f"🧾 JSON:  {bundle['files']['summary_csv']} / {Path(bundle['files']['combined_pdf']).with_suffix('.json').name}")
    if bundle['files']['preview_html']:
        print(f"🔎 HTML:  {bundle['files']['preview_html']}")


if __name__ == "__main__":
    main()
