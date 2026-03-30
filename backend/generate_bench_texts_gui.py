import os, re, json, math, asyncio, threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from slugify import slugify
from openai import AsyncOpenAI, APIStatusError, APIConnectionError

from dotenv import load_dotenv
load_dotenv()

_TIKTOKEN_OK = False
try:
    import tiktoken
    _TIKTOKEN_OK = True
except Exception:
    _TIKTOKEN_OK = False

import tkinter as tk
from tkinter import ttk, messagebox

LENGTH_SPEC = {
    "short":  dict(words=(350, 700),   tokens=900),
    "medium": dict(words=(900, 1600),  tokens=2200),
    "long":   dict(words=(1800, 3200), tokens=4400),
}

SYSTEM_INSTRUCTIONS = (
    "You are an expert long-form writer. Generate coherent, human-like Markdown "
    "with headings (#, ##, ###), paragraphs, and occasional lists. No front matter. "
    "Avoid meta talk and placeholders. Title must be first H1."
)

USER_PROMPT_TEMPLATE = (
    "Topic area: {topic}\n\n"
    "Write a {length_label} article with:\n"
    "- Markdown headings and paragraphs; few lists OK\n"
    "- Natural flow; avoid excessive enumerations\n"
    "- Self-contained; no references to being an AI\n"
    "- Title as first H1 (# Title)\n\n"
    "Target length (soft): {words_target_min}-{words_target_max} words "
    "(~{tokens_target} tokens)."
)

MODEL_FALLBACKS = ["gpt-5", "gpt-4o", "gpt-4o-mini"]

# --------- Text utils ---------
def count_words(s: str) -> int:
    return len(re.findall(r"\S+", s or ""))

def count_tokens(s: str, model_name: str = "cl100k_base", approx_words_per_token: float = 0.75) -> int:
    if _TIKTOKEN_OK:
        try:
            enc = tiktoken.get_encoding(model_name)
            return len(enc.encode(s))
        except Exception:
            pass
    return max(1, int(count_words(s) / approx_words_per_token))

def sanitize_text(s: str) -> str:
    if not s: return s
    s = s.replace("\u2011", "-").replace("\u2012", "-")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201A", ",")
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u2212", "-")
    s = s.replace("\u2026", "...")
    s = s.replace("\u202f", " ")
    return s

def parse_markdown_basic(md_text: str) -> List[Dict[str, Any]]:
    """
    Markdown -> block list with nested lists.

    Blocks emitted:
      - {"type":"heading","level":1..3,"text":...}
      - {"type":"paragraph","text":...}
      - {"type":"code","lang":..., "text":...}
      - {"type":"list","numbered":bool,"items":[
            {"text":"...", "children":[<list blocks>]},
            ...
         ]}

    Nesting is determined by leading spaces (every 2+ spaces increases depth).
    Ordered items:  "1. xxx", "1) xxx", or "1 xxx" (we always renumber 1..N).
    Unordered items: "- xxx".
    """
    lines = md_text.splitlines()
    blocks: List[Dict[str, Any]] = []

    # --- helpers ------------------------------------------------------------
    def push_paragraph(buf: List[str]):
        if buf:
            blocks.append({"type": "paragraph", "text": " ".join(buf).strip()})
            buf.clear()

    def start_new_list(numbered: bool):
        return {"type": "list", "numbered": numbered, "items": []}

    def add_list_item(lst: Dict[str, Any], text: str):
        lst["items"].append({"text": text, "children": []})

    def current_item(lst: Dict[str, Any]):
        return lst["items"][-1] if lst["items"] else None

    # --- state --------------------------------------------------------------
    para_buf: List[str] = []
    in_code = False
    code_lang = ""
    code_buf: List[str] = []

    list_stack: List[Tuple[int, Dict[str, Any]]] = []

    # --- regex --------------------------------------------------------------
    rgx_codefence = re.compile(r"^```(\w+)?\s*$")
    rgx_h1 = re.compile(r"^# (.+)")
    rgx_h2 = re.compile(r"^## (.+)")
    rgx_h3 = re.compile(r"^### (.+)")
    rgx_unordered = re.compile(r"^(?P<indent>\s*)-\s+(?P<text>.+)")
    rgx_ordered = re.compile(r"^(?P<indent>\s*)\d+[.)]?\s+(?P<text>.+)")

    def close_all_lists():
        nonlocal list_stack
        if not list_stack:
            return
        blocks.append(list_stack[0][1])
        list_stack = []

    # --- main loop ----------------------------------------------------------
    for raw in lines:
        line = raw.rstrip("\n")

        m_f = rgx_codefence.match(line)
        if m_f:
            if in_code:
                blocks.append({"type": "code", "lang": code_lang, "text": "\n".join(code_buf)})
                code_buf, code_lang, in_code = [], "", False
            else:
                push_paragraph(para_buf)
                close_all_lists()
                in_code, code_lang = True, (m_f.group(1) or "")
            continue
        if in_code:
            code_buf.append(line)
            continue

        m1, m2, m3 = rgx_h1.match(line), rgx_h2.match(line), rgx_h3.match(line)
        if m1 or m2 or m3:
            push_paragraph(para_buf)
            close_all_lists()
            text = (m1 or m2 or m3).group(1).strip()
            level = 1 if m1 else 2 if m2 else 3
            blocks.append({"type": "heading", "level": level, "text": text})
            continue

        mu = rgx_unordered.match(line)
        if mu:
            push_paragraph(para_buf)
            indent = len(mu.group("indent").expandtabs(2))
            text = mu.group("text").strip()
            while list_stack and list_stack[-1][0] > indent:
                list_stack.pop()
            if not list_stack or list_stack[-1][0] < indent or list_stack[-1][1]["numbered"]:
                lst = start_new_list(False)
                if list_stack and list_stack[-1][0] < indent:
                    parent = list_stack[-1][1]
                    itm = current_item(parent)
                    if itm is None:
                        add_list_item(parent, "")
                        itm = current_item(parent)
                    itm["children"].append(lst)
                else:
                    close_all_lists()
                    list_stack = [(indent, lst)]
                    continue
                list_stack.append((indent, lst))
            add_list_item(list_stack[-1][1], text)
            continue

        mo = rgx_ordered.match(line)
        if mo:
            push_paragraph(para_buf)
            indent = len(mo.group("indent").expandtabs(2))
            text = mo.group("text").strip()
            while list_stack and list_stack[-1][0] > indent:
                list_stack.pop()
            if not list_stack or list_stack[-1][0] < indent or not list_stack[-1][1]["numbered"]:
                lst = start_new_list(True)
                if list_stack and list_stack[-1][0] < indent:
                    parent = list_stack[-1][1]
                    itm = current_item(parent)
                    if itm is None:
                        add_list_item(parent, "")
                        itm = current_item(parent)
                    itm["children"].append(lst)
                else:
                    close_all_lists()
                    list_stack = [(indent, lst)]
                    continue
                list_stack.append((indent, lst))
            add_list_item(list_stack[-1][1], text)
            continue

        if not line.strip():
            push_paragraph(para_buf)
            continue

        para_buf.append(line.strip())

    if in_code:
        blocks.append({"type": "code", "lang": code_lang, "text": "\n".join(code_buf)})
    push_paragraph(para_buf)
    close_all_lists()
    return blocks

def extract_headings_with_spans(md_text: str):
    headings = []
    char_pos = 0
    lines = md_text.splitlines(keepends=True)
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        m1 = re.match(r"^# (.+)", line) or re.match(r"^## (.+)", line) or re.match(r"^### (.+)", line)
        if m1:
            title = m1.group(1).strip()
            level = 1 if line.startswith("# ") else 2 if line.startswith("## ") else 3
            start_char = char_pos
            j = i; end_char = start_char + len(raw)
            for k in range(i, len(lines)):
                nxt = lines[k]
                if re.match(r"^#{1,3} ", nxt): break
                end_char += len(nxt); j = k+1
            headings.append({
                "title": title, "level": level,
                "line_start": i, "char_start": start_char,
                "line_end": j, "char_end": end_char-1,
                "id": slugify(title)
            })
        char_pos += len(raw)
    return headings, len(headings)

# --------- ReportLab rendering ---------
from reportlab.platypus import SimpleDocTemplate, Paragraph, ListFlowable, ListItem, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def register_fonts():
    tried = [
        ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("NotoSans",   "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        ("SegoeUI",    "C:/Windows/Fonts/segoeui.ttf"),
        ("Arial",      "C:/Windows/Fonts/arial.ttf"),
    ]
    for name, path in tried:
        p = Path(path)
        if p.is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, str(p)))
                return name
            except Exception:
                pass
    return "Helvetica"

def register_mono():
    tried = [
        ("DejaVuSansMono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        ("Consolas",       "C:/Windows/Fonts/consola.ttf"),
        ("CascadiaCode",   "C:/Windows/Fonts/CascadiaCode.ttf"),
    ]
    for name, path in tried:
        p = Path(path)
        if p.is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, str(p)))
                return name
            except Exception:
                pass
    return "Courier"

def make_styles(font_name: str, mono_name: str):
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=font_name, fontSize=18, leading=22, spaceAfter=8)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=font_name, fontSize=14, leading=18, spaceBefore=8, spaceAfter=6)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=font_name, fontSize=12, leading=16, spaceBefore=6, spaceAfter=4)
    Body = ParagraphStyle("Body", parent=styles["BodyText"], fontName=font_name, fontSize=10.5, leading=14, alignment=TA_LEFT, spaceAfter=6)
    Code = ParagraphStyle("Code", parent=styles["Code"] if "Code" in styles else Body,
                          fontName=(mono_name or "Courier"), fontSize=9.5, leading=12, spaceBefore=6, spaceAfter=6)
    return H1, H2, H3, Body, Code

def md_inline_to_xml(s: str, mono_font: Optional[str]) -> str:
    s = sanitize_text(s)
    s = re.sub(r"(\*\*|__)(.+?)\1", r"<b>\2</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"_(.+?)_", r"<i>\1</i>", s)
    def code_sub(m):
        txt = m.group(1).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        return f"<font face='{mono_font or 'Courier'}'>{txt}</font>"
    s = re.sub(r"`([^`]+?)`", code_sub, s)
    return s

def build_flowables_from_markdown(md_text: str, styles, mono_font: str) -> List[Any]:
    from reportlab.platypus import Paragraph, ListFlowable, ListItem, Preformatted
    H1, H2, H3, Body, Code = styles
    blocks = parse_markdown_basic(md_text)
    flow: List[Any] = []

    def render_list(block: Dict[str, Any]) -> ListFlowable:
        """Recursively render a list block with possible nested children."""
        items_flow = []
        for it in block["items"]:
            content = [Paragraph(md_inline_to_xml(it.get("text",""), mono_font), Body)]
            for child in it.get("children", []):
                content.append(render_list(child))
            items_flow.append(ListItem(content))

        if block.get("numbered"):
            return ListFlowable(
                items_flow,
                bulletType='1',
                start="1",
                leftIndent=18,
                bulletFontName=Body.fontName
            )
        else:
            return ListFlowable(
                items_flow,
                bulletType='bullet',
                leftIndent=18,
                bulletChar=u'•',
                bulletFontName=Body.fontName
            )

    for b in blocks:
        t = b["type"]
        if t == "heading":
            lvl = b.get("level", 2)
            flow.append(Paragraph(md_inline_to_xml(b["text"], mono_font),
                                  H1 if lvl==1 else H2 if lvl==2 else H3))
        elif t == "paragraph":
            flow.append(Paragraph(md_inline_to_xml(b["text"], mono_font), Body))
        elif t == "code":
            flow.append(Preformatted(b["text"], Code))
        elif t == "list":
            flow.append(render_list(b))

    return flow


def markdown_to_pdf(md_text: str, out_pdf: Path, margins_cm: float = 2.0):
    font = register_fonts()
    mono = register_mono()
    styles = make_styles(font, mono)
    story = build_flowables_from_markdown(md_text, styles, mono)
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4,
                            leftMargin=margins_cm*cm, rightMargin=margins_cm*cm,
                            topMargin=margins_cm*cm, bottomMargin=margins_cm*cm)
    doc.build(story)

# --------- OpenAI generation ---------
@dataclass
class GenTask:
    topic: str
    length_key: str
    index: int
    out_dir: Path
    model: str

async def generate_one(client: AsyncOpenAI, task: GenTask, max_retries: int = 3):
    spec = LENGTH_SPEC[task.length_key]
    words_min, words_max = spec["words"]
    tokens_target = spec["tokens"]
    prompt = USER_PROMPT_TEMPLATE.format(
        topic=task.topic, length_label=task.length_key,
        words_target_min=words_min, words_target_max=words_max,
        tokens_target=tokens_target
    )
    last_err = None
    for attempt in range(1, max_retries+1):
        for model_name in [task.model] + [m for m in MODEL_FALLBACKS if m != task.model]:
            try:
                resp = await client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": prompt},
                    ],
                    max_output_tokens=max(1200, tokens_target+400),
                )
                text = getattr(resp, "output_text", None) or str(resp)
                return True, text, model_name
            except (APIStatusError, APIConnectionError, Exception) as e:
                last_err = e
                await asyncio.sleep(0.6 * attempt)
    return False, None, None

def build_json_metadata(md_text: str, title_fallback: str, topic: str, length_key: str, src_id: str) -> Dict[str, Any]:
    text = md_text.strip()
    m = re.match(r"^#\s+(.+)$", text, flags=re.M)
    title = m.group(1).strip() if m else title_fallback
    words = count_words(text)
    tokens = count_tokens(text)
    headings, hcount = extract_headings_with_spans(text)
    pages_est = max(1, math.ceil(words / 550))
    return {
        "filename": "",
        "title": title,
        "id": src_id,
        "tags": [topic, length_key],
        "description": f"{length_key.capitalize()} Markdown article for topic '{topic}'.",
        "source": "ChatGPT-generated benchmark text",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": "Markdown includes headings. Structure contains heading spans for benchmarking.",
        "metrics": {
            "words_estimated": words,
            "tokens_estimated": tokens,
            "pages_estimated": pages_est
        },
        "structure": {
            "headings": headings,
            "heading_count": hcount,
            "schema": "level=1..3; line_start/end 1-based; char_start/end 0-based inclusive"
        }
    }

async def generate_batch(topic: str, n_short: int, n_med: int, n_long: int, base_out: Path, model_pref: str, concurrency: int = 5, status_cb=None):
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    topic_slug = slugify(topic)
    root = base_out / topic_slug
    dirs = {"short": root / "short", "medium": root / "medium", "long": root / "long"}
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    tasks: List[GenTask] = []
    for i in range(1, n_short+1): tasks.append(GenTask(topic, "short",  i, dirs["short"],  model_pref))
    for i in range(1, n_med+1):   tasks.append(GenTask(topic, "medium", i, dirs["medium"], model_pref))
    for i in range(1, n_long+1):  tasks.append(GenTask(topic, "long",   i, dirs["long"],   model_pref))

    sem = asyncio.Semaphore(concurrency)

    async def worker(t: GenTask):
        async with sem:
            if status_cb: status_cb(f"Generating {t.length_key} #{t.index} ...")
            ok, text, model_used = await generate_one(client, t)
            if not ok or not text:
                if status_cb: status_cb(f"FAILED: {t.length_key} #{t.index}")
                return
            text = sanitize_text(text)
            base_name = f"{topic_slug}_{t.length_key}_{t.index:03d}"
            md_path = t.out_dir / f"{base_name}.md"
            pdf_path = t.out_dir / f"{base_name}.pdf"
            json_path = t.out_dir / f"{base_name}.json"

            md_path.write_text(text, encoding="utf-8")
            meta = build_json_metadata(text, f"{topic} ({t.length_key})", topic, t.length_key, src_id=base_name)
            meta["filename"] = md_path.name
            meta["model_used"] = model_used
            json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            markdown_to_pdf(text, pdf_path)
            if status_cb: status_cb(f"Saved: {md_path.name}, {json_path.name}, {pdf_path.name}")

    await asyncio.gather(*(worker(t) for t in tasks))
    if status_cb: status_cb(f"Done. Output in: {root}")

# --------- GUI ---------
class LabeledScale(ttk.Frame):
    """
    Composite: Label 'Short', Scale 0..N, Spinbox (0..N) kept in sync.
    """
    def __init__(self, master, text: str, from_=0, to=20, **kwargs):
        super().__init__(master, **kwargs)
        self.var = tk.IntVar(value=0)
        ttk.Label(self, text=text).grid(row=0, column=0, sticky="w")
        self.scale = ttk.Scale(self, from_=from_, to=to, orient="horizontal",
                               command=self._on_scale, length=300)
        self.scale.grid(row=0, column=1, padx=(8,6))
        self.spin = ttk.Spinbox(self, from_=from_, to=to, width=4, textvariable=self.var, command=self._on_spin,
                                justify="center")
        self.spin.grid(row=0, column=2)

    def _on_scale(self, val):
        v = int(round(float(val)))
        self.var.set(v)

    def _on_spin(self):
        try:
            v = int(self.var.get())
        except Exception:
            v = 0
        v = max(int(self.spin.cget("from")), min(int(self.spin.cget("to")), v))
        self.var.set(v)
        self.scale.set(v)

    def get(self) -> int:
        return int(self.var.get())

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Benchmark PDF Generator (OpenAI)")
        self.geometry("680x500")
        self.resizable(False, False)

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Topic area (e.g., computer science, health, glass, geography):").grid(row=0, column=0, sticky="w")
        self.topic_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.topic_var, width=64).grid(row=1, column=0, columnspan=4, sticky="we", pady=(0,8))

        ttk.Label(frm, text="Model preference (falls back automatically):").grid(row=2, column=0, sticky="w")
        self.model_var = tk.StringVar(value="gpt-5")
        ttk.Entry(frm, textvariable=self.model_var, width=24).grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text="How many texts to generate per length:").grid(row=3, column=0, sticky="w", pady=(8,2))
        self.s_short  = LabeledScale(frm, "Short", 0, 20);  self.s_short.grid(row=4, column=0, columnspan=4, sticky="w", pady=2)
        self.s_medium = LabeledScale(frm, "Medium",0, 20);  self.s_medium.grid(row=5, column=0, columnspan=4, sticky="w", pady=2)
        self.s_long   = LabeledScale(frm, "Long",  0, 20);  self.s_long.grid(row=6, column=0, columnspan=4, sticky="w", pady=2)

        self.total_var = tk.StringVar(value="Total texts: 0")
        def update_total(*_):
            total = self.s_short.get() + self.s_medium.get() + self.s_long.get()
            self.total_var.set(f"Total texts: {total}")
        for w in (self.s_short, self.s_medium, self.s_long):
            w.var.trace_add("write", update_total)
        ttk.Label(frm, textvariable=self.total_var, foreground="#0a7").grid(row=7, column=0, sticky="w", pady=(2,6))

        ttk.Label(frm, text="Parallel requests (1–8):").grid(row=8, column=0, sticky="w")
        self.conc_var = tk.IntVar(value=4)
        self.conc = ttk.Spinbox(frm, from_=1, to=8, width=4, textvariable=self.conc_var, justify="center")
        self.conc.grid(row=8, column=1, sticky="w")

        ttk.Label(frm, text="Base output folder:").grid(row=9, column=0, sticky="w", pady=(8,0))
        self.out_var = tk.StringVar(value=str((Path.cwd() / "generated").resolve()))
        ttk.Entry(frm, textvariable=self.out_var, width=64).grid(row=10, column=0, columnspan=4, sticky="we")

        btns = ttk.Frame(frm); btns.grid(row=11, column=0, columnspan=4, pady=10, sticky="we")
        ttk.Button(btns, text="Generate", command=self.on_generate).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        ttk.Label(frm, text="Log:").grid(row=12, column=0, sticky="w")
        self.log = tk.Text(frm, width=86, height=10, wrap="word")
        self.log.grid(row=13, column=0, columnspan=4, sticky="nsew")
        frm.rowconfigure(13, weight=1)

        self.after(200, lambda: self.s_short.var.set(1))
        self.after(200, lambda: self.s_medium.var.set(1))
        self.after(200, lambda: self.s_long.var.set(1))

    def log_msg(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def on_generate(self):
        topic = (self.topic_var.get() or "").strip()
        if not topic:
            messagebox.showerror("Missing topic", "Please enter a topic area."); return
        if not os.environ.get("OPENAI_API_KEY"):
            messagebox.showerror("Missing API key", "Please set OPENAI_API_KEY in your environment."); return

        n_short, n_med, n_long = self.s_short.get(), self.s_medium.get(), self.s_long.get()
        if n_short + n_med + n_long == 0:
            messagebox.showerror("Nothing to do", "All counts are 0. Increase at least one."); return

        try:
            out_base = Path(self.out_var.get()).expanduser().resolve()
            out_base.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Output error", f"Cannot create output folder:\n{e}"); return

        model_pref = (self.model_var.get() or "gpt-5").strip()
        concurrency = max(1, min(8, int(self.conc_var.get())))

        def status_cb(m): self.log_msg(m)

        def _runner():
            asyncio.run(generate_batch(
                topic=topic, n_short=n_short, n_med=n_med, n_long=n_long,
                base_out=out_base, model_pref=model_pref, concurrency=concurrency, status_cb=status_cb
            ))
        threading.Thread(target=_runner, daemon=True).start()
        self.log_msg(f"Started: '{topic}' with S/M/L = {n_short}/{n_med}/{n_long} (parallel={concurrency})")

# --------- main ---------
if __name__ == "__main__":
    app = App()
    app.mainloop()
