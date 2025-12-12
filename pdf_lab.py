import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ============================================================
#  FONT (robust rendering)
#  - Prefer repo fonts in assets/fonts
#  - Fallback to system DejaVu
#  - Final fallback to Helvetica
# ============================================================

def _try_register_font(font_name: str, path: str) -> bool:
    try:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(font_name, path))
            return True
    except Exception:
        pass
    return False


def _ensure_fonts() -> Tuple[str, str]:
    """
    Returns (regular_font, bold_font). Falls back safely.
    """
    # 1) Repo fonts
    repo_reg = os.path.join("assets", "fonts", "DejaVuSans.ttf")
    repo_bold = os.path.join("assets", "fonts", "DejaVuSans-Bold.ttf")

    # 2) System fonts (common on Linux)
    sys_reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    sys_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    reg_ok = _try_register_font("LucasFont", repo_reg) or _try_register_font("LucasFont", sys_reg)
    bold_ok = _try_register_font("LucasFontBold", repo_bold) or _try_register_font("LucasFontBold", sys_bold)

    if reg_ok and bold_ok:
        return "LucasFont", "LucasFontBold"
    if reg_ok:
        return "LucasFont", "LucasFont"
    return "Helvetica", "Helvetica-Bold"


REG_FONT, BOLD_FONT = _ensure_fonts()


# ============================================================
#  EXTRACT
# ============================================================

@dataclass
class PDFRow:
    key: str
    label: str
    value: str
    source: str
    section: str  # "main" / "options" / "other"


MOTOR_BLACKLIST = [
    "motor", "moteur",
    "torque", "couple",
    "brake", "frein",
    "inertia", "inertie",
    "required motor speed", "vitesse moteur",
    "emergency stop", "arrêt d'urgence", "arret d'urgence",
    "nominal torque",
    "kg.cm", "kgcm",
]
BOM_BLACKLIST = [
    "bill of material", "product bill of material", "bom", "nomenclature",
]
ILLUSTRATION_BLACKLIST = [
    "illustration", "illustrations", "photo", "image", "images",
]

def _is_blacklisted(label: str) -> bool:
    s = (label or "").lower()
    return (
        any(k in s for k in MOTOR_BLACKLIST)
        or any(k in s for k in BOM_BLACKLIST)
        or any(k in s for k in ILLUSTRATION_BLACKLIST)
    )


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: List[str] = []
    for p in reader.pages:
        out.append(p.extract_text() or "")
    txt = "\n".join(out).replace("\r", "\n")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt


def _find_section_bounds(text: str, start_patterns: List[str], end_patterns: List[str]) -> Optional[Tuple[int, int]]:
    start_idx = None
    for pat in start_patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            start_idx = m.end()
            break
    if start_idx is None:
        return None

    end_idx = len(text)
    for pat in end_patterns:
        m = re.search(pat, text[start_idx:], flags=re.I)
        if m:
            end_idx = start_idx + m.start()
            break
    return (start_idx, end_idx)


def _canon(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "")).strip()


def _parse_table_like_lines(block: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue

        # skip obvious headers
        if re.match(r"^(Variable|Option / Condition|Valeur|Value|Source)\b", line, flags=re.I):
            continue

        # prefer 2+ spaces split
        if re.search(r"\s{2,}", line):
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) >= 2:
                rows.append((parts[0], " ".join(parts[1:])))
                continue

        # fallback: label then rest starting with number or +/-
        m = re.match(r"^(.*?)(\s[-+0-9].*)$", line)
        if m:
            rows.append((m.group(1).strip().rstrip(":"), m.group(2).strip()))
        else:
            # keep as label only
            rows.append((line.rstrip(":"), ""))

    return rows


def detect_language(text: str) -> str:
    # Tiny heuristic: only used for header subtitle, not translating content
    if re.search(r"\bMain characteristics\b", text, flags=re.I):
        return "EN"
    if re.search(r"\bOptions\s*&\s*Conditions\b", text, flags=re.I):
        return "EN"
    if re.search(r"\bCaract[ée]ristiques\b", text, flags=re.I):
        return "FR"
    return "EN"


def extract_pdf_rows(pdf_bytes: bytes, source_label: str) -> List[PDFRow]:
    text = _pdf_text(pdf_bytes)

    # We keep sections flexible / generic (no hard assumptions on headings)
    main_bounds = _find_section_bounds(
        text,
        start_patterns=[r"\bMain characteristics\b", r"\bGeneral characteristics\b", r"\bCaract[ée]ristiques\b"],
        end_patterns=[r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b", r"\bConditions\b"],
    )
    opt_bounds = _find_section_bounds(
        text,
        start_patterns=[r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b"],
        end_patterns=[r"$"],
    )

    counters: Dict[str, int] = {}
    out: List[PDFRow] = []

    def add_block(block: str, section: str):
        nonlocal out, counters
        for label, value in _parse_table_like_lines(block):
            c = _canon(label)
            if not c:
                continue
            if _is_blacklisted(c):
                continue
            counters[c] = counters.get(c, 0) + 1
            key = f"{source_label}::{section}::{c}::{counters[c]}"
            out.append(PDFRow(key=key, label=label, value=value, source=source_label, section=section))

    if main_bounds:
        add_block(text[main_bounds[0]:main_bounds[1]].strip(), "main")
    if opt_bounds:
        add_block(text[opt_bounds[0]:opt_bounds[1]].strip(), "options")

    # If nothing extracted, fallback to any "label  value" lines
    if not out:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) < 2:
                continue
            label, value = parts[0], " ".join(parts[1:])
            c = _canon(label)
            if not c or _is_blacklisted(c):
                continue
            counters[c] = counters.get(c, 0) + 1
            key = f"{source_label}::fallback::{c}::{counters[c]}"
            out.append(PDFRow(key=key, label=label, value=value, source=source_label, section="other"))

    return out


# ============================================================
#  BUILD PDF (selected rows only, no auto-merge)
# ============================================================

def _draw_header(c: canvas.Canvas, lang: str, title: str):
    w, h = A4
    # black band
    c.setFillGray(0.05)
    c.rect(0, h - 22 * mm, w, 22 * mm, stroke=0, fill=1)
    # red accent
    c.setFillRGB(0.85, 0.0, 0.0)
    c.rect(0, h - 22 * mm, w, 3.5 * mm, stroke=0, fill=1)

    c.setFillRGB(1, 1, 1)
    c.setFont(BOLD_FONT, 13)
    c.drawString(12 * mm, h - 14 * mm, "LUCAS ROBOTIC SYSTEM")

    c.setFont(REG_FONT, 10)
    subtitle = "TECHNICAL DATA SHEET" if lang == "EN" else "FICHE TECHNIQUE"
    c.drawRightString(w - 12 * mm, h - 14 * mm, subtitle)

    c.setFont(BOLD_FONT, 11)
    c.drawString(12 * mm, h - 30 * mm, title)


def _wrap(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_table(c: canvas.Canvas, x: float, y: float, w: float, rows: List[PDFRow], lang: str, title: str) -> float:
    c.setFont(BOLD_FONT, 11)
    c.setFillGray(0)
    c.drawString(x, y, title)
    y -= 6 * mm

    col1 = x
    col2 = x + w * 0.58
    col3 = x + w * 0.86

    c.setFont(BOLD_FONT, 9)
    c.drawString(col1, y, "Variable")
    c.drawString(col2, y, "Value" if lang == "EN" else "Valeur")
    c.drawString(col3, y, "Source")
    y -= 3.5 * mm
    c.setLineWidth(0.3)
    c.setStrokeGray(0.7)
    c.line(x, y, x + w, y)
    y -= 3.5 * mm

    c.setFont(REG_FONT, 9)

    for r in rows:
        lab_lines = _wrap(r.label, 55)
        val_lines = _wrap(r.value, 40)
        n = max(len(lab_lines), len(val_lines))
        row_h = n * 4.2 * mm

        if y - row_h < 15 * mm:
            c.showPage()
            _draw_header(c, lang, title="")  # keep band only on new page
            y = A4[1] - 30 * mm
            # redraw table header on new page
            c.setFont(BOLD_FONT, 11)
            c.drawString(x, y, title)
            y -= 6 * mm
            c.setFont(BOLD_FONT, 9)
            c.drawString(col1, y, "Variable")
            c.drawString(col2, y, "Value" if lang == "EN" else "Valeur")
            c.drawString(col3, y, "Source")
            y -= 3.5 * mm
            c.setStrokeGray(0.7)
            c.line(x, y, x + w, y)
            y -= 3.5 * mm
            c.setFont(REG_FONT, 9)

        for i in range(n):
            c.drawString(col1, y - i * 4.2 * mm, lab_lines[i] if i < len(lab_lines) else "")
            c.drawString(col2, y - i * 4.2 * mm, val_lines[i] if i < len(val_lines) else "")
            if i == 0:
                c.drawString(col3, y, r.source)

        y -= row_h + 1.5 * mm

    return y


def build_output_pdf(
    pdf1_bytes: bytes,
    pdf2_bytes: Optional[bytes],
    source1_name: str,
    source2_name: str,
    keep_keys: List[str],
    overrides: Dict[str, str],
) -> bytes:
    rows_1 = extract_pdf_rows(pdf1_bytes, source_label=source1_name)
    rows_2 = extract_pdf_rows(pdf2_bytes, source_label=source2_name) if pdf2_bytes else []
    all_rows = rows_1 + rows_2

    # apply overrides first
    if overrides:
        for r in all_rows:
            if r.key in overrides:
                r.value = overrides[r.key]

    keep_set = set(keep_keys)
    kept = [r for r in all_rows if r.key in keep_set]

    # If user didn't select anything, output empty but valid PDF with header
    lang = detect_language(_pdf_text(pdf1_bytes))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _draw_header(c, lang, title="")

    x = 12 * mm
    y = A4[1] - 40 * mm
    w = A4[0] - 24 * mm

    # split by section for readability
    main_rows = [r for r in kept if r.section == "main"]
    opt_rows = [r for r in kept if r.section == "options"]
    other_rows = [r for r in kept if r.section not in ("main", "options")]

    if main_rows:
        y = _draw_table(c, x, y, w, main_rows, lang, "Main characteristics" if lang == "EN" else "Caractéristiques principales")
        y -= 2 * mm
    if opt_rows:
        y = _draw_table(c, x, y, w, opt_rows, lang, "Options & Conditions")
        y -= 2 * mm
    if other_rows:
        _draw_table(c, x, y, w, other_rows, lang, "Other")

    c.save()
    return buf.getvalue()


# ============================================================
#  STREAMLIT UI
# ============================================================

def render_pdf_lab_panel():
    st.subheader("📄 PDF Lab – Clean / Merge (generic)")

    mode = st.radio(
        "Mode",
        ["Clean (1 PDF)", "Merge (2 PDFs)"],
        horizontal=True,
        key="pdf_mode",
    )

    # Editable generic names
    n1, n2 = st.columns(2)
    with n1:
        src1_name = st.text_input("Document 1 name", value="Ensemble 1", key="src1_name")
    with n2:
        src2_name = st.text_input("Document 2 name", value="Ensemble 2", key="src2_name")

    if mode == "Clean (1 PDF)":
        pdf1 = st.file_uploader("PDF – Document 1", type=["pdf"], key="pdf_single")
        pdf2 = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            pdf1 = st.file_uploader("PDF – Document 1", type=["pdf"], key="pdf_1")
        with c2:
            pdf2 = st.file_uploader("PDF – Document 2", type=["pdf"], key="pdf_2")

    if not pdf1:
        st.info("Charge at least one PDF to continue.")
        return
    if mode == "Merge (2 PDFs)" and not pdf2:
        st.info("Charge the second PDF to merge.")
        return

    b1 = pdf1.read()
    b2 = pdf2.read() if pdf2 else None

    rows_1 = extract_pdf_rows(b1, source_label=src1_name)
    rows_2 = extract_pdf_rows(b2, source_label=src2_name) if b2 else []

    st.info(f"Detected rows – {src1_name}: {len(rows_1)} | {src2_name}: {len(rows_2)}")

    df1 = pd.DataFrame([{
        "Keep": False,  # default OFF
        "Label": r.label,
        "Value": r.value,
        "Section": r.section,
        "_key": r.key,
    } for r in rows_1])

    df2 = pd.DataFrame([{
        "Keep": False,
        "Label": r.label,
        "Value": r.value,
        "Section": r.section,
        "_key": r.key,
    } for r in rows_2])

    search = st.text_input("Search (Label/Value):", value="", key="pdf_search")

    tab1, tab2 = st.tabs([f"Document 1 – {src1_name}", f"Document 2 – {src2_name}"])

    def editor(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
        show_only_checked = st.toggle("Show only checked (this document)", value=False, key=f"{key_prefix}_soc")

        df_view = df.copy()
        if search.strip():
            s = search.strip().lower()
            df_view = df_view[
                df_view["Label"].str.lower().str.contains(s, na=False)
                | df_view["Value"].astype(str).str.lower().str.contains(s, na=False)
            ]
        if show_only_checked:
            df_view = df_view[df_view["Keep"] == True]

        b_on, b_off, _ = st.columns([1, 1, 3])
        with b_on:
            if st.button("Check all (filtered)", key=f"{key_prefix}_all_on"):
                df.loc[df_view.index, "Keep"] = True
                df_view["Keep"] = True
        with b_off:
            if st.button("Uncheck all (filtered)", key=f"{key_prefix}_all_off"):
                df.loc[df_view.index, "Keep"] = False
                df_view["Keep"] = False

        edited = st.data_editor(
            df_view[["Keep", "Label", "Value", "Section", "_key"]],
            use_container_width=True,
            hide_index=True,
            height=900,
            column_config={
                "Keep": st.column_config.CheckboxColumn("Keep", default=False, width="small"),
                "Label": st.column_config.TextColumn("Label", disabled=True, width="large"),
                "Value": st.column_config.TextColumn("Value", width="large"),
                "Section": st.column_config.TextColumn("Section", disabled=True, width="small"),
                "_key": st.column_config.TextColumn("_key", disabled=True, width="small"),
            },
            key=f"{key_prefix}_editor",
        )

        # Apply edits back to full df by key
        ed_map = {row["_key"]: row for _, row in edited.iterrows()}
        for i in range(len(df)):
            k = df.at[i, "_key"]
            if k in ed_map:
                df.at[i, "Keep"] = bool(ed_map[k]["Keep"])
                df.at[i, "Value"] = str(ed_map[k]["Value"])

        return df

    with tab1:
        df1 = editor(df1, "doc1")

    with tab2:
        if b2:
            df2 = editor(df2, "doc2")
        else:
            st.info("No Document 2 in Clean mode.")

    # Generate
    if st.button("Generate output PDF", type="primary", key="btn_generate_pdf"):
        df_all = pd.concat([df1, df2], ignore_index=True)

        keep_keys = df_all[df_all["Keep"]]["_key"].tolist()
        overrides = {row["_key"]: str(row["Value"]) for _, row in df_all.iterrows()}

        out = build_output_pdf(
            pdf1_bytes=b1,
            pdf2_bytes=b2,
            source1_name=src1_name,
            source2_name=src2_name,
            keep_keys=keep_keys,
            overrides=overrides,
        )

        st.download_button(
            "💾 Download PDF",
            data=out,
            file_name="TECHNICAL_SHEET.pdf",
            mime="application/pdf",
        )
