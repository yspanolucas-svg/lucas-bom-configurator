import io
import re
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


# =========================================================
# DATA MODEL
# =========================================================
@dataclass
class PDFRow:
    key: str
    label: str
    value: str
    source: str
    section: str  # "main"


# =========================================================
# HELPERS
# =========================================================
def _hash_bytes(b: Optional[bytes]) -> str:
    if not b:
        return ""
    return hashlib.sha1(b).hexdigest()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================
def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for p in reader.pages:
        chunks.append(p.extract_text() or "")
    txt = "\n".join(chunks)
    return txt.replace("\r", "\n")


def _parse_pairs_all(text: str) -> List[tuple]:
    """
    Parse table-like lines.
    Handles:
      - "Label  Value" (double spaces)
      - "Robot weight 250.0 kg" (single-space export)
      - fallback: last token as value
    """
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Prefer: "Label  Value"
        m = re.match(r"^(.*?)[ ]{2,}(.+)$", line)
        if m:
            rows.append((m.group(1).strip(), m.group(2).strip()))
            continue

        # Common: label + numeric chunk at end
        m2 = re.match(r"^(.*?)([-+]?\d[\d.,]*\s*.*)$", line)
        if m2 and len(m2.group(1).strip()) >= 3:
            rows.append((m2.group(1).strip(), m2.group(2).strip()))
            continue

        # fallback: last token as value
        parts = line.split(" ")
        if len(parts) >= 2:
            rows.append((" ".join(parts[:-1]).strip(), parts[-1].strip()))
        else:
            rows.append((line, ""))

    return rows


def extract_pdf_rows(pdf_bytes: Optional[bytes], doc_id: str, source_display: str) -> List[PDFRow]:
    """
    IMPORTANT:
      - keys are stable and depend ONLY on doc_id + label + counter
      - source_display is cosmetic and can change without resetting checks
    """
    if not pdf_bytes:
        return []

    text = _pdf_text(pdf_bytes)
    pairs = _parse_pairs_all(text)

    rows: List[PDFRow] = []
    counter: Dict[str, int] = {}

    for label, value in pairs:
        label = (label or "").strip()
        value = (value or "").strip()
        if len(label) < 3:
            continue

        base = re.sub(r"\s+", "_", label.lower()).strip("_")
        counter[base] = counter.get(base, 0) + 1

        # STABLE key (no dependency on display name)
        key = f"{doc_id}::{base}::{counter[base]}"

        rows.append(PDFRow(
            key=key,
            label=label,
            value=value,
            source=source_display,
            section="main",
        ))

    return rows


# =========================================================
# PDF GENERATION (REPORTLAB) - includes page2+ visibility fix
# =========================================================
def _draw_header(c: canvas.Canvas, title: str = "") -> float:
    w, h = A4

    # Black band
    c.setFillColorRGB(0.0, 0.0, 0.0)
    c.rect(0, h - 22 * mm, w, 22 * mm, stroke=0, fill=1)

    # Red accent
    c.setFillColorRGB(0.85, 0.0, 0.0)
    c.rect(0, h - 22 * mm, w, 3.5 * mm, stroke=0, fill=1)

    # White header text
    c.setFillColorRGB(1.0, 1.0, 1.0)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(12 * mm, h - 14 * mm, "LUCAS ROBOTIC SYSTEM")

    c.setFont("Helvetica", 10)
    right = "TECHNICAL DATA SHEET"
    if title:
        right = f"{right}  –  {title}"
    c.drawRightString(w - 12 * mm, h - 14 * mm, right)

    return h - 28 * mm


def _wrap_simple(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_table_header(c: canvas.Canvas, col_label: float, col_value: float, col_source: float, y: float) -> float:
    # CRITICAL: reset to black text after header (header uses white)
    c.setFillColorRGB(0.0, 0.0, 0.0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_label, y, "Variable")
    c.drawString(col_value, y, "Value")
    c.drawString(col_source, y, "Source")
    y -= 4 * mm

    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setLineWidth(0.4)
    c.line(col_label, y, (A4[0] - 12 * mm), y)
    y -= 4 * mm

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.0, 0.0, 0.0)
    return y


def _draw_table(c: canvas.Canvas, rows: List[PDFRow], y_start: float) -> float:
    x = 12 * mm
    w = A4[0] - 24 * mm
    y = y_start

    col_label = x
    col_value = x + w * 0.58
    col_source = x + w * 0.85

    y = _draw_table_header(c, col_label, col_value, col_source, y)

    for r in rows:
        lab_lines = _wrap_simple(r.label, 55)
        val_lines = _wrap_simple(r.value, 30)
        n = max(len(lab_lines), len(val_lines), 1)
        row_h = n * 4.2 * mm

        if y - row_h < 15 * mm:
            c.showPage()
            y = _draw_header(c, title="")
            y = _draw_table_header(c, col_label, col_value, col_source, y)

        c.setFillColorRGB(0.0, 0.0, 0.0)
        c.setFont("Helvetica", 9)

        for i in range(n):
            c.drawString(col_label, y - i * 4.2 * mm, lab_lines[i] if i < len(lab_lines) else "")
            c.drawString(col_value, y - i * 4.2 * mm, val_lines[i] if i < len(val_lines) else "")
            if i == 0:
                c.drawString(col_source, y, r.source)

        y -= row_h + 1.5 * mm

    return y


def build_output_pdf(selected_rows: List[PDFRow]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = _draw_header(c, title="")
    _draw_table(c, selected_rows, y)
    c.save()
    return buf.getvalue()


# =========================================================
# STREAMLIT UI
# =========================================================
def render_pdf_lab_panel():
    st.subheader("PDF Lab – Clean / Assemble (generic)")

    mode = st.radio(
        "Mode",
        ["Clean (1 PDF)", "Assemble (2 PDFs)"],
        horizontal=True,
        key="pdf_mode",
    )

    coln1, coln2 = st.columns(2)
    with coln1:
        src1_name = st.text_input("Document 1 name", value="Ensemble 1", key="src1_name")
    with coln2:
        src2_name = st.text_input("Document 2 name", value="Ensemble 2", key="src2_name")

    if mode == "Clean (1 PDF)":
        pdf1 = st.file_uploader("PDF – Document 1", type=["pdf"], key="pdf1")
        pdf2 = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            pdf1 = st.file_uploader("PDF – Document 1", type=["pdf"], key="pdf1")
        with c2:
            pdf2 = st.file_uploader("PDF – Document 2", type=["pdf"], key="pdf2")

    if not pdf1:
        st.info("Upload at least 1 PDF to continue.")
        return
    if mode == "Assemble (2 PDFs)" and not pdf2:
        st.info("Upload the second PDF to assemble.")
        return

    b1 = pdf1.read()
    b2 = pdf2.read() if pdf2 else None

    # Stable doc ids based ONLY on pdf hash
    h1 = _hash_bytes(b1)
    h2 = _hash_bytes(b2)
    doc1_id = f"doc1::{h1}"
    doc2_id = f"doc2::{h2}" if b2 else ""

    # session keys depend ONLY on pdf hash (never on display names)
    rows1_key = f"pdf_lab::rows::{doc1_id}"
    df1_key = f"pdf_lab::df::{doc1_id}"

    rows2_key = f"pdf_lab::rows::{doc2_id}"
    df2_key = f"pdf_lab::df::{doc2_id}"

    # Initialize doc1 once per PDF
    if rows1_key not in st.session_state:
        rows1 = extract_pdf_rows(b1, doc_id=doc1_id, source_display=src1_name)
        st.session_state[rows1_key] = rows1
        st.session_state[df1_key] = pd.DataFrame([{
            "Keep": False,
            "Label": r.label,
            "Value": r.value,
            "Source": r.source,
            "_key": r.key,
        } for r in rows1])

    # Initialize doc2 once per PDF
    if b2 and rows2_key not in st.session_state:
        rows2 = extract_pdf_rows(b2, doc_id=doc2_id, source_display=src2_name)
        st.session_state[rows2_key] = rows2
        st.session_state[df2_key] = pd.DataFrame([{
            "Keep": False,
            "Label": r.label,
            "Value": r.value,
            "Source": r.source,
            "_key": r.key,
        } for r in rows2])

    df1 = st.session_state[df1_key]
    rows1 = st.session_state[rows1_key]

    if b2:
        df2 = st.session_state[df2_key]
        rows2 = st.session_state[rows2_key]
    else:
        df2 = pd.DataFrame(columns=df1.columns)
        rows2 = []

    # Cosmetic rename must update "Source" column + rows[].source without resetting checks
    df1["Source"] = src1_name
    for r in rows1:
        r.source = src1_name

    if b2:
        df2["Source"] = src2_name
        for r in rows2:
            r.source = src2_name

    st.info(f"Detected rows – {src1_name}: {len(df1)} | {src2_name}: {len(df2) if b2 else 0}")

    search = st.text_input("Search (Label/Value):", value="", key="pdf_search")

    tab_titles = [f"Document 1 – {src1_name}"]
    if b2:
        tab_titles.append(f"Document 2 – {src2_name}")
    tabs = st.tabs(tab_titles)

    def _apply_editor_changes(df: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
        if edited is None or edited.empty:
            return df

        ed_map = {}
        for _, r in edited.iterrows():
            ed_map[str(r["_key"])] = {"Keep": bool(r["Keep"]), "Value": str(r["Value"])}

        for i in range(len(df)):
            k = str(df.at[i, "_key"])
            if k in ed_map:
                df.at[i, "Keep"] = ed_map[k]["Keep"]
                df.at[i, "Value"] = ed_map[k]["Value"]
        return df

    def editor(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
        show_only_checked = st.toggle(
            "Show only checked (this document)",
            value=False,
            key=f"{key_prefix}_show_checked",
        )

        # view from df
        df_view = df.copy()

        if search.strip():
            s = search.strip().lower()
            df_view = df_view[
                df_view["Label"].str.lower().str.contains(s, na=False)
                | df_view["Value"].astype(str).str.lower().str.contains(s, na=False)
            ]

        if show_only_checked:
            df_view = df_view[df_view["Keep"] == True]
            if df_view.empty:
                st.info("No checked rows in this document yet.")
                return df

        # Buttons apply to filtered view
        cA, cB, _ = st.columns([1, 1, 6])
        with cA:
            do_check_all = st.button("Check all (filtered)", key=f"{key_prefix}_all_on")
        with cB:
            do_uncheck_all = st.button("Uncheck all (filtered)", key=f"{key_prefix}_all_off")

        if do_check_all:
            df.loc[df_view.index, "Keep"] = True
        if do_uncheck_all:
            df.loc[df_view.index, "Keep"] = False

        # rebuild view after button actions
        df_view = df.copy()
        if search.strip():
            s = search.strip().lower()
            df_view = df_view[
                df_view["Label"].str.lower().str.contains(s, na=False)
                | df_view["Value"].astype(str).str.lower().str.contains(s, na=False)
            ]
        if show_only_checked:
            df_view = df_view[df_view["Keep"] == True]
            if df_view.empty:
                st.info("No checked rows in this document yet.")
                return df

        # force rerender on toggle state
        editor_key = f"{key_prefix}_editor__{'checked' if show_only_checked else 'all'}"

        edited = st.data_editor(
            df_view[["Keep", "Label", "Value", "Source", "_key"]],
            use_container_width=True,
            hide_index=True,
            height=900,
            column_config={
                "Keep": st.column_config.CheckboxColumn("Keep", default=False, width="small"),
                "Label": st.column_config.TextColumn("Label", width="large"),
                "Value": st.column_config.TextColumn("Value", width="large"),
                "Source": st.column_config.TextColumn("Source", width="medium"),
                "_key": st.column_config.TextColumn("_key", width="small"),
            },
            key=editor_key,
        )

        df = _apply_editor_changes(df, edited)
        return df

    with tabs[0]:
        df1 = editor(df1, "doc1")
        st.session_state[df1_key] = df1

    if b2:
        with tabs[1]:
            df2 = editor(df2, "doc2")
            st.session_state[df2_key] = df2

    if st.button("Generate output PDF", type="primary", key="btn_generate_pdf"):
        selected: List[PDFRow] = []

        def collect(df: pd.DataFrame, rows: List[PDFRow]):
            rows_map = {r.key: r for r in rows}
            checked = df[df["Keep"] == True]
            for _, r in checked.iterrows():
                obj = rows_map.get(str(r["_key"]))
                if obj:
                    obj.value = str(r["Value"])
                    obj.source = str(r["Source"])
                    selected.append(obj)

        collect(df1, rows1)
        collect(df2, rows2)

        if not selected:
            st.warning("No rows selected: please check at least one variable.")
            return

        out = build_output_pdf(selected)

        st.download_button(
            "💾 Download PDF",
            data=out,
            file_name="TECHNICAL_SHEET.pdf",
            mime="application/pdf",
        )
