import io
import re
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
    section: str  # "main" (on reste simple pour l’instant)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================
def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for p in reader.pages:
        chunks.append(p.extract_text() or "")
    txt = "\n".join(chunks)
    txt = txt.replace("\r", "\n")
    return txt


def _parse_pairs_all(text: str) -> List[tuple]:
    """
    Parse large tables that appear as:
        Robot brand ABB
        Maximum speed on the y-axis 0.5 m/s
    Many PDFs from CADENAS export lines as single-space separated.
    We split into (label, value) by taking last "value-ish" chunk.
    Fallback: try double-spaces first.
    """
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # prefer "Label  Value"
        m = re.match(r"^(.*?)[ ]{2,}(.+)$", line)
        if m:
            rows.append((m.group(1).strip(), m.group(2).strip()))
            continue

        # else: attempt split by last numeric/unit occurrence
        # Example: "Robot weight 250.0 kg" => label="Robot weight", value="250.0 kg"
        m2 = re.match(r"^(.*?)([-+]?\d[\d.,]*\s*.*)$", line)
        if m2 and len(m2.group(1).strip()) >= 3:
            rows.append((m2.group(1).strip(), m2.group(2).strip()))
            continue

        # else: try split at last word (not perfect, but better than dropping)
        parts = line.split(" ")
        if len(parts) >= 2:
            rows.append((" ".join(parts[:-1]).strip(), parts[-1].strip()))
        else:
            rows.append((line, ""))

    return rows


def extract_pdf_rows(pdf_bytes: bytes, source_label: str) -> List[PDFRow]:
    """
    IMPORTANT: on ne filtre rien -> tout remonte.
    Tout est décoché par défaut côté UI.
    """
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
        key = f"{source_label}::{base}::{counter[base]}"

        rows.append(PDFRow(
            key=key,
            label=label,
            value=value,
            source=source_label,
            section="main",
        ))

    return rows


# =========================================================
# PDF GENERATION (REPORTLAB)
# =========================================================
def _draw_header(c: canvas.Canvas, title: str = "") -> float:
    w, h = A4

    # Black band
    c.setFillColorRGB(0.0, 0.0, 0.0)
    c.rect(0, h - 22 * mm, w, 22 * mm, stroke=0, fill=1)

    # Red accent
    c.setFillColorRGB(0.85, 0.0, 0.0)
    c.rect(0, h - 22 * mm, w, 3.5 * mm, stroke=0, fill=1)

    # White text
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
    lines = []
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


def _draw_table(c: canvas.Canvas, rows: List[PDFRow], y_start: float) -> float:
    x = 12 * mm
    w = A4[0] - 24 * mm
    y = y_start

    col_label = x
    col_value = x + w * 0.58
    col_source = x + w * 0.85

    # Header row
    c.setFillColorRGB(0.0, 0.0, 0.0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_label, y, "Variable")
    c.drawString(col_value, y, "Value")
    c.drawString(col_source, y, "Source")
    y -= 4 * mm

    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setLineWidth(0.4)
    c.line(x, y, x + w, y)
    y -= 4 * mm

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.0, 0.0, 0.0)

    for r in rows:
        lab_lines = _wrap_simple(r.label, 55)
        val_lines = _wrap_simple(r.value, 30)
        n = max(len(lab_lines), len(val_lines), 1)
        row_h = n * 4.2 * mm

        if y - row_h < 15 * mm:
            c.showPage()
            y = _draw_header(c, title="")
            c.setFont("Helvetica-Bold", 9)
            c.drawString(col_label, y, "Variable")
            c.drawString(col_value, y, "Value")
            c.drawString(col_source, y, "Source")
            y -= 4 * mm
            c.setStrokeColorRGB(0.7, 0.7, 0.7)
            c.line(x, y, x + w, y)
            y -= 4 * mm
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

    # Source renaming (generic)
    coln1, coln2 = st.columns(2)
    with coln1:
        src1_name = st.text_input("Document 1 name", value="Ensemble 1", key="src1_name")
    with coln2:
        src2_name = st.text_input("Document 2 name", value="Ensemble 2", key="src2_name")

    # Uploads
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

    rows1 = extract_pdf_rows(b1, source_label=src1_name)
    rows2 = extract_pdf_rows(b2, source_label=src2_name) if b2 else []

    st.info(f"Detected rows – {src1_name}: {len(rows1)} | {src2_name}: {len(rows2)}")

    # DataFrames (all unchecked by default)
    df1 = pd.DataFrame([{
        "Keep": False,
        "Label": r.label,
        "Value": r.value,
        "Source": r.source,
        "_key": r.key,
    } for r in rows1])

    df2 = pd.DataFrame([{
        "Keep": False,
        "Label": r.label,
        "Value": r.value,
        "Source": r.source,
        "_key": r.key,
    } for r in rows2])

    search = st.text_input("Search (Label/Value):", value="", key="pdf_search")

    tab_titles = [f"Document 1 – {src1_name}"]
    if b2:
        tab_titles.append(f"Document 2 – {src2_name}")
    tabs = st.tabs(tab_titles)

    def editor(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
        show_only_checked = st.toggle(
            "Show only checked (this document)",
            value=False,
            key=f"{key_prefix}_show_checked"
        )

        df_view = df.copy()

        if search.strip():
            s = search.strip().lower()
            df_view = df_view[
                df_view["Label"].str.lower().str.contains(s, na=False)
                | df_view["Value"].astype(str).str.lower().str.contains(s, na=False)
            ]

        if show_only_checked:
            df_view = df_view[df_view["Keep"] == True]

        cA, cB, cC = st.columns([1, 1, 3])
        with cA:
            if st.button("Check all (filtered)", key=f"{key_prefix}_all_on"):
                df.loc[df_view.index, "Keep"] = True
                df_view["Keep"] = True
        with cB:
            if st.button("Uncheck all (filtered)", key=f"{key_prefix}_all_off"):
                df.loc[df_view.index, "Keep"] = False
                df_view["Keep"] = False

        edited = st.data_editor(
            df_view[["Keep", "Label", "Value", "Source", "_key"]],
            use_container_width=True,
            hide_index=True,
            height=900,
            column_config={
                "Keep": st.column_config.CheckboxColumn("Keep", default=False, width="small"),
                "Label": st.column_config.TextColumn("Label", disabled=True, width="large"),
                "Value": st.column_config.TextColumn("Value", width="large"),
                "Source": st.column_config.TextColumn("Source", disabled=True, width="medium"),
                "_key": st.column_config.TextColumn("_key", disabled=True, width="small"),
            },
            key=f"{key_prefix}_editor",
        )

        # reinject edits into original df via _key
        ed_map = {row["_key"]: row for _, row in edited.iterrows()}
        for i in range(len(df)):
            k = df.at[i, "_key"]
            if k in ed_map:
                df.at[i, "Keep"] = bool(ed_map[k]["Keep"])
                df.at[i, "Value"] = str(ed_map[k]["Value"])

        return df

    with tabs[0]:
        df1 = editor(df1, "doc1")

    if b2:
        with tabs[1]:
            df2 = editor(df2, "doc2")

    # Generate
    if st.button("Generate output PDF", type="primary", key="btn_generate_pdf"):
        selected: List[PDFRow] = []

        # Apply overrides & select by Keep
        def collect(df: pd.DataFrame, rows: List[PDFRow]):
            rows_map = {r.key: r for r in rows}
            for _, r in df.iterrows():
                if bool(r["Keep"]):
                    obj = rows_map.get(r["_key"])
                    if obj:
                        obj.value = str(r["Value"])
                        selected.append(obj)

        collect(df1, rows1)
        collect(df2, rows2)

        out = build_output_pdf(selected)
        st.download_button(
            "💾 Download PDF",
            data=out,
            file_name="TECHNICAL_SHEET.pdf",
            mime="application/pdf",
        )
