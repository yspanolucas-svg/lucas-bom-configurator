import io
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


# =========================================================
# DATA STRUCTURE
# =========================================================

@dataclass
class PDFRow:
    key: str
    label: str
    value: str
    source: str
    section: str


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text)


def parse_lines(text: str) -> List[tuple]:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # format: "Label  Value"
        m = re.match(r"^(.*?)[ ]{2,}(.+)$", line)
        if m:
            rows.append((m.group(1).strip(), m.group(2).strip()))
        else:
            # fallback: keep label only
            rows.append((line, ""))
    return rows


def extract_pdf_rows(pdf_bytes: bytes, source_label: str) -> List[PDFRow]:
    text = extract_pdf_text(pdf_bytes)
    pairs = parse_lines(text)

    rows = []
    counter = {}

    for label, value in pairs:
        if len(label) < 3:
            continue

        key_base = re.sub(r"\s+", "_", label.lower())
        counter[key_base] = counter.get(key_base, 0) + 1
        key = f"{source_label}::{key_base}::{counter[key_base]}"

        rows.append(
            PDFRow(
                key=key,
                label=label,
                value=value,
                source=source_label,
                section="main",
            )
        )
    return rows


# =========================================================
# PDF GENERATION
# =========================================================

def draw_header(c: canvas.Canvas):
    w, h = A4
    c.setFillColorRGB(0, 0, 0)
    c.rect(0, h - 22 * mm, w, 22 * mm, stroke=0, fill=1)
    c.setFillColorRGB(0.85, 0.0, 0.0)
    c.rect(0, h - 22 * mm, w, 3.5 * mm, stroke=0, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(12 * mm, h - 14 * mm, "LUCAS ROBOTIC SYSTEM")

    c.setFont("Helvetica", 10)
    c.drawRightString(w - 12 * mm, h - 14 * mm, "TECHNICAL DATA SHEET")

    return h - 28 * mm


def draw_table(c, rows: List[PDFRow], start_y):
    x = 12 * mm
    w = A4[0] - 24 * mm
    y = start_y

    col_label = x
    col_value = x + w * 0.6
    col_source = x + w * 0.85

    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_label, y, "Variable")
    c.drawString(col_value, y, "Value")
    c.drawString(col_source, y, "Source")
    y -= 4 * mm

    c.line(x, y, x + w, y)
    y -= 4 * mm

    c.setFont("Helvetica", 9)

    for r in rows:
        if y < 15 * mm:
            c.showPage()
            y = draw_header(c)
            c.setFont("Helvetica", 9)

        c.drawString(col_label, y, r.label)
        c.drawString(col_value, y, r.value)
        c.drawString(col_source, y, r.source)
        y -= 4 * mm

    return y


def build_output_pdf(
    pdf1_bytes: bytes,
    pdf2_bytes: Optional[bytes],
    rows: List[PDFRow],
):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    y = draw_header(c)
    y = draw_table(c, rows, y)

    c.save()
    return buf.getvalue()


# =========================================================
# STREAMLIT UI
# =========================================================

def render_pdf_lab_panel():
    st.subheader("PDF Lab – Generic clean / assembly")

    mode = st.radio(
        "Mode",
        ["Clean (1 PDF)", "Assemble (2 PDFs)"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        src1_name = st.text_input("Document 1 name", "Ensemble 1")
    with col2:
        src2_name = st.text_input("Document 2 name", "Ensemble 2")

    if mode == "Clean (1 PDF)":
        pdf1 = st.file_uploader("PDF – Document 1", type=["pdf"])
        pdf2 = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            pdf1 = st.file_uploader("PDF – Document 1", type=["pdf"])
        with c2:
            pdf2 = st.file_uploader("PDF – Document 2", type=["pdf"])

    if not pdf1:
        st.info("Upload at least one PDF")
        return

    b1 = pdf1.read()
    rows1 = extract_pdf_rows(b1, src1_name)

    rows2 = []
    if pdf2:
        b2 = pdf2.read()
        rows2 = extract_pdf_rows(b2, src2_name)
    else:
        b2 = None

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

    search = st.text_input("Search (label / value)")

    def editor(df, key):
        df_view = df.copy()
        if search:
            s = search.lower()
            df_view = df_view[
                df_view["Label"].str.lower().str.contains(s, na=False)
                | df_view["Value"].astype(str).str.lower().str.contains(s, na=False)
            ]

        edited = st.data_editor(
            df_view,
            height=900,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Keep": st.column_config.CheckboxColumn("Keep", default=False),
                "Label": st.column_config.TextColumn("Label", disabled=True),
                "Value": st.column_config.TextColumn("Value"),
                "Source": st.column_config.TextColumn("Source", disabled=True),
                "_key": st.column_config.TextColumn("_key", disabled=True),
            },
            key=key,
        )

        for i in range(len(df)):
            k = df.at[i, "_key"]
            row = edited[edited["_key"] == k]
            if not row.empty:
                df.at[i, "Keep"] = bool(row.iloc[0]["Keep"])
                df.at[i, "Value"] = str(row.iloc[0]["Value"])
        return df

    tabs = ["Document 1"]
    if pdf2:
        tabs.append("Document 2")

    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        df1 = editor(df1, "doc1")

    if pdf2:
        with tab_objs[1]:
            df2 = editor(df2, "doc2")

    if st.button("Generate PDF", type="primary"):
        rows_out = []

        for df, rows in [(df1, rows1), (df2, rows2)]:
            for _, r in df.iterrows():
                if r["Keep"]:
                    match = next(x for x in rows if x.key == r["_key"])
                    match.value = r["Value"]
                    rows_out.append(match)

        out = build_output_pdf(b1, b2, rows_out)

        st.download_button(
            "Download PDF",
            data=out,
            file_name="TECHNICAL_SHEET.pdf",
            mime="application/pdf",
        )
