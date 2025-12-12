import io
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


# =========================
#  PDF LAB - EXTRACTION
# =========================

@dataclass
class PDFRow:
    key: str          # internal unique key (stable in app run)
    label: str
    value: str
    source: str       # "Ensemble X" / "Ensemble YZ"
    section: str      # "main" or "options"


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        chunks.append(t)
    text = "\n".join(chunks)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    # Very small heuristic: keep output language aligned with source
    if re.search(r"\bMain characteristics\b", text, flags=re.I):
        return "EN"
    if re.search(r"\bOptions\s*&\s*Conditions\b", text, flags=re.I):
        return "EN"
    return "FR"


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


def parse_table_like_lines(block: str) -> List[Tuple[str, str]]:
    """
    Parses lines like:
      "Useful stroke X 10000 mm"
      "Place of delivery and installation France"
    Heuristic split: double spaces OR "label + rest".
    """
    rows: List[Tuple[str, str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue

        # skip table headers
        if re.match(r"^(Variable|Option / Condition)\b", line, flags=re.I):
            continue
        if re.match(r"^(Valeur|Value)\b", line, flags=re.I):
            continue
        if re.match(r"^(Source)\b", line, flags=re.I):
            continue

        # prefer "  " split
        if "  " in line:
            parts = [p.strip() for p in line.split("  ") if p.strip()]
            if len(parts) >= 2:
                label = parts[0]
                value = " ".join(parts[1:])
                rows.append((label, value))
                continue

        # fallback: "label" then "rest"
        m = re.search(r"^(.*?)(\s[-+0-9].*)$", line)
        if m:
            label = m.group(1).strip()
            value = m.group(2).strip()
            rows.append((label, value))
        else:
            # last resort
            rows.append((line, ""))

    return rows


# =========================
#  FILTERS (avoid confusion)
# =========================

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
    return any(k in s for k in MOTOR_BLACKLIST) or any(k in s for k in BOM_BLACKLIST) or any(k in s for k in ILLUSTRATION_BLACKLIST)


def _canon(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "")).strip()


def extract_pdf_rows(pdf_bytes: bytes, source_label: str) -> List[PDFRow]:
    """
    Extracts rows from:
      - Main characteristics
      - Options & Conditions
    Returns list of PDFRow. Keys are unique (even if label duplicates).
    """
    text = _pdf_text(pdf_bytes)
    lang = detect_language(text)

    if lang == "EN":
        main_start = [r"\bMain characteristics\b", r"\bGeneral characteristics\b"]
        main_end = [r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b"]
        opt_start = [r"\bOptions\s*&\s*Conditions\b"]
        opt_end = [r"\b$"]
    else:
        main_start = [r"\bCaract[ée]ristiques principales\b", r"\bCaract[ée]ristiques\b"]
        main_end = [r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b", r"\bConditions\b"]
        opt_start = [r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b"]
        opt_end = [r"\b$"]

    rows: List[PDFRow] = []
    counters: Dict[str, int] = {}  # per canonical label

    def add_rows(block: str, section: str):
        nonlocal rows, counters
        for label, value in parse_table_like_lines(block):
            c = _canon(label)
            if not c:
                continue
            if _is_blacklisted(c):
                continue

            counters[c] = counters.get(c, 0) + 1
            # unique key even if same label appears multiple times in same source
            key = f"{source_label}::{section}::{c}::{counters[c]}"
            rows.append(PDFRow(key=key, label=label, value=value, source=source_label, section=section))

    # main
    bounds = _find_section_bounds(text, main_start, main_end)
    if bounds:
        add_rows(text[bounds[0]:bounds[1]].strip(), section="main")

    # options
    bounds = _find_section_bounds(text, opt_start, opt_end)
    if bounds:
        add_rows(text[bounds[0]:bounds[1]].strip(), section="options")

    # Fallback if sections not found
    if not rows:
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if re.match(r"^(LUCAS\b|HIGH\b|2-AXES\b|RC2\b|ALES\b)", line, flags=re.I):
                continue
            m = re.match(r"^(.*?)[ ]{2,}(.+)$", line)
            if not m:
                continue
            label = m.group(1).strip().rstrip(":")
            value = m.group(2).strip()
            c = _canon(label)
            if not c or _is_blacklisted(c):
                continue
            counters[c] = counters.get(c, 0) + 1
            key = f"{source_label}::fallback::{c}::{counters[c]}"
            rows.append(PDFRow(key=key, label=label, value=value, source=source_label, section="main"))

    return rows


# =========================
#  PDF LAB - GENERATION
# =========================

def _draw_lucas_header(c: canvas.Canvas, lang: str) -> float:
    """
    Draws Lucas-style top band (black + red accent). Returns y cursor start for content.
    """
    w, h = A4

    c.setFillGray(0.05)
    c.rect(0, h - 22*mm, w, 22*mm, stroke=0, fill=1)

    c.setFillColorRGB(0.85, 0.0, 0.0)
    c.rect(0, h - 22*mm, w, 3.5*mm, stroke=0, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(12*mm, h - 14*mm, "LUCAS ROBOTIC SYSTEM")

    c.setFont("Helvetica", 10)
    subtitle = "TECHNICAL DATA SHEET" if lang == "EN" else "FICHE TECHNIQUE"
    c.drawRightString(w - 12*mm, h - 14*mm, subtitle)

    return h - 28*mm


def _draw_table(c: canvas.Canvas, x: float, y: float, w: float, rows: List[PDFRow], lang: str, title: str) -> float:
    """
    Simple table with 3 columns: Variable / Value / Source.
    Returns new y position.
    """
    c.setFont("Helvetica-Bold", 11)
    c.setFillGray(0)
    c.drawString(x, y, title)
    y -= 6*mm

    col1 = x
    col2 = x + w*0.54
    col3 = x + w*0.83

    c.setFont("Helvetica-Bold", 9)
    h1 = "Variable"
    h2 = "Value" if lang == "EN" else "Valeur"
    h3 = "Source"
    c.drawString(col1, y, h1)
    c.drawString(col2, y, h2)
    c.drawString(col3, y, h3)
    y -= 3.5*mm

    c.setLineWidth(0.3)
    c.setStrokeGray(0.7)
    c.line(x, y, x+w, y)
    y -= 3.5*mm

    c.setFont("Helvetica", 9)
    c.setFillGray(0)

    def wrap(text: str, max_chars: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return [""]
        words = text.split(" ")
        lines: List[str] = []
        cur = ""
        for wd in words:
            if len(cur) + len(wd) + 1 <= max_chars:
                cur = (cur + " " + wd).strip()
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return lines or [""]

    for r in rows:
        lab_lines = wrap(r.label, 48)
        val_lines = wrap(r.value, 36)
        n = max(len(lab_lines), len(val_lines), 1)
        row_h = n * 4.2*mm

        if y - row_h < 15*mm:
            c.showPage()
            y = _draw_lucas_header(c, lang)

            c.setFont("Helvetica-Bold", 11)
            c.drawString(x, y, title)
            y -= 6*mm
            c.setFont("Helvetica-Bold", 9)
            c.drawString(col1, y, h1)
            c.drawString(col2, y, h2)
            c.drawString(col3, y, h3)
            y -= 3.5*mm
            c.setStrokeGray(0.7)
            c.line(x, y, x+w, y)
            y -= 3.5*mm
            c.setFont("Helvetica", 9)

        for i in range(n):
            c.drawString(col1, y - i*4.2*mm, lab_lines[i] if i < len(lab_lines) else "")
            c.drawString(col2, y - i*4.2*mm, val_lines[i] if i < len(val_lines) else "")
            if i == 0:
                c.drawString(col3, y, r.source)
        y -= row_h + 1.5*mm

    return y


def build_fusion_pdf(
    pdf_x_bytes: bytes,
    pdf_yz_bytes: bytes,
    keep_keys: Optional[List[str]] = None,
) -> bytes:
    """
    Create a cleaned & fused PDF.
    - Sources renamed to Ensemble X and Ensemble YZ.
    - No "FUSION PDF / document généré..." sentence.
    - Removes motor/BOM variables BEFORE selection.
    """
    rows_x = extract_pdf_rows(pdf_x_bytes, source_label="Ensemble X")
    rows_yz = extract_pdf_rows(pdf_yz_bytes, source_label="Ensemble YZ")

    # Decide language based on X first then YZ (keeps “source-like” behavior)
    lang = detect_language(_pdf_text(pdf_x_bytes)) or detect_language(_pdf_text(pdf_yz_bytes))

    # Merge rows (keep all; keys are unique)
    merged_rows = rows_x + rows_yz

    # Filter by selection
    if keep_keys is not None:
        keep_set = set(keep_keys)
        merged_rows = [r for r in merged_rows if r.key in keep_set]

    # Split sections by their extracted section
    main_rows = [r for r in merged_rows if r.section == "main"]
    opt_rows = [r for r in merged_rows if r.section == "options"]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    y = _draw_lucas_header(c, lang)
    x = 12*mm
    w = A4[0] - 24*mm

    title_main = "Main characteristics" if lang == "EN" else "Caractéristiques principales"
    title_opt = "Options & Conditions"  # same text both languages (OK)

    y = _draw_table(c, x, y, w, main_rows, lang, title_main)
    y -= 2*mm
    _draw_table(c, x, y, w, opt_rows, lang, title_opt)

    c.save()
    return buf.getvalue()


# =========================
#  STREAMLIT UI (PDF LAB)
# =========================

def render_pdf_lab_panel():
    import streamlit as st

    st.subheader("PDF Lab – Fusion & simplification des fiches techniques")

    st.write(
        "Objectif : générer une fiche technique **propre et rigoureuse** à partir de deux PDFs source.\n\n"
        "- Les variables moteur/BOM n'apparaissent pas dans la sélection.\n"
        "- L'utilisateur choisit quelles variables conserver.\n"
        "- Sources renommées : **Ensemble X** et **Ensemble YZ**.\n"
        "- Aucune traduction : chaque valeur reste dans la langue du PDF source."
    )

    colA, colB = st.columns(2)
    with colA:
        pdf_x = st.file_uploader("PDF source – Ensemble X (ex : axe élevé ES)", type=["pdf"], key="pdf_x")
    with colB:
        pdf_yz = st.file_uploader("PDF source – Ensemble YZ (ex : RC2YZ)", type=["pdf"], key="pdf_yz")

    if not (pdf_x and pdf_yz):
        st.info("Chargez les deux PDFs pour activer la fusion.")
        return

    x_bytes = pdf_x.read()
    yz_bytes = pdf_yz.read()

    rows_x = extract_pdf_rows(x_bytes, source_label="Ensemble X")
    rows_yz = extract_pdf_rows(yz_bytes, source_label="Ensemble YZ")
    all_rows = rows_x + rows_yz

    # Dataframe for UI
    df = pd.DataFrame([{
        "Keep": True,
        "Label": r.label,
        "Value": r.value,
        "Source": r.source,
        "Section": r.section,
        "_key": r.key,
    } for r in all_rows])

    # Counters (to remove doubt about fusion)
    st.info(
        f"Detected rows: {len(df)}  |  "
        f"Ensemble X: {(df['Source']=='Ensemble X').sum()}  |  "
        f"Ensemble YZ: {(df['Source']=='Ensemble YZ').sum()}"
    )

    st.markdown("#### Sélection des variables à conserver")

    search = st.text_input("Search (filters Label/Value):", value="", key="pdf_search")
    df_view = df.copy()
    if search.strip():
        s = search.strip().lower()
        df_view = df_view[
            df_view["Label"].str.lower().str.contains(s, na=False)
            | df_view["Value"].astype(str).str.lower().str.contains(s, na=False)
        ]

    edited = st.data_editor(
        df_view[["Keep", "Label", "Value", "Source", "Section", "_key"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Keep": st.column_config.CheckboxColumn("Keep", default=True),
            "Label": st.column_config.TextColumn("Label", disabled=True),
            "Value": st.column_config.TextColumn("Value"),
            "Source": st.column_config.TextColumn("Source", disabled=True),
            "Section": st.column_config.TextColumn("Section", disabled=True),
            "_key": st.column_config.TextColumn("key", disabled=True, width="small"),
        },
        key="pdf_rows_editor",
    )

    # Reinject edited rows into full df by key
    # (only changed rows in view are applied)
    ed_map = {row["_key"]: row for _, row in edited.iterrows()}
    for i in range(len(df)):
        k = df.at[i, "_key"]
        if k in ed_map:
            df.at[i, "Keep"] = bool(ed_map[k]["Keep"])
            df.at[i, "Value"] = str(ed_map[k]["Value"])

    if st.button("Générer le PDF fusionné", key="btn_pdf_fusion"):
        try:
            keep_keys = df[df["Keep"]]["_key"].tolist()

            # apply edited values back into objects
            # (we regenerate from rows_x/rows_yz then override values from df)
            all_rows_map = {r.key: r for r in all_rows}
            for _, row in df.iterrows():
                k = row["_key"]
                if k in all_rows_map:
                    all_rows_map[k].value = str(row["Value"])

            # build pdf using selected keys (with updated values)
            # we just pass original bytes + keep list; rows extraction is deterministic
            out = build_fusion_pdf(x_bytes, yz_bytes, keep_keys=keep_keys)

            st.success("PDF généré ✅")
            st.download_button(
                "💾 Télécharger la fiche technique fusionnée",
                data=out,
                file_name="FICHE_TECHNIQUE_FUSION.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Erreur pendant la génération : {e}")
