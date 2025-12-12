import io
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# =========================
#  PDF LAB - EXTRACTION
# =========================

@dataclass
class PDFRow:
    label: str
    value: str
    source: str  # "Ensemble X" / "Ensemble YZ" etc.


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        chunks.append(t)
    # Normalize whitespace
    text = "\n".join(chunks)
    text = text.replace("\r", "\n")
    # NOTE: keep multiple spaces (PDF extraction often uses them as column separators)
    # collapse too many blank lines
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
      "Charge admissible (X) 1000.0 kg ES"
    or:
      "Place of delivery and installation France"
    We try to split with "  " or last numeric-ish token; robust enough for our generated PDFs.
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

        # If the line has at least 2 "columns", split heuristically
        # Prefer split by "  " if present in extracted text (often not).
        if "  " in line:
            parts = [p.strip() for p in line.split("  ") if p.strip()]
            if len(parts) >= 2:
                label = parts[0]
                value = " ".join(parts[1:])
                rows.append((label, value))
                continue

        # Otherwise: split at last occurrence of " kg", " mm", " m/s", " m/s²", " months" etc.
        m = re.search(r"^(.*?)(\s[-+0-9].*)$", line)
        if m:
            label = m.group(1).strip()
            value = m.group(2).strip()
            rows.append((label, value))
        else:
            # fallback: keep whole line as label, empty value
            rows.append((line, ""))
    return rows


def extract_pdf_variables(pdf_bytes: bytes, source_label: str) -> Dict[str, PDFRow]:
    """
    Extracts:
      - Main characteristics
      - Options & conditions
    Returns dict keyed by canonical label.
    """
    text = _pdf_text(pdf_bytes)
    lang = detect_language(text)

    if lang == "EN":
        main_start = [r"\bMain characteristics\b"]
        main_end = [r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b"]
        opt_start = [r"\bOptions\s*&\s*Conditions\b"]
        opt_end = [r"\b$"]  # until end
    else:
        main_start = [r"\bCaract[ée]ristiques principales\b"]
        main_end = [r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b", r"\bConditions\b"]
        opt_start = [r"\bOptions\s*&\s*Conditions\b", r"\bOptions\b"]
        opt_end = [r"\b$"]

    out: Dict[str, PDFRow] = {}

    bounds = _find_section_bounds(text, main_start, main_end)
    if bounds:
        block = text[bounds[0]:bounds[1]].strip()
        for label, value in parse_table_like_lines(block):
            key = re.sub(r"\s+", " ", label).strip()
            if key:
                out[key] = PDFRow(label=label, value=value, source=source_label)

    bounds = _find_section_bounds(text, opt_start, opt_end)
    if bounds:
        block = text[bounds[0]:bounds[1]].strip()
        for label, value in parse_table_like_lines(block):
            key = re.sub(r"\s+", " ", label).strip()
            if key and key not in out:
                out[key] = PDFRow(label=label, value=value, source=source_label)


    # Fallback: if the PDF doesn't contain our section titles, parse all "Label  Value" lines.
    if not out:
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            # Skip obvious titles
            if re.match(r"^(LUCAS\b|HIGH\b|2-AXES\b|RC2\b|ALES\b)", line, flags=re.I):
                continue
            # Split on 2+ spaces (typical extraction)
            m = re.match(r"^(.*?)[ ]{2,}(.+)$", line)
            if not m:
                continue
            label = m.group(1).strip().rstrip(":")
            value = m.group(2).strip()
            if len(label) < 2 or len(value) < 1:
                continue
            key = re.sub(r"\s+", " ", label).strip()
            out[key] = PDFRow(label=label, value=value, source=source_label)

    return out


# =========================
#  PDF LAB - GENERATION
# =========================

def _draw_lucas_header(c: canvas.Canvas, lang: str) -> float:
    """
    Draws Lucas-style top band (black + red accent). Returns y cursor start for content.
    """
    w, h = A4

    # Black band
    c.setFillGray(0.05)
    c.rect(0, h - 22*mm, w, 22*mm, stroke=0, fill=1)

    # Red accent
    c.setFillColorRGB(0.85, 0.0, 0.0)
    c.rect(0, h - 22*mm, w, 3.5*mm, stroke=0, fill=1)

    # White text
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(12*mm, h - 14*mm, "LUCAS ROBOTIC SYSTEM")

    # Subtitle (generic, no "RC2/ES" wording)
    c.setFont("Helvetica", 10)
    subtitle = "TECHNICAL DATA SHEET" if lang == "EN" else "FICHE TECHNIQUE"
    c.drawRightString(w - 12*mm, h - 14*mm, subtitle)

    return h - 28*mm


def _draw_table(c: canvas.Canvas, x: float, y: float, w: float, rows: List[PDFRow], lang: str, title: str) -> float:
    """
    Simple table with 3 columns: Variable / Value / Source.
    Returns new y position.
    """
    # Table style
    c.setFont("Helvetica-Bold", 11)
    c.setFillGray(0)
    c.drawString(x, y, title)
    y -= 6*mm

    col1 = x
    col2 = x + w*0.54
    col3 = x + w*0.83

    c.setFont("Helvetica-Bold", 9)
    h1 = "Variable" if lang == "EN" else "Variable"
    h2 = "Value" if lang == "EN" else "Valeur"
    h3 = "Source" if lang == "EN" else "Source"
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
        # crude wrap by chars; works well for our short labels/values
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
        # wrap label/value
        lab_lines = wrap(r.label, 45)
        val_lines = wrap(r.value, 35)
        n = max(len(lab_lines), len(val_lines), 1)
        row_h = n * 4.2*mm

        # page break
        if y - row_h < 15*mm:
            c.showPage()
            y = _draw_lucas_header(c, lang)
            # redraw section title + headers quickly
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
    keep_labels: Optional[List[str]] = None,
) -> bytes:
    """
    Create a cleaned & fused PDF.
    - Sources renamed to Ensemble X and Ensemble YZ (as requested).
    - No "FUSION PDF / document généré..." sentence.
    - Keeps Lucas header style.
    """
    vars_x = extract_pdf_variables(pdf_x_bytes, source_label="Ensemble X")
    vars_yz = extract_pdf_variables(pdf_yz_bytes, source_label="Ensemble YZ")

    # Decide language based on X first then YZ
    lang = detect_language(_pdf_text(pdf_x_bytes)) or detect_language(_pdf_text(pdf_yz_bytes))

    # Merge
    merged: Dict[str, PDFRow] = {}
    for d in (vars_x, vars_yz):
        for k, row in d.items():
            merged.setdefault(k, row)  # keep first occurrence

    # Filter
    keys = list(merged.keys())
    if keep_labels:
        keep_set = set(keep_labels)
        keys = [k for k in keys if k in keep_set]

    # Split into 2 sections based on common keywords (simple heuristic)
    main_keys = []
    opt_keys = []
    for k in keys:
        if re.search(r"(Warranty|Place of delivery|Distance|tests|garant|livraison|distance|test|warranty)", k, flags=re.I):
            opt_keys.append(k)
        else:
            main_keys.append(k)

    main_rows = [merged[k] for k in main_keys]
    opt_rows = [merged[k] for k in opt_keys]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    y = _draw_lucas_header(c, lang)
    x = 12*mm
    w = A4[0] - 24*mm

    title_main = "Main characteristics" if lang == "EN" else "Caractéristiques principales"
    title_opt = "Options & Conditions" if lang == "EN" else "Options & Conditions"

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
        "- Suppression implicite : illustrations, nomenclature/BOM, dimensionnement moteur (non repris dans la sortie).\n"
        "- L'utilisateur choisit quelles variables conserver.\n"
        "- Sources renommées : **Ensemble X** et **Ensemble YZ**."
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

    # Preview list of variables
    vars_x = extract_pdf_variables(x_bytes, source_label="Ensemble X")
    vars_yz = extract_pdf_variables(yz_bytes, source_label="Ensemble YZ")
    all_keys = sorted(set(vars_x.keys()) | set(vars_yz.keys()))

    st.markdown("#### Sélection des variables à conserver")
    keep = st.multiselect(
        "Variables (décocher pour supprimer) :",
        options=all_keys,
        default=all_keys,
        help="La sortie PDF ne reprend que les variables sélectionnées."
    )

    if st.button("Générer le PDF fusionné", key="btn_pdf_fusion"):
        try:
            out = build_fusion_pdf(x_bytes, yz_bytes, keep_labels=keep)
            st.success("PDF généré ✅")
            st.download_button(
                "💾 Télécharger la fiche technique fusionnée",
                data=out,
                file_name="FICHE_TECHNIQUE_FUSION.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Erreur pendant la génération : {e}")
