# -*- coding: utf-8 -*-
"""
PDF Lab – Fusion & simplification des fiches techniques (Lucas)

Objectifs:
- Fusionner 2 PDFs source en une fiche technique propre.
- Sources renommées : "Ensemble X" et "Ensemble YZ".
- L'utilisateur sélectionne quelles variables conserver + peut éditer les valeurs.
- Variables moteur / BOM / illustrations : filtrées (n'apparaissent pas dans la liste).
- Sortie PDF avec bandeau Lucas (noir + liseré rouge).
- Police "sûre" : DejaVu Sans embarquée si disponible, sinon fallback système.

Important:
- Ce module est indépendant de la partie XML (ne rien supprimer côté app.py).
"""
import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================
#  FONT (robuste)
# =========================

def _register_lucas_fonts() -> Tuple[str, str]:
    """
    Register a robust font for cross-viewer rendering.
    Priority:
    1) assets/fonts/DejaVuSans(.ttf) and DejaVuSans-Bold(.ttf) if present in repo
    2) system DejaVu (common on Linux)
    3) fallback Helvetica / Helvetica-Bold
    Returns (regular_font_name, bold_font_name)
    """
    # already registered?
    if "LucasSans" in pdfmetrics.getRegisteredFontNames():
        return ("LucasSans", "LucasSans-Bold") if "LucasSans-Bold" in pdfmetrics.getRegisteredFontNames() else ("LucasSans", "LucasSans")

    candidates = [
        ("assets/fonts/DejaVuSans.ttf", "assets/fonts/DejaVuSans-Bold.ttf"),
        ("assets/fonts/dejavusans.ttf", "assets/fonts/dejavusans-bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ]

    for reg_path, bold_path in candidates:
        if os.path.exists(reg_path):
            try:
                pdfmetrics.registerFont(TTFont("LucasSans", reg_path))
                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont("LucasSans-Bold", bold_path))
                    return ("LucasSans", "LucasSans-Bold")
                return ("LucasSans", "LucasSans")
            except Exception:
                pass

    return ("Helvetica", "Helvetica-Bold")


# =========================
#  EXTRACTION
# =========================

@dataclass
class PDFRow:
    key: str
    label: str
    value: str
    source: str     # Ensemble X / Ensemble YZ
    section: str    # main / options


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    text = "\n".join(chunks)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    # Heuristique minimale: on n'impose pas une langue, on s'aligne sur le PDF
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
    Parse lines roughly like: "Label   Value"
    Works with PDF text extraction which often uses multiple spaces.
    """
    rows: List[Tuple[str, str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue

        # skip column headers
        if re.match(r"^(Variable|Option / Condition)\b", line, flags=re.I):
            continue
        if re.match(r"^(Valeur|Value)\b", line, flags=re.I):
            continue
        if re.match(r"^(Source)\b", line, flags=re.I):
            continue

        # Prefer split on 2+ spaces
        m = re.match(r"^(.*?)[ ]{2,}(.+)$", line)
        if m:
            label = m.group(1).strip().rstrip(":")
            value = m.group(2).strip()
            rows.append((label, value))
            continue

        # Fallback: split at first numeric-like token
        m = re.search(r"^(.*?)(\s[-+0-9].*)$", line)
        if m:
            rows.append((m.group(1).strip(), m.group(2).strip()))
        else:
            rows.append((line, ""))

    return rows


# =========================
#  FILTERS (moteur / BOM)
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


def extract_pdf_rows(pdf_bytes: bytes, source_label: str) -> List[PDFRow]:
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
    counters: Dict[str, int] = {}

    def add(block: str, section: str):
        for label, value in parse_table_like_lines(block):
            canon = re.sub(r"\s+", " ", (label or "")).strip()
            if not canon:
                continue
            if _is_blacklisted(canon):
                continue
            counters[canon] = counters.get(canon, 0) + 1
            key = f"{source_label}::{section}::{canon}::{counters[canon]}"
            rows.append(PDFRow(key=key, label=label, value=value, source=source_label, section=section))

    b = _find_section_bounds(text, main_start, main_end)
    if b:
        add(text[b[0]:b[1]].strip(), "main")

    b = _find_section_bounds(text, opt_start, opt_end)
    if b:
        add(text[b[0]:b[1]].strip(), "options")

    # fallback (si titres non trouvés)
    if not rows:
        for label, value in parse_table_like_lines(text):
            canon = re.sub(r"\s+", " ", (label or "")).strip()
            if not canon or _is_blacklisted(canon):
                continue
            counters[canon] = counters.get(canon, 0) + 1
            key = f"{source_label}::fallback::{canon}::{counters[canon]}"
            rows.append(PDFRow(key=key, label=label, value=value, source=source_label, section="main"))

    return rows


# =========================
#  GENERATION PDF (ReportLab Canvas)
# =========================

def _draw_lucas_header(c: canvas.Canvas, lang: str, font_reg: str, font_bold: str) -> float:
    w, h = A4

    # bandeau noir
    c.setFillGray(0.05)
    c.rect(0, h - 22*mm, w, 22*mm, stroke=0, fill=1)

    # liseré rouge
    c.setFillColorRGB(0.85, 0.0, 0.0)
    c.rect(0, h - 22*mm, w, 3.5*mm, stroke=0, fill=1)

    # texte blanc
    c.setFillColorRGB(1, 1, 1)
    c.setFont(font_bold, 13)
    c.drawString(12*mm, h - 14*mm, "LUCAS ROBOTIC SYSTEM")

    c.setFont(font_reg, 10)
    subtitle = "TECHNICAL DATA SHEET" if lang == "EN" else "FICHE TECHNIQUE"
    c.drawRightString(w - 12*mm, h - 14*mm, subtitle)

    c.setFillColorRGB(0, 0, 0)
    return h - 28*mm


def _wrap_words(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_table(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    rows: List[PDFRow],
    lang: str,
    title: str,
    font_reg: str,
    font_bold: str,
) -> float:
    # titre section
    c.setFont(font_bold, 11)
    c.setFillGray(0)
    c.drawString(x, y, title)
    y -= 6*mm

    col1 = x
    col2 = x + w * 0.54
    col3 = x + w * 0.83

    # entêtes
    c.setFont(font_bold, 9)
    c.drawString(col1, y, "Variable")
    c.drawString(col2, y, "Value" if lang == "EN" else "Valeur")
    c.drawString(col3, y, "Source")
    y -= 3.5*mm

    c.setLineWidth(0.3)
    c.setStrokeGray(0.7)
    c.line(x, y, x + w, y)
    y -= 3.5*mm

    c.setFont(font_reg, 9)
    c.setFillGray(0)

    for r in rows:
        lab_lines = _wrap_words(r.label, 48)
        val_lines = _wrap_words(r.value, 36)
        n = max(len(lab_lines), len(val_lines), 1)
        row_h = n * 4.2 * mm

        # saut de page
        if y - row_h < 15 * mm:
            c.showPage()
            y = _draw_lucas_header(c, lang, font_reg, font_bold)

            c.setFont(font_bold, 11)
            c.drawString(x, y, title)
            y -= 6*mm

            c.setFont(font_bold, 9)
            c.drawString(col1, y, "Variable")
            c.drawString(col2, y, "Value" if lang == "EN" else "Valeur")
            c.drawString(col3, y, "Source")
            y -= 3.5*mm

            c.setStrokeGray(0.7)
            c.line(x, y, x + w, y)
            y -= 3.5*mm

            c.setFont(font_reg, 9)

        # lignes
        for i in range(n):
            c.drawString(col1, y - i * 4.2 * mm, lab_lines[i] if i < len(lab_lines) else "")
            c.drawString(col2, y - i * 4.2 * mm, val_lines[i] if i < len(val_lines) else "")
            if i == 0:
                c.drawString(col3, y, r.source)

        y -= row_h + 1.5 * mm

    return y


def build_fusion_pdf(
    pdf_x_bytes: bytes,
    pdf_yz_bytes: bytes,
    keep_keys: Optional[List[str]] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Génère le PDF final.
    - keep_keys: liste de clés (PDFRow.key) à conserver (sinon tout).
    - overrides: dict key -> nouvelle valeur (édition dans Streamlit).
    """
    font_reg, font_bold = _register_lucas_fonts()

    rows_x = extract_pdf_rows(pdf_x_bytes, "Ensemble X")
    rows_yz = extract_pdf_rows(pdf_yz_bytes, "Ensemble YZ")

    # important: on garde TOUT (pas de setdefault qui écrase YZ)
    merged_rows: List[PDFRow] = rows_x + rows_yz

    # appliquer overrides
    if overrides:
        for r in merged_rows:
            if r.key in overrides:
                r.value = overrides[r.key]

    # filtrer keep_keys
    if keep_keys is not None:
        ks = set(keep_keys)
        merged_rows = [r for r in merged_rows if r.key in ks]

    # tri "lisible commercial" : X puis YZ, puis label
    merged_rows.sort(key=lambda r: (0 if r.source == "Ensemble X" else 1, r.section, r.label.lower()))

    lang = detect_language(_pdf_text(pdf_x_bytes)) or detect_language(_pdf_text(pdf_yz_bytes))

    main_rows = [r for r in merged_rows if r.section == "main"]
    opt_rows = [r for r in merged_rows if r.section == "options"]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    y = _draw_lucas_header(c, lang, font_reg, font_bold)
    x = 12 * mm
    w = A4[0] - 24 * mm

    title_main = "Main characteristics" if lang == "EN" else "Caractéristiques principales"
    y = _draw_table(c, x, y, w, main_rows, lang, title_main, font_reg, font_bold)

    # éviter pages blanches : n'afficher Options que s'il y a du contenu
    if opt_rows:
        y -= 2 * mm
        _draw_table(c, x, y, w, opt_rows, lang, "Options & Conditions", font_reg, font_bold)

    c.save()
    return buf.getvalue()


# =========================
#  STREAMLIT UI
# =========================

def render_pdf_lab_panel():
    import streamlit as st

    st.subheader("PDF Lab – Fusion & simplification des fiches techniques")

    st.write(
        "Objectif : générer une fiche technique **propre et rigoureuse** à partir de deux PDFs source.\n\n"
        "- Variables moteur/BOM/illustrations : **filtrées** (n’apparaissent pas).\n"
        "- Les valeurs peuvent être **éditées**.\n"
        "- Les variables peuvent être **désélectionnées**.\n"
        "- Sources renommées : **Ensemble X** et **Ensemble YZ**.\n"
        "- Aucune traduction : on conserve la langue du PDF source."
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

    rows_x = extract_pdf_rows(x_bytes, "Ensemble X")
    rows_yz = extract_pdf_rows(yz_bytes, "Ensemble YZ")
    all_rows = rows_x + rows_yz

    df = pd.DataFrame([{
        "_key": r.key,
        "Keep": True,
        "Variable": r.label,
        "Valeur": r.value,
        "Source": r.source,
        "Section": r.section,
    } for r in all_rows])

    st.info(
        f"Variables détectées: {len(df)} | "
        f"Ensemble X: {(df['Source']=='Ensemble X').sum()} | "
        f"Ensemble YZ: {(df['Source']=='Ensemble YZ').sum()}"
    )

    # UI lisible
    search = st.text_input("Recherche (Variable/Valeur):", value="")
    df_view = df.copy()
    if search.strip():
        s = search.strip().lower()
        df_view = df_view[
            df_view["Variable"].astype(str).str.lower().str.contains(s, na=False)
            | df_view["Valeur"].astype(str).str.lower().str.contains(s, na=False)
        ]

    edited = st.data_editor(
        df_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Keep": st.column_config.CheckboxColumn("Keep", default=True),
            "Variable": st.column_config.TextColumn("Variable", disabled=True),
            "Valeur": st.column_config.TextColumn("Valeur"),
            "Source": st.column_config.TextColumn("Source", disabled=True),
            "Section": st.column_config.TextColumn("Section", disabled=True),
            "_key": st.column_config.TextColumn("_key", disabled=True),
        },
        key="pdf_editor",
    )

    # réinjecter edits dans df complet via _key
    ed_map = {row["_key"]: row for _, row in edited.iterrows()}
    for i in range(len(df)):
        k = df.at[i, "_key"]
        if k in ed_map:
            df.at[i, "Keep"] = bool(ed_map[k]["Keep"])
            df.at[i, "Valeur"] = str(ed_map[k]["Valeur"])

    if st.button("Générer le PDF fusionné"):
        keep_keys = df[df["Keep"]]["_key"].tolist()
        overrides = {row["_key"]: str(row["Valeur"]) for _, row in df.iterrows()}

        out = build_fusion_pdf(
            x_bytes,
            yz_bytes,
            keep_keys=keep_keys,
            overrides=overrides,
        )

        st.success("PDF généré ✅")
        st.download_button(
            "💾 Télécharger la fiche technique fusionnée",
            data=out,
            file_name="FICHE_TECHNIQUE_FUSION.pdf",
            mime="application/pdf",
        )
