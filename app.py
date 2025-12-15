import io
import copy
import xml.etree.ElementTree as ET
import streamlit as st

import pdf_lab
import step_lab  # ✅ AJOUT : module STEP Lab (Option B)

# =========================
#  OUTILS XML GÉNÉRIQUES
# =========================

def get_ref_sylob(part: ET.Element) -> str:
    vars_el = part.find("VARIABLES")
    if vars_el is None:
        return ""
    return (vars_el.findtext("REFSYLOB") or "").strip()


def get_qty_sylob(part: ET.Element) -> float:
    vars_el = part.find("VARIABLES")
    if vars_el is None:
        return 0.0
    txt = (vars_el.findtext("QTESYLOB") or "0").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return 0.0


def set_qty_sylob(part: ET.Element, qty: float) -> None:
    vars_el = part.find("VARIABLES")
    if vars_el is None:
        vars_el = ET.SubElement(part, "VARIABLES")
    qty_el = vars_el.find("QTESYLOB")
    if qty_el is None:
        qty_el = ET.SubElement(vars_el, "QTESYLOB")
    qty_el.text = f"{qty:g}"


# =========================
#  RC3 -> RCXY 2 AXES SURFACIQUE
# =========================

def is_z_part(part: ET.Element) -> bool:
    """
    Détecte si un PART appartient à l'axe Z.
    Règles :
      - NB commence par "rcz_"
      - OU texte lisible contient "Bras Z" ou "axe Z"
    MAIS :
      - on NE supprime PAS le chariot standard Z (interface robot)
    """
    nn = (part.findtext("NN") or "").lower()
    nb = (part.findtext("NB") or "").lower()
    nt = (part.findtext("NT") or "").lower()
    lina = (part.findtext("LINA") or "").lower()

    # Exception : chariot standard Z = interface à conserver
    if nb.startswith("rcz_chariot_standard"):
        return False
    if "chariot standard bras z" in nn:
        return False

    # Nom interne typé Z
    if nb.startswith("rcz_"):
        return True

    # Texte lisible mentionnant bras / axe Z
    z_keywords = ["bras z", "axe z"]
    text_blob = " ".join([nn, nt, lina])
    if any(kw in text_blob for kw in z_keywords):
        return True

    return False


def transform_to_rcxy_2axes(root: ET.Element, comment_override: str | None = None) -> None:
    """
    Modifie l'ASSEMBLY en place :
      - Supprime les PART de l'axe Z (sauf chariot interface)
      - Renomme l'assemblage en RCXY 2 axes surfacique
      - Adapte le commentaire final, ou le remplace si comment_override fourni
    """
    # 1) Renommer l'assemblage
    nt = root.find("NT")
    if nt is not None:
        nt.text = "ROBOT CARTESIEN 2 AXES SURFACIQUE"

    lina = root.find("LINA")
    if lina is not None and lina.text:
        lina.text = lina.text.replace(
            "ROBOT CARTESIEN 3 AXES",
            "ROBOT CARTESIEN 2 AXES SURFACIQUE"
        )

    # 2) Supprimer les PART Z dans SUBPARTS
    subparts = root.find("SUBPARTS")
    if subparts is not None:
        to_remove = []
        for part in subparts.findall("PART"):
            if is_z_part(part):
                to_remove.append(part)

        for part in to_remove:
            subparts.remove(part)

    # 3) Adapter le commentaire final
    if subparts is not None:
        # On applique l'override directement sur tous les PART "Commentaire"
        comment_parts = []
        for part in subparts.findall("PART"):
            nn = (part.findtext("NN") or "").strip().lower()
            if nn == "commentaire":
                comment_parts.append(part)

        if comment_parts:
            main_part = comment_parts[0]
            vars_el = main_part.find("VARIABLES")
            if vars_el is None:
                vars_el = ET.SubElement(main_part, "VARIABLES")
            comment_el = vars_el.find("COMMENTAIRE")
            if comment_el is None:
                comment_el = ET.SubElement(vars_el, "COMMENTAIRE")

            if comment_override is not None:
                comment_el.text = comment_override
            else:
                txt = comment_el.text or ""
                txt = txt.replace(
                    "Robot cartésien 3 axes",
                    "Robot cartésien 2 axes surfacique"
                )
                txt = txt.replace(
                    "Taille du bras : 4",
                    "Taille du bras : 0"
                )
                txt = txt.replace(
                    "Taille du bras Z : 4",
                    "Taille du bras : 0"
                )
                comment_el.text = txt

            # On supprime d'autres éventuels PART "Commentaire"
            for extra in comment_parts[1:]:
                subparts.remove(extra)


def extract_default_comment_rcxy(xml_bytes: bytes) -> str:
    """
    À partir d'un XML RC3, génère en mémoire une version RCXY 2 axes surfacique
    et retourne le texte du COMMENTAIRE résultant (par défaut).
    """
    buf = io.BytesIO(xml_bytes)
    tree = ET.parse(buf)
    root = tree.getroot()

    if root.tag != "ASSEMBLY":
        raise ValueError("Fichier inattendu : racine != ASSEMBLY")

    # On applique la transformation SANS override de commentaire
    transform_to_rcxy_2axes(root, comment_override=None)

    # On récupère le commentaire
    subparts = root.find("SUBPARTS")
    if subparts is not None:
        for part in subparts.findall("PART"):
            nn = (part.findtext("NN") or "").strip().lower()
            if nn == "commentaire":
                vars_el = part.find("VARIABLES")
                if vars_el is None:
                    break
                comment_el = vars_el.find("COMMENTAIRE")
                if comment_el is not None and comment_el.text:
                    return comment_el.text

    # fallback si rien trouvé
    return "Robot cartésien 2 axes surfacique - Taille du bras : 0 - Charge : ..."


def convert_rc3_to_rcxy(xml_bytes: bytes, comment_override: str | None = None) -> bytes:
    """
    Prend le contenu XML (bytes) d'un RC3,
    renvoie le contenu XML (bytes) du RCXY 2 axes surfacique.
    """
    buf_in = io.BytesIO(xml_bytes)
    tree = ET.parse(buf_in)
    root = tree.getroot()

    if root.tag != "ASSEMBLY":
        raise ValueError("Fichier inattendu : racine != ASSEMBMBLY")

    transform_to_rcxy_2axes(root, comment_override=comment_override)

    buf_out = io.BytesIO()
    tree.write(buf_out, encoding="utf-8", xml_declaration=True)
    return buf_out.getvalue()


# =========================
#  LOGIQUE CANTILEVER (ES + RC2YZ)
# =========================

SERVICE_REFS = {
    "AL00-ETUDE-MECANIQUE",
    "AL00-MONTAGE-MECANIQUE",
    "AL00-CONTROLE-QUALITE",
    "AL00-DEMONTAGE-CHARGEMENT",
    "AL00-EMBALLAGE",
    "AL00-TEST-CMU",
    "AL00-TRANSPORT",
    "AL00-MONTAGE",
}

SUPPORT_REFS_RC2 = {
    "AL00-POTEAU-BAS",
    "ALCS-NEZ-DE-POTEAU",
    "ALCS-BARREAU-POTEAU-CAT03",
    "ALP0-CHEVILLE",
}

CHARIOT_REFS_ES = {
    "ALES-CHARIOT-CAT08",
}


def get_n_chariots(root_es: ET.Element) -> int:
    """
    Retourne le nombre de chariots (têtes YZ) sur l'axe élevé ES,
    d'après la QTESYLOB du chariot ALES.
    """
    sub = root_es.find("SUBPARTS")
    if sub is None:
        return 1
    for part in sub.findall("PART"):
        ref = get_ref_sylob(part)
        if ref in CHARIOT_REFS_ES:
            qty = get_qty_sylob(part)
            try:
                n = int(round(qty))
                return max(1, n)
            except Exception:
                return 1
    return 1


def split_parts_services(subparts: ET.Element):
    """
    Sépare les PART d'un SUBPARTS en :
      - non_services : liste des PART non "service"
      - services_qty : dict ref -> qty
      - services_tpl : dict ref -> PART (template pour recréer plus tard)
    """
    non_services = []
    services_qty = {}
    services_tpl = {}

    for part in subparts.findall("PART"):
        ref = get_ref_sylob(part)
        if not ref or ref not in SERVICE_REFS:
            non_services.append(part)
            continue

        qty = get_qty_sylob(part)
        services_qty[ref] = services_qty.get(ref, 0.0) + qty
        if ref not in services_tpl:
            services_tpl[ref] = part

    return non_services, services_qty, services_tpl


def compute_final_services(es_services: dict, rc2_services: dict, n_chariots: int) -> dict:
    """
    Applique les règles de combinaison ES + RC2 en fonction du nombre de chariots.
    Retourne un dict { ref: qty_final }.
    """
    final = {}
    all_refs = set(es_services.keys()) | set(rc2_services.keys())

    for ref in all_refs:
        E = es_services.get(ref, 0.0)
        R = rc2_services.get(ref, 0.0)

        if ref == "AL00-ETUDE-MECANIQUE":
            total = E + 0.9 * R
        elif ref == "AL00-MONTAGE-MECANIQUE":
            total = E + 0.9 * (R * n_chariots)
        elif ref in {"AL00-CONTROLE-QUALITE", "AL00-DEMONTAGE-CHARGEMENT", "AL00-TEST-CMU"}:
            total = E + (R * n_chariots)
        elif ref in {"AL00-EMBALLAGE", "AL00-TRANSPORT", "AL00-MONTAGE"}:
            total = E
        else:
            total = E + (R * n_chariots)

        if total > 0:
            final[ref] = total

    return final


def build_cantilever(xml_es: bytes, xml_rc2: bytes, comment_override: str | None = None) -> bytes:
    """
    Construit un BOM 'ROBOT CANTILEVER 3 AXES XYZ' à partir de :
      - xml_es  : axe élevé sur poteaux (ES1, ES2...)
      - xml_rc2 : robot 2 axes YZ (RC2)
    """
    tree_es = ET.parse(io.BytesIO(xml_es))
    root_es = tree_es.getroot()

    tree_r2 = ET.parse(io.BytesIO(xml_rc2))
    root_r2 = tree_r2.getroot()

    if root_es.tag != "ASSEMBLY" or root_r2.tag != "ASSEMBLY":
        raise ValueError("Fichiers inattendus : racine != ASSEMBLY")

    sub_es = root_es.find("SUBPARTS")
    sub_r2 = root_r2.find("SUBPARTS")
    if sub_es is None or sub_r2 is None:
        raise ValueError("SUBPARTS manquant dans un des fichiers.")

    n_chariots = get_n_chariots(root_es)

    es_non_services, es_services_qty, es_services_tpl = split_parts_services(sub_es)
    rc2_non_services_raw, rc2_services_qty_base, rc2_services_tpl = split_parts_services(sub_r2)

    rc2_non_services = []
    for part in rc2_non_services_raw:
        ref = get_ref_sylob(part)
        if ref in SUPPORT_REFS_RC2:
            continue
        rc2_non_services.append(part)

    for p in list(sub_es.findall("PART")):
        sub_es.remove(p)

    mechanical_by_ref = {}
    no_ref_parts = []

    for part in es_non_services:
        ref = get_ref_sylob(part)
        if not ref:
            no_ref_parts.append(copy.deepcopy(part))
            continue
        qty = get_qty_sylob(part)
        if ref not in mechanical_by_ref:
            mechanical_by_ref[ref] = {"part": copy.deepcopy(part), "qty": qty}
        else:
            mechanical_by_ref[ref]["qty"] += qty

    for part in rc2_non_services:
        ref = get_ref_sylob(part)
        qty_base = get_qty_sylob(part)
        qty = qty_base * n_chariots
        if not ref:
            for _ in range(max(1, n_chariots)):
                no_ref_parts.append(copy.deepcopy(part))
            continue

        if ref not in mechanical_by_ref:
            mechanical_by_ref[ref] = {"part": copy.deepcopy(part), "qty": qty}
        else:
            mechanical_by_ref[ref]["qty"] += qty

    for ref, info in mechanical_by_ref.items():
        part = info["part"]
        set_qty_sylob(part, info["qty"])
        sub_es.append(part)

    for p in no_ref_parts:
        sub_es.append(p)

    final_services_qty = compute_final_services(es_services_qty, rc2_services_qty_base, n_chariots)

    service_templates = {}
    for ref in SERVICE_REFS:
        if ref in es_services_tpl:
            service_templates[ref] = es_services_tpl[ref]
        elif ref in rc2_services_tpl:
            service_templates[ref] = rc2_services_tpl[ref]

    for ref, qty in final_services_qty.items():
        tpl = service_templates.get(ref)
        if tpl is None:
            continue
        new_part = copy.deepcopy(tpl)
        set_qty_sylob(new_part, qty)
        sub_es.append(new_part)

    nt_el = root_es.find("NT")
    if nt_el is None:
        nt_el = ET.SubElement(root_es, "NT")
    nt_el.text = "ROBOT CANTILEVER 3 AXES XYZ"

    lina_el = root_es.find("LINA")
    if lina_el is None:
        lina_el = ET.SubElement(root_es, "LINA")
    charge_label = "CHARGE" if n_chariots <= 1 else "CHARGES"
    lina_el.text = f"LUCAS - ROBOT CANTILEVER 3 AXES XYZ - {n_chariots} {charge_label}"

    sub_es = root_es.find("SUBPARTS")
    if sub_es is not None:
        comment_parts = []
        for part in sub_es.findall("PART"):
            nn = (part.findtext("NN") or "").strip().lower()
            if nn == "commentaire":
                comment_parts.append(part)

        if comment_parts:
            main_part = comment_parts[0]
            vars_el = main_part.find("VARIABLES")
            if vars_el is None:
                vars_el = ET.SubElement(main_part, "VARIABLES")
            comment_el = vars_el.find("COMMENTAIRE")
            if comment_el is None:
                comment_el = ET.SubElement(vars_el, "COMMENTAIRE")

            if comment_override is not None:
                comment_el.text = comment_override

            for extra in comment_parts[1:]:
                sub_es.remove(extra)
        else:
            if comment_override:
                main_part = ET.SubElement(sub_es, "PART")
                nn_el = ET.SubElement(main_part, "NN")
                nn_el.text = "Commentaire"
                vars_el = ET.SubElement(main_part, "VARIABLES")
                com_el = ET.SubElement(vars_el, "COMMENTAIRE")
                com_el.text = comment_override

    buf_out = io.BytesIO()
    tree_es.write(buf_out, encoding="utf-8", xml_declaration=True)
    return buf_out.getvalue()


def extract_default_comment_cantilever(xml_rc2: bytes) -> str:
    """
    Récupère le commentaire du RC2 (pour servir de base au commentaire cantilever).
    """
    buf = io.BytesIO(xml_rc2)
    tree = ET.parse(buf)
    root = tree.getroot()
    if root.tag != "ASSEMBLY":
        return "Robot cantilever 3 axes XYZ - ..."

    sub = root.find("SUBPARTS")
    if sub is not None:
        for part in sub.findall("PART"):
            nn = (part.findtext("NN") or "").strip().lower()
            if nn == "commentaire":
                vars_el = part.find("VARIABLES")
                if vars_el is None:
                    break
                comment_el = vars_el.find("COMMENTAIRE")
                if comment_el is not None and comment_el.text:
                    txt = comment_el.text
                    txt = txt.replace("Robot cartésien 2 axes YZ", "Robot cantilever 3 axes XYZ")
                    return txt
    return "Robot cantilever 3 axes XYZ - ..."


# =========================
#  INTERFACE STREAMLIT
# =========================

st.set_page_config(
    page_title="Lucas – Config BOM",
    page_icon="🤖",
    layout="wide"
)

st.title("🧩 Lucas – Configurateur BOM (XML → Sylob)")
st.write(
    "Choisissez un type de produit ci-dessous pour appliquer une transformation "
    "sur les fichiers XML issus du configurateur (BOM → Sylob)."
)

st.markdown("### Choix du produit")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

if "mode" not in st.session_state:
    st.session_state.mode = None

with col1:
    st.markdown("#### 🧱 RCXY 2 axes surfacique")
    st.write(
        "À partir d’un RC3 (3 axes), suppression de l’axe Z (sauf le chariot interface), "
        "renommage du produit et adaptation du commentaire pour obtenir un robot "
        "**2 axes surfacique** compatible Sylob."
    )
    if st.button("Ouvrir", key="btn_rcxy"):
        st.session_state.mode = "rcxy_2axes"

with col2:
    st.markdown("#### 🏗️ Robot cantilever 3 axes XYZ")
    st.write(
        "Fusion d’un axe élevé sur poteaux (ES) avec un robot 2 axes YZ (RC2). "
        "Le configurateur génère automatiquement un robot cantilever 3 axes XYZ, "
        "en gérant 1 ou 2 têtes YZ selon le nombre de chariots sur l’axe élevé."
    )
    if st.button("Ouvrir", key="btn_cantilever"):
        st.session_state.mode = "cantilever"

with col3:
    st.markdown("#### 📄 PDF Lab (fusion / nettoyage)")
    st.write(
        "Outil indépendant pour nettoyer ou assembler des fiches techniques : "
        "sélection des lignes à conserver, édition de certaines valeurs, "
        "export PDF + Word (Word modifiable)."
    )
    if st.button("Ouvrir", key="btn_pdf_lab"):
        st.session_state.mode = "pdf_lab"

with col4:
    st.markdown("#### 🧊 STEP Lab (assemblage cantilever)")
    st.write(
        "Assemblage automatique de 2 STEP (Ensemble 1 + Ensemble 2) sans CAO : "
        "application d’une règle de placement déterministe (rotation + translation) "
        "et export d’un STEP assemblé."
    )
    if st.button("Ouvrir", key="btn_step_lab"):
        st.session_state.mode = "step_lab"

st.markdown("---")

if st.session_state.mode == "rcxy_2axes":
    st.subheader("RC3 → RCXY 2 axes surfacique")

    st.write(
        "1️⃣ Chargez un XML de robot cartésien 3 axes (RC3).\n"
        "2️⃣ L’outil supprime l’axe Z (sauf le chariot interface), renomme le produit "
        "et adapte le commentaire pour générer un **RCXY 2 axes surfacique**.\n"
        "3️⃣ Vous pouvez modifier le commentaire avant la génération du XML final."
    )

    uploaded_file = st.file_uploader("Fichier XML RC3 (3 axes)", type=["xml"], key="upload_rc3")

    if uploaded_file is not None:
        xml_in = uploaded_file.read()
        st.session_state["xml_rc3_in"] = xml_in

        try:
            default_comment = extract_default_comment_rcxy(xml_in)
        except Exception as e:
            st.error(f"Erreur pendant l'analyse du fichier : {e}")
            default_comment = ""

        st.markdown("#### Commentaire RCXY 2 axes surfacique")
        st.text_area(
            "Texte du commentaire (modifiez-le librement avant conversion) :",
            value=default_comment,
            key="comment_rcxy",
            height=150
        )

        if st.button("Convertir avec ce commentaire"):
            try:
                user_comment = st.session_state.get("comment_rcxy", default_comment)
                xml_out = convert_rc3_to_rcxy(xml_in, comment_override=user_comment)

                st.success("Conversion terminée ✅")
                st.download_button(
                    label="💾 Télécharger le XML RCXY 2 axes surfacique",
                    data=xml_out,
                    file_name="RCXY_2axes_surfacique.xml",
                    mime="application/xml"
                )

            except Exception as e:
                st.error(f"Erreur pendant la conversion : {e}")
    else:
        st.info("Veuillez charger un fichier XML RC3 pour afficher et éditer le commentaire.")

elif st.session_state.mode == "cantilever":
    st.subheader("Cantilever = axe élevé ES + robot 2 axes RC2 (1 ou 2 têtes)")

    st.write(
        "1️⃣ Chargez un XML d’axe linéaire élevé sur poteaux (ES1, ES2...).\n"
        "2️⃣ Chargez un XML de robot cartésien 2 axes YZ (RC2).\n"
        "3️⃣ L’outil fusionne les deux BOM, gère automatiquement 1 ou 2 têtes YZ "
        "et produit un **robot cantilever 3 axes XYZ** compatible Sylob.\n"
        "4️⃣ Vous pouvez éditer le commentaire avant de générer le XML cantilever."
    )

    es_file = st.file_uploader("XML Axe élevé sur poteaux (ES1 / ES2...)", type=["xml"], key="upload_es")
    rc2_file = st.file_uploader("XML Robot 2 axes YZ (RC2)", type=["xml"], key="upload_rc2")

    if es_file is not None and rc2_file is not None:
        xml_es = es_file.read()
        xml_rc2 = rc2_file.read()

        st.session_state["xml_es_in"] = xml_es
        st.session_state["xml_rc2_in"] = xml_rc2

        try:
            default_comment_cant = extract_default_comment_cantilever(xml_rc2)
        except Exception as e:
            st.error(f"Erreur pendant l'analyse du RC2 pour le commentaire : {e}")
            default_comment_cant = "Robot cantilever 3 axes XYZ - ..."

        st.markdown("#### Commentaire cantilever")
        st.text_area(
            "Texte du commentaire (modifiez-le librement avant fusion) :",
            value=default_comment_cant,
            key="comment_cantilever",
            height=150
        )

        if st.button("Fusionner ES + RC2 et générer le cantilever"):
            try:
                user_comment = st.session_state.get("comment_cantilever", default_comment_cant)
                xml_out = build_cantilever(xml_es, xml_rc2, comment_override=user_comment)

                st.success("Cantilever généré ✅")
                st.download_button(
                    label="💾 Télécharger le XML ROBOT CANTILEVER 3 AXES XYZ",
                    data=xml_out,
                    file_name="ROBOT_CANTILEVER_3AXES_XYZ.xml",
                    mime="application/xml"
                )

            except Exception as e:
                st.error(f"Erreur pendant la fusion : {e}")
    else:
        st.info("Veuillez charger les deux fichiers XML (ES + RC2) pour pouvoir fusionner.")

elif st.session_state.mode == "pdf_lab":
    pdf_lab.render_pdf_lab_panel()

elif st.session_state.mode == "step_lab":
    step_lab.render_step_lab_panel()

else:
    st.info("Sélectionnez un produit ci-dessus pour commencer.")
