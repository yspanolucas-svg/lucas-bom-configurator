import io
import xml.etree.ElementTree as ET
import streamlit as st

# =========================
#  LOGIQUE MÉTIER : RC3 -> RCXY 2 AXES SURFACIQUE
# =========================

def is_z_part(part: ET.Element) -> bool:
    """
    Détecte si un PART appartient à l'axe Z.
    Règles :
      - NB commence par "rcz_"  (Z)
      - OU NN / NT / LINA contient "Bras Z" ou "axe Z"
    MAIS :
      - on NE supprime PAS le chariot standard Z (nouvelle interface robot)
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
      - Adapte le commentaire final
      - Si comment_override est fourni, l'utilise comme COMMENTAIRE final
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
        for part in subparts.findall("PART"):
            nn = (part.findtext("NN") or "").strip().lower()
            if nn == "commentaire":
                vars_el = part.find("VARIABLES")
                if vars_el is None:
                    continue
                comment_el = vars_el.find("COMMENTAIRE")
                if comment_el is not None:
                    if comment_override is not None:
                        # On force exactement le texte choisi par l'utilisateur
                        comment_el.text = comment_override
                    else:
                        # Version par défaut : on adapte le texte existant
                        txt = comment_el.text or ""
                        txt = txt.replace(
                            "Robot cartésien 3 axes",
                            "Robot cartésien 2 axes surfacique"
                        )
                        # On force aussi la taille du bras à 0 si présent
                        txt = txt.replace(
                            "Taille du bras : 4",
                            "Taille du bras : 0"
                        )
                        comment_el.text = txt
                break


def extract_default_comment(xml_bytes: bytes) -> str:
    """
    À partir d'un XML RC3, génère la version RCXY 2 axes surfacique en mémoire
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
    return "Robot cartésien 2 axes surfacique - Taille du bras : 0 - ..."    


def convert_rc3_to_rcxy(xml_bytes: bytes, comment_override: str | None = None) -> bytes:
    """
    Prend le contenu XML (bytes) d'un RC3,
    renvoie le contenu XML (bytes) du RCXY 2 axes surfacique,
    avec éventuellement un COMMENTAIRE surchargé.
    """
    buf_in = io.BytesIO(xml_bytes)
    tree = ET.parse(buf_in)
    root = tree.getroot()

    if root.tag != "ASSEMBLY":
        raise ValueError("Fichier inattendu : racine != ASSEMBLY")

    transform_to_rcxy_2axes(root, comment_override=comment_override)

    buf_out = io.BytesIO()
    tree.write(buf_out, encoding="utf-8", xml_declaration=True)
    return buf_out.getvalue()


# =========================
#  INTERFACE STREAMLIT
# =========================

st.set_page_config(
    page_title="Lucas – Config BOM",
    page_icon="🤖",
    layout="centered"
)

st.title("🧩 Lucas – Configurateur BOM (XML → Sylob)")
st.write(
    "Choisissez un type de produit ci-dessous pour appliquer une transformation "
    "sur les fichiers XML issus du configurateur."
)

# ------- Grille de “cartes produit” -------

st.markdown("### Choix du produit")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

if "mode" not in st.session_state:
    st.session_state.mode = None

# Carte 1 : RCXY 2 axes surfacique
with col1:
    st.markdown("#### 🧱 RCXY 2 axes surfacique")
    st.write("À partir d’un RC3 (3 axes), suppression de l’axe Z, "
             "en gardant le chariot Z comme interface robot.")
    if st.button("Ouvrir", key="btn_rcxy"):
        st.session_state.mode = "rcxy_2axes"

# Carte 2 : Robot cantilever (placeholder)
with col2:
    st.markdown("#### 🏗️ Robot cantilever")
    st.write("Bientôt disponible.")
    st.button("Bientôt", key="btn_cantilever", disabled=True)

# Carte 3 : Axes verticaux (placeholder)
with col3:
    st.markdown("#### ⬆️ Axes verticaux")
    st.write("Bientôt disponible.")
    st.button("Bientôt", key="btn_verticaux", disabled=True)

# Carte 4 : Axes verticaux sur axe X (placeholder)
with col4:
    st.markdown("#### ↗️ Axes verticaux sur axe X")
    st.write("Bientôt disponible.")
    st.button("Bientôt", key="btn_verticaux_x", disabled=True)


st.markdown("---")

# ------- Panneau correspondant au mode sélectionné -------

if st.session_state.mode == "rcxy_2axes":
    st.subheader("RC3 → RCXY 2 axes surfacique")

    st.write(
        "1️⃣ Charge un XML de robot cartésien 3 axes (RC3). "
        "2️⃣ L’outil prépare un RCXY 2 axes surfacique et propose un COMMENTAIRE. "
        "3️⃣ Tu peux modifier le commentaire ci-dessous. "
        "4️⃣ Clique sur « Convertir avec ce commentaire » pour générer le XML final."
    )

    uploaded_file = st.file_uploader(
        "Fichier XML RC3 (3 axes)",
        type=["xml"],
        key="upload_rc3"
    )

    # On stocke le XML brut dans la session pour pouvoir le réutiliser
    if uploaded_file is not None:
        xml_in = uploaded_file.read()
        st.session_state["xml_in"] = xml_in

        # Calcul du commentaire par défaut à partir du XML
        try:
            default_comment = extract_default_comment(xml_in)
        except Exception as e:
            st.error(f"Erreur pendant l'analyse du fichier : {e}")
            default_comment = ""

        # Zone de texte éditable pour le commentaire
        st.markdown("#### Commentaire BOM")
        st.text_area(
            "Texte du commentaire (modifie librement avant conversion) :",
            value=default_comment,
            key="comment_text",
            height=150
        )

        if st.button("Convertir avec ce commentaire"):
            try:
                user_comment = st.session_state.get("comment_text", default_comment)
                xml_out = convert_rc3_to_rcxy(xml_in, comment_override=user_comment)

                st.success("Conversion terminée ✅")

                st.download_button(
                    label="💾 Télécharger le XML RCXY 2 axes surfacique",
                    data=xml_out,
                    file_name="RCXY_2axes_surfacique.xml",
                    mime="application/xml"
                )

                with st.expander("Voir un extrait du XML généré"):
                    st.code(xml_out.decode("utf-8")[:1500])

            except Exception as e:
                st.error(f"Erreur pendant la conversion : {e}")
    else:
        st.info("Charge un fichier XML RC3 pour afficher et éditer le commentaire.")
else:
    st.info("Sélectionne un produit ci-dessus pour commencer.")
