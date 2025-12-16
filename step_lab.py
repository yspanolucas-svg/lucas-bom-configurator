import io
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st


# ============================================================
# STEP Lab (Option B) : Assemblage STEP par réécriture texte
# - Pas de CAO, pas de pythonOCC
# - On merge ES + RC2 et on crée un SHAPE_REPRESENTATION
#   contenant 2 MAPPED_ITEM, dont RC2 avec transformation.
# ============================================================

@dataclass
class StepParse:
    header_lines: List[str]
    data_lines_raw: List[str]          # lignes DATA brutes
    entities: Dict[int, str]           # id -> "ENTITY(...);"
    end_lines: List[str]               # ENDSEC/EOF


# ----------------------------
# Utils
# ----------------------------
def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _split_step_sections(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Retourne (header_lines, data_lines, end_lines)
    """
    lines = text.splitlines(True)  # keep \n
    header, data, tail = [], [], []
    mode = "HEADER"
    for ln in lines:
        up = ln.upper()
        if "DATA;" in up:
            mode = "DATA"
            header.append(ln)
            continue
        if "ENDSEC;" in up and mode == "DATA":
            mode = "TAIL"
            tail.append(ln)
            continue

        if mode == "HEADER":
            header.append(ln)
        elif mode == "DATA":
            data.append(ln)
        else:
            tail.append(ln)
    return header, data, tail


def _collect_entities(data_lines: List[str]) -> Dict[int, str]:
    """
    Collecte des entités STEP en gérant les entités sur plusieurs lignes.
    On assemble jusqu'au ';'
    """
    entities: Dict[int, str] = {}
    buf = ""
    for ln in data_lines:
        if not ln.strip():
            continue
        buf += ln
        if ";" not in ln:
            continue

        # parfois plusieurs ';' sur une ligne (rare) → on split
        parts = buf.split(";")
        for i, part in enumerate(parts[:-1]):
            s = (part + ";").strip()
            m = re.match(r"^\s*#(\d+)\s*=\s*(.+);\s*$", s, flags=re.IGNORECASE | re.DOTALL)
            if m:
                eid = int(m.group(1))
                body = m.group(2).strip()
                entities[eid] = f"{body};"
        buf = parts[-1]  # reste
    return entities


def parse_step(step_bytes: bytes) -> StepParse:
    txt = step_bytes.decode("utf-8", errors="ignore")
    header, data, tail = _split_step_sections(txt)
    ent = _collect_entities(data)
    return StepParse(header_lines=header, data_lines_raw=data, entities=ent, end_lines=tail)


def max_entity_id(entities: Dict[int, str]) -> int:
    return max(entities.keys()) if entities else 0


def renumber_entities(entities: Dict[int, str], offset: int) -> Dict[int, str]:
    """
    Renumérote toutes les références #id dans le corps.
    """
    if offset == 0:
        return dict(entities)

    # Regex #123
    ref_re = re.compile(r"#(\d+)\b")

    new_entities: Dict[int, str] = {}
    for old_id, body in entities.items():
        new_id = old_id + offset

        def _sub(m):
            return f"#{int(m.group(1)) + offset}"

        new_body = ref_re.sub(_sub, body)
        new_entities[new_id] = new_body
    return new_entities


def find_first_id_containing(entities: Dict[int, str], keywords: List[str]) -> Optional[int]:
    """
    Trouve la première entité dont le type contient un des keywords.
    Ex: "GEOMETRIC_REPRESENTATION_CONTEXT"
    """
    for eid in sorted(entities.keys()):
        up = entities[eid].upper()
        for kw in keywords:
            if kw.upper() in up:
                return eid
    return None


def find_first_shape_representation_id(entities: Dict[int, str]) -> Optional[int]:
    """
    Heuristique: on préfère ADVANCED_BREP_SHAPE_REPRESENTATION
    sinon SHAPE_REPRESENTATION
    """
    for kw in ["ADVANCED_BREP_SHAPE_REPRESENTATION", "MANIFOLD_SURFACE_SHAPE_REPRESENTATION", "SHAPE_REPRESENTATION"]:
        eid = find_first_id_containing(entities, [kw])
        if eid is not None:
            return eid
    return None


def replace_shape_definition_representation_target(
    entities: Dict[int, str],
    new_shape_rep_id: int
) -> bool:
    """
    Modifie la première entité SHAPE_DEFINITION_REPRESENTATION pour pointer vers new_shape_rep_id.
    Retourne True si modifié.
    """
    for eid in sorted(entities.keys()):
        body = entities[eid]
        if "SHAPE_DEFINITION_REPRESENTATION" in body.upper():
            # pattern: SHAPE_DEFINITION_REPRESENTATION(#a,#b);
            # on remplace le 2e argument par #new
            m = re.match(r"^\s*SHAPE_DEFINITION_REPRESENTATION\s*\(\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;\s*$",
                         body, flags=re.IGNORECASE)
            if m:
                a = m.group(1)
                entities[eid] = f"SHAPE_DEFINITION_REPRESENTATION({a},#{new_shape_rep_id});"
                return True
    return False


def build_step_assembly_option_b(
    es_bytes: bytes,
    rc2_bytes: bytes,
    # transformation RC2 -> monde (mm)
    tx_mm: float = 679.0,
    ty_mm: float = 811.0,
    tz_mm: float = 3334.5,
) -> bytes:
    """
    Construit un STEP assemblé :
    - ES = base (identité)
    - RC2 = MAPPED_ITEM avec RotY(+90°) + translation (tx,ty,tz)
    Sortie: un STEP texte unique.
    """
    es = parse_step(es_bytes)
    rc2 = parse_step(rc2_bytes)

    # 1) Renumérotation RC2 pour éviter collisions
    es_max = max_entity_id(es.entities)
    offset = es_max + 1000  # marge confortable
    rc2_renum = renumber_entities(rc2.entities, offset)

    # 2) Fusion entités (ES + RC2)
    merged = dict(es.entities)
    merged.update(rc2_renum)

    # 3) Trouver context (on reprend celui de ES)
    ctx_id = find_first_id_containing(merged, ["GEOMETRIC_REPRESENTATION_CONTEXT"])
    if ctx_id is None:
        # fallback: pas bloquant pour certains viewers, mais on tente quand même
        raise ValueError("Impossible de trouver GEOMETRIC_REPRESENTATION_CONTEXT dans le STEP ES.")

    # 4) Trouver shape representations de ES et RC2
    es_shape_id = find_first_shape_representation_id(es.entities)
    if es_shape_id is None:
        raise ValueError("Impossible de trouver SHAPE_REPRESENTATION dans ES.")
    rc2_shape_id = find_first_shape_representation_id(rc2_renum)
    if rc2_shape_id is None:
        raise ValueError("Impossible de trouver SHAPE_REPRESENTATION dans RC2.")

    # 5) Créer les nouvelles entités d’assemblage
    new_id_start = max_entity_id(merged) + 1
    nid = new_id_start

    def add(entity: str) -> int:
        nonlocal nid
        merged[nid] = entity.strip() if entity.strip().endswith(";") else (entity.strip() + ";")
        nid += 1
        return nid - 1

    # ---- ES mapping (identité) ----
    # AXIS2_PLACEMENT_3D('',#loc,#axis,#ref)
    # origin 0,0,0 ; axis Z=(0,0,1) ; ref X=(1,0,0)
    es_loc = add(f"CARTESIAN_POINT('',(0.,0.,0.))")
    es_axis = add(f"DIRECTION('',(0.,0.,1.))")
    es_ref  = add(f"DIRECTION('',(1.,0.,0.))")
    es_ax2  = add(f"AXIS2_PLACEMENT_3D('',#{es_loc},#{es_axis},#{es_ref})")
    es_map  = add(f"REPRESENTATION_MAP(#{es_ax2},#{es_shape_id})")

    # transformation opérateur identité
    es_t_axis1 = add(f"DIRECTION('',(1.,0.,0.))")
    es_t_axis2 = add(f"DIRECTION('',(0.,1.,0.))")
    es_t_axis3 = add(f"DIRECTION('',(0.,0.,1.))")
    es_t_loc   = add(f"CARTESIAN_POINT('',(0.,0.,0.))")
    es_xform   = add(f"CARTESIAN_TRANSFORMATION_OPERATOR_3D('',#{es_t_axis1},#{es_t_axis2},#{es_t_loc},1.,#{es_t_axis3})")
    es_mapped  = add(f"MAPPED_ITEM(#{es_map},#{es_xform})")

    # ---- RC2 mapping (RotY +90 + translation) ----
    # Matrice (colonnes X,Y,Z locaux dans global) issue de ton asseb.stp :
    # X=(0,0,-1) ; Y=(0,1,0) ; Z=(1,0,0)
    rc2_loc = add(f"CARTESIAN_POINT('',(0.,0.,0.))")
    rc2_axis = add(f"DIRECTION('',(0.,0.,1.))")
    rc2_ref  = add(f"DIRECTION('',(1.,0.,0.))")
    rc2_ax2  = add(f"AXIS2_PLACEMENT_3D('',#{rc2_loc},#{rc2_axis},#{rc2_ref})")
    rc2_map  = add(f"REPRESENTATION_MAP(#{rc2_ax2},#{rc2_shape_id})")

    # Transformation operator: axis1/axis2/axis3 + origin
    # axis1 = X = (0,0,-1)
    # axis2 = Y = (0,1,0)
    # axis3 = Z = (1,0,0)
    r_axis1 = add(f"DIRECTION('',(0.,0.,-1.))")
    r_axis2 = add(f"DIRECTION('',(0.,1.,0.))")
    r_axis3 = add(f"DIRECTION('',(1.,0.,0.))")
    r_loc   = add(f"CARTESIAN_POINT('',({tx_mm:.6g},{ty_mm:.6g},{tz_mm:.6g}))")
    r_xform = add(f"CARTESIAN_TRANSFORMATION_OPERATOR_3D('',#{r_axis1},#{r_axis2},#{r_loc},1.,#{r_axis3})")
    rc2_mapped = add(f"MAPPED_ITEM(#{rc2_map},#{r_xform})")

    # 6) Nouveau SHAPE_REPRESENTATION global
    # SHAPE_REPRESENTATION('',(items),#context);
    new_shape_rep = add(f"SHAPE_REPRESENTATION('',(#{es_mapped},#{rc2_mapped}),#{ctx_id})")

    # 7) Pointer la définition de forme principale vers cette nouvelle représentation
    modified = replace_shape_definition_representation_target(merged, new_shape_rep)
    if not modified:
        # fallback : si pas trouvé, on tente un SHAPE_DEFINITION_REPRESENTATION minimal
        # en accrochant au premier PRODUCT_DEFINITION trouvé
        pd_id = find_first_id_containing(merged, ["PRODUCT_DEFINITION"])
        if pd_id is not None:
            add(f"SHAPE_DEFINITION_REPRESENTATION(#{pd_id},#{new_shape_rep})")
        # sinon, on laisse comme ça (certains viewers afficheront quand même la SHAPE_REPRESENTATION)

    # 8) Recomposer le fichier STEP final
    # On reprend le HEADER de ES tel quel, puis DATA; puis toutes les entités triées, ENDSEC; EOF
    out = []
    out.extend(es.header_lines)

    # S’assurer qu’on a DATA;
    if not any("DATA;" in ln.upper() for ln in out):
        out.append("DATA;\n")

    # Entités triées
    out.append("\n")
    for eid in sorted(merged.keys()):
        out.append(f"#{eid}={merged[eid].rstrip()}\n")
    out.append("\nENDSEC;\n")
    out.append("END-ISO-10303-21;\n")

    return "".join(out).encode("utf-8")


# ============================================================
# STREAMLIT PANEL (même UX que PDF)
# ============================================================
def render_step_lab_panel():
    st.subheader("STEP Lab – Assemblage Cantilever (sans CAO)")

    with st.expander("Mode d’emploi (aperçu rapide)", expanded=True):
        st.markdown(
            """
**Objectif :** assembler automatiquement 2 fichiers STEP (Ensemble 1 + Ensemble 2) et exporter un STEP final,
sans ouvrir de logiciel de CAO.

**Déroulé**
1) Charger le STEP de l’Ensemble 1  
2) Charger le STEP de l’Ensemble 2  
3) Cliquer sur **Assembler**  
4) Télécharger le STEP assemblé

Le positionnement de l’Ensemble 2 est basé sur une règle déterministe (rotation + translation),
équivalente à ce que tu as fait manuellement dans ton fichier STEP assemblé de référence.
            """
        )

    c1, c2 = st.columns(2)
    with c1:
        name1 = st.text_input("Nom Ensemble 1", value="Ensemble 1", key="step_name_1")
        step1 = st.file_uploader("STEP – Ensemble 1", type=["stp", "step"], key="step_file_1")
    with c2:
        name2 = st.text_input("Nom Ensemble 2", value="Ensemble 2", key="step_name_2")
        step2 = st.file_uploader("STEP – Ensemble 2", type=["stp", "step"], key="step_file_2")

    st.markdown("---")
    st.caption("Paramètres de placement (cantilever) – visibles pour contrôle, mais utilisés automatiquement.")

    colA, colB, colC = st.columns(3)
    with colA:
        tx = st.number_input("Tx (mm)", value=679.0, step=1.0, format="%.3f")
    with colB:
        ty = st.number_input("Ty (mm)", value=811.0, step=1.0, format="%.3f")
    with colC:
        tz = st.number_input("Tz (mm)", value=3334.5, step=1.0, format="%.3f")

    st.info("Règle actuelle : rotation Y +90° appliquée à l’Ensemble 2, puis translation (Tx, Ty, Tz).")

    if not step1 or not step2:
        st.warning("Veuillez charger deux fichiers STEP.")
        return

    b1 = step1.read()
    b2 = step2.read()

    if st.button("Assembler", type="primary"):
        try:
            out = build_step_assembly_option_b(
                es_bytes=b1,
                rc2_bytes=b2,
                tx_mm=tx,
                ty_mm=ty,
                tz_mm=tz,
            )
            st.success("Assemblage STEP généré ✅")
            st.download_button(
                "💾 Télécharger le STEP assemblé",
                data=out,
                file_name="CANTILEVER_ASSEMBLY.stp",
                mime="application/step",
            )
        except Exception as e:
            st.error(f"Erreur pendant l’assemblage STEP : {e}")
