import re
import math
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import streamlit as st


# ============================================================
# STEP Lab (Option B)
# - Assemblage STEP par réécriture texte (sans CAO)
# - Apprentissage 1 fois via un STEP assemblé correct
# - Sauvegarde d’un preset JSON (axes + origin) dans /presets
# ============================================================

PRESET_PATH = Path("presets/step_cantilever_default.json")


# ----------------------------
# Presets
# ----------------------------
def load_step_preset() -> Optional[dict]:
    if PRESET_PATH.exists():
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_step_preset(axis1, axis2, axis3, origin) -> None:
    PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "axis1": [float(axis1[0]), float(axis1[1]), float(axis1[2])],
        "axis2": [float(axis2[0]), float(axis2[1]), float(axis2[2])],
        "axis3": [float(axis3[0]), float(axis3[1]), float(axis3[2])],
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
    }
    with open(PRESET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def default_preset_dict() -> dict:
    return {
        "axis1": [1.0, 0.0, 0.0],
        "axis2": [0.0, 1.0, 0.0],
        "axis3": [0.0, 0.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
    }


# ----------------------------
# Parser STEP (texte)
# ----------------------------
@dataclass
class StepParse:
    header_lines: List[str]
    data_lines_raw: List[str]
    entities: Dict[int, str]     # id -> "ENTITY(...);"
    end_lines: List[str]


def _split_step_sections(text: str) -> Tuple[List[str], List[str], List[str]]:
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
    Collecte des entités STEP en gérant les entités multi-lignes.
    """
    entities: Dict[int, str] = {}
    buf = ""

    for ln in data_lines:
        if not ln.strip():
            continue

        buf += ln
        if ";" not in ln:
            continue

        parts = buf.split(";")
        for part in parts[:-1]:
            s = (part + ";").strip()
            m = re.match(r"^\s*#(\d+)\s*=\s*(.+);\s*$", s, flags=re.IGNORECASE | re.DOTALL)
            if m:
                eid = int(m.group(1))
                body = m.group(2).strip()
                entities[eid] = f"{body};"

        buf = parts[-1]

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

    ref_re = re.compile(r"#(\d+)\b")
    out: Dict[int, str] = {}

    for old_id, body in entities.items():
        new_id = old_id + offset

        def _sub(m):
            return f"#{int(m.group(1)) + offset}"

        out[new_id] = ref_re.sub(_sub, body)

    return out


def find_first_id_containing(entities: Dict[int, str], keywords: List[str]) -> Optional[int]:
    for eid in sorted(entities.keys()):
        up = entities[eid].upper()
        for kw in keywords:
            if kw.upper() in up:
                return eid
    return None


def find_first_shape_representation_id(entities: Dict[int, str]) -> Optional[int]:
    """
    Heuristique : on prend en priorité ADVANCED_BREP_SHAPE_REPRESENTATION
    """
    for kw in [
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        "SHAPE_REPRESENTATION",
    ]:
        eid = find_first_id_containing(entities, [kw])
        if eid is not None:
            return eid
    return None


def replace_shape_definition_representation_target(
    entities: Dict[int, str],
    new_shape_rep_id: int
) -> bool:
    """
    Modifie la première SHAPE_DEFINITION_REPRESENTATION pour pointer vers new_shape_rep_id.
    """
    for eid in sorted(entities.keys()):
        body = entities[eid]
        if "SHAPE_DEFINITION_REPRESENTATION" in body.upper():
            m = re.match(
                r"^\s*SHAPE_DEFINITION_REPRESENTATION\s*\(\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;\s*$",
                body,
                flags=re.IGNORECASE
            )
            if m:
                a = m.group(1)
                entities[eid] = f"SHAPE_DEFINITION_REPRESENTATION({a},#{new_shape_rep_id});"
                return True
    return False


# ----------------------------
# Extraction transformation depuis un STEP assemblé référence
# ----------------------------
def _parse_cartesian_point(body: str) -> Optional[Tuple[float, float, float]]:
    m = re.search(r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*([^\)]+)\s*\)\s*\)", body, flags=re.I)
    if not m:
        return None
    coords = m.group(1).replace(" ", "")
    parts = coords.split(",")
    if len(parts) != 3:
        return None

    def f(x: str) -> float:
        return float(x.replace("E", "e"))

    try:
        return (f(parts[0]), f(parts[1]), f(parts[2]))
    except Exception:
        return None


def _parse_direction(body: str) -> Optional[Tuple[float, float, float]]:
    m = re.search(r"DIRECTION\s*\(\s*'[^']*'\s*,\s*\(\s*([^\)]+)\s*\)\s*\)", body, flags=re.I)
    if not m:
        return None
    coords = m.group(1).replace(" ", "")
    parts = coords.split(",")
    if len(parts) != 3:
        return None

    def f(x: str) -> float:
        return float(x.replace("E", "e"))

    try:
        return (f(parts[0]), f(parts[1]), f(parts[2]))
    except Exception:
        return None


def _norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def extract_transform_from_reference(ref_bytes: bytes) -> Tuple[
    Tuple[float,float,float],
    Tuple[float,float,float],
    Tuple[float,float,float],
    Tuple[float,float,float]
]:
    """
    Cherche la meilleure CARTESIAN_TRANSFORMATION_OPERATOR_3D non-identité.
    Heuristique cantilever :
    - on ignore origine ~ (0,0,0)
    - on prend la transformation avec la plus grande distance d'origine
    """
    ref = parse_step(ref_bytes)
    ents = ref.entities

    points: Dict[int, Tuple[float,float,float]] = {}
    dirs: Dict[int, Tuple[float,float,float]] = {}

    for eid, body in ents.items():
        up = body.upper().strip()
        if up.startswith("CARTESIAN_POINT"):
            p = _parse_cartesian_point(body)
            if p:
                points[eid] = p
        elif up.startswith("DIRECTION"):
            d = _parse_direction(body)
            if d:
                dirs[eid] = d

    re_cto = re.compile(
        r"^CARTESIAN_TRANSFORMATION_OPERATOR_3D\s*\(\s*'[^']*'\s*,\s*(#\d+)\s*,\s*(#\d+)\s*,\s*(#\d+)\s*,\s*([^,]+)\s*,\s*(#\d+)\s*\)\s*;\s*$",
        flags=re.I
    )

    best = None
    best_score = -1.0

    for eid in sorted(ents.keys()):
        body = ents[eid].strip()
        m = re_cto.match(body)
        if not m:
            continue

        a1 = int(m.group(1)[1:])
        a2 = int(m.group(2)[1:])
        loc = int(m.group(3)[1:])
        a3 = int(m.group(5)[1:])

        if loc not in points or a1 not in dirs or a2 not in dirs or a3 not in dirs:
            continue

        origin = points[loc]
        d_origin = _norm(origin)

        if d_origin < 1e-6:
            continue

        # axes proches de unitaires (tolérance)
        if abs(_norm(dirs[a1]) - 1.0) > 1e-2:
            continue
        if abs(_norm(dirs[a2]) - 1.0) > 1e-2:
            continue
        if abs(_norm(dirs[a3]) - 1.0) > 1e-2:
            continue

        if d_origin > best_score:
            best_score = d_origin
            best = (dirs[a1], dirs[a2], dirs[a3], origin)

    if best is None:
        raise ValueError("Impossible d’extraire une transformation non-identité dans la référence.")

    return best  # axis1, axis2, axis3, origin


# ----------------------------
# Construction STEP assemblé (Option B)
# ----------------------------
def build_step_assembly_option_b(
    base_bytes: bytes,
    part_bytes: bytes,
    axis1: Tuple[float,float,float],
    axis2: Tuple[float,float,float],
    axis3: Tuple[float,float,float],
    origin: Tuple[float,float,float],
) -> bytes:
    """
    Construit un STEP assemblé :
    - Base = ensemble maître (identité)
    - Part = ensemble ajouté placé via (axis1/axis2/axis3/origin)
    """
    base = parse_step(base_bytes)
    part = parse_step(part_bytes)

    # 1) Renumérotation PART pour éviter collisions
    base_max = max_entity_id(base.entities)
    offset = base_max + 1000
    part_renum = renumber_entities(part.entities, offset)

    # 2) Fusion entités (Base + Part)
    merged = dict(base.entities)
    merged.update(part_renum)

    # 3) Contexte (on reprend celui du fichier base)
    ctx_id = find_first_id_containing(merged, ["GEOMETRIC_REPRESENTATION_CONTEXT"])
    if ctx_id is None:
        raise ValueError("GEOMETRIC_REPRESENTATION_CONTEXT introuvable dans le STEP base.")

    # 4) Trouver les SHAPE_REPRESENTATION des deux ensembles
    base_shape_id = find_first_shape_representation_id(base.entities)
    if base_shape_id is None:
        raise ValueError("SHAPE_REPRESENTATION introuvable dans le STEP base.")
    part_shape_id = find_first_shape_representation_id(part_renum)
    if part_shape_id is None:
        raise ValueError("SHAPE_REPRESENTATION introuvable dans le STEP ajouté.")

    # 5) Ajout des entités nécessaires pour construire une SHAPE_REPRESENTATION globale
    nid = max_entity_id(merged) + 1

    def add(entity: str) -> int:
        nonlocal nid
        e = entity.strip()
        if not e.endswith(";"):
            e += ";"
        merged[nid] = e
        nid += 1
        return nid - 1

    # ---- Mapping base (identité)
    b_loc = add("CARTESIAN_POINT('',(0.,0.,0.))")
    b_axis = add("DIRECTION('',(0.,0.,1.))")
    b_ref  = add("DIRECTION('',(1.,0.,0.))")
    b_ax2  = add(f"AXIS2_PLACEMENT_3D('',#{b_loc},#{b_axis},#{b_ref})")
    b_map  = add(f"REPRESENTATION_MAP(#{b_ax2},#{base_shape_id})")

    b_a1 = add("DIRECTION('',(1.,0.,0.))")
    b_a2 = add("DIRECTION('',(0.,1.,0.))")
    b_a3 = add("DIRECTION('',(0.,0.,1.))")
    b_org = add("CARTESIAN_POINT('',(0.,0.,0.))")
    b_xf  = add(f"CARTESIAN_TRANSFORMATION_OPERATOR_3D('',#{b_a1},#{b_a2},#{b_org},1.,#{b_a3})")
    b_mapped = add(f"MAPPED_ITEM(#{b_map},#{b_xf})")

    # ---- Mapping part (transformation preset)
    p_loc = add("CARTESIAN_POINT('',(0.,0.,0.))")
    p_axis = add("DIRECTION('',(0.,0.,1.))")
    p_ref  = add("DIRECTION('',(1.,0.,0.))")
    p_ax2  = add(f"AXIS2_PLACEMENT_3D('',#{p_loc},#{p_axis},#{p_ref})")
    p_map  = add(f"REPRESENTATION_MAP(#{p_ax2},#{part_shape_id})")

    p_a1 = add(f"DIRECTION('',({axis1[0]:.6g},{axis1[1]:.6g},{axis1[2]:.6g}))")
    p_a2 = add(f"DIRECTION('',({axis2[0]:.6g},{axis2[1]:.6g},{axis2[2]:.6g}))")
    p_a3 = add(f"DIRECTION('',({axis3[0]:.6g},{axis3[1]:.6g},{axis3[2]:.6g}))")
    p_org = add(f"CARTESIAN_POINT('',({origin[0]:.6g},{origin[1]:.6g},{origin[2]:.6g}))")
    p_xf  = add(f"CARTESIAN_TRANSFORMATION_OPERATOR_3D('',#{p_a1},#{p_a2},#{p_org},1.,#{p_a3})")
    p_mapped = add(f"MAPPED_ITEM(#{p_map},#{p_xf})")

    # ---- Nouvelle représentation globale
    new_shape_rep = add(f"SHAPE_REPRESENTATION('',(#{b_mapped},#{p_mapped}),#{ctx_id})")

    # ---- Pointer la définition de forme principale vers la nouvelle shape rep
    modified = replace_shape_definition_representation_target(merged, new_shape_rep)
    if not modified:
        # fallback : ajouter une SHAPE_DEFINITION_REPRESENTATION si possible
        pd_id = find_first_id_containing(merged, ["PRODUCT_DEFINITION"])
        if pd_id is not None:
            add(f"SHAPE_DEFINITION_REPRESENTATION(#{pd_id},#{new_shape_rep})")

    # 6) Recomposer un fichier STEP propre (un seul modèle)
    out = []
    out.extend(base.header_lines)

    if not any("DATA;" in ln.upper() for ln in out):
        out.append("DATA;\n")

    out.append("\n")
    for eid in sorted(merged.keys()):
        out.append(f"#{eid}={merged[eid].rstrip()}\n")

    out.append("\nENDSEC;\n")
    out.append("END-ISO-10303-21;\n")
    return "".join(out).encode("utf-8")


# ============================================================
# STREAMLIT PANEL
# ============================================================
def render_step_lab_panel():
    st.subheader("🧊 STEP Lab – Assemblage cantilever (sans CAO)")

    st.markdown(
        """
**But :** assembler deux STEP (Ensemble A + Ensemble B) en appliquant une règle de placement **déterministe**,
comme pour le PDF Lab, sans ouvrir de logiciel CAO.

**Mode recommandé :**
1) **Apprendre une fois** la règle avec un STEP assemblé correct (référence).  
2) Réutiliser ensuite cette règle automatiquement via un **preset** (fichier JSON léger).

Le preset est enregistré ici : `presets/step_cantilever_default.json`
        """
    )

    preset = load_step_preset()
    colp1, colp2 = st.columns([2, 1])
    with colp1:
        if preset:
            st.success("Preset STEP trouvé ✅ (il sera utilisé pour l’assemblage)")
        else:
            st.warning("Aucun preset STEP trouvé. Apprenez-le avec une référence assemblée.")
    with colp2:
        if st.button("🧹 Réinitialiser le preset", help="Remet le preset à l’identité (0,0,0)"):
            save_step_preset(
                axis1=(1.0,0.0,0.0),
                axis2=(0.0,1.0,0.0),
                axis3=(0.0,0.0,1.0),
                origin=(0.0,0.0,0.0),
            )
            st.success("Preset réinitialisé.")
            st.rerun()

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1) Fichiers à assembler")
        step_a_up = st.file_uploader("STEP – Ensemble A (base)", type=["stp", "step"], key="step_a")
        step_b_up = st.file_uploader("STEP – Ensemble B (à placer)", type=["stp", "step"], key="step_b")

    with c2:
        st.markdown("### 2) Apprentissage (1 fois)")
        ref_up = st.file_uploader(
            "STEP référence assemblé (cantilever correct)",
            type=["stp", "step"],
            key="step_ref"
        )
        st.caption("Astuce : ce fichier ne sera pas stocké. On ne garde que la règle (axes + origin).")

    # Lire bytes 1 seule fois
    step_a_bytes = step_a_up.read() if step_a_up else None
    step_b_bytes = step_b_up.read() if step_b_up else None
    ref_bytes = ref_up.read() if ref_up else None

    # --- Apprentissage preset
    st.markdown("### Apprendre / mettre à jour le preset")
    if st.button("📚 Apprendre depuis la référence et enregistrer le preset", disabled=(ref_bytes is None)):
        try:
            a1, a2, a3, org = extract_transform_from_reference(ref_bytes)
            save_step_preset(a1, a2, a3, org)
            st.success("Preset cantilever enregistré ✅")
            st.json({
                "axis1": a1,
                "axis2": a2,
                "axis3": a3,
                "origin": org
            })
            st.info("Vous pouvez maintenant assembler sans recharger la référence.")
        except Exception as e:
            st.error(f"Erreur apprentissage : {e}")

    st.markdown("---")

    # --- Affichage preset actuel
    preset = load_step_preset() or default_preset_dict()
    with st.expander("Voir le preset actuel", expanded=False):
        st.json(preset)

    # --- Assemblage
    st.markdown("### Génération du STEP assemblé")
    if not step_a_bytes or not step_b_bytes:
        st.info("Chargez Ensemble A et Ensemble B pour activer l’assemblage.")
        return

    # Option debug : override manuel (facultatif)
    with st.expander("Réglages avancés (optionnel)", expanded=False):
        use_override = st.checkbox("Forcer une transformation manuelle (debug)", value=False)
        if use_override:
            tx = st.number_input("Tx", value=float(preset["origin"][0]), step=1.0, format="%.3f")
            ty = st.number_input("Ty", value=float(preset["origin"][1]), step=1.0, format="%.3f")
            tz = st.number_input("Tz", value=float(preset["origin"][2]), step=1.0, format="%.3f")

            a1x = st.number_input("Axis1 X", value=float(preset["axis1"][0]), step=0.1, format="%.3f")
            a1y = st.number_input("Axis1 Y", value=float(preset["axis1"][1]), step=0.1, format="%.3f")
            a1z = st.number_input("Axis1 Z", value=float(preset["axis1"][2]), step=0.1, format="%.3f")

            a2x = st.number_input("Axis2 X", value=float(preset["axis2"][0]), step=0.1, format="%.3f")
            a2y = st.number_input("Axis2 Y", value=float(preset["axis2"][1]), step=0.1, format="%.3f")
            a2z = st.number_input("Axis2 Z", value=float(preset["axis2"][2]), step=0.1, format="%.3f")

            a3x = st.number_input("Axis3 X", value=float(preset["axis3"][0]), step=0.1, format="%.3f")
            a3y = st.number_input("Axis3 Y", value=float(preset["axis3"][1]), step=0.1, format="%.3f")
            a3z = st.number_input("Axis3 Z", value=float(preset["axis3"][2]), step=0.1, format="%.3f")

    if st.button("⚙️ Assembler (avec le preset)", type="primary"):
        try:
            if use_override:
                axis1 = (a1x, a1y, a1z)
                axis2 = (a2x, a2y, a2z)
                axis3 = (a3x, a3y, a3z)
                origin = (tx, ty, tz)
            else:
                if not PRESET_PATH.exists():
                    st.error("Aucun preset trouvé. Apprenez-le avec une référence assemblée.")
                    return
                axis1 = tuple(preset["axis1"])
                axis2 = tuple(preset["axis2"])
                axis3 = tuple(preset["axis3"])
                origin = tuple(preset["origin"])

            out = build_step_assembly_option_b(
                base_bytes=step_a_bytes,
                part_bytes=step_b_bytes,
                axis1=axis1,
                axis2=axis2,
                axis3=axis3,
                origin=origin,
            )

            st.success("STEP assemblé généré ✅")
            st.download_button(
                "💾 Télécharger le STEP assemblé",
                data=out,
                file_name="ASSEMBLY_STEP_LAB.stp",
                mime="application/step",
            )

        except Exception as e:
            st.error(f"Erreur pendant l’assemblage : {e}")
