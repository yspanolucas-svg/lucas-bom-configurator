import re
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st


# ============================================================
# Preset (manuel) : translation (mm) + rotations (deg)
# ============================================================
PRESET_PATH = Path("presets/step_lab_default.json")


def _default_preset() -> dict:
    return {
        "translation_mm": {"tx": 0.0, "ty": 0.0, "tz": 0.0},
        "rotation_deg": {"rx": 0.0, "ry": 0.0, "rz": 0.0},
    }


def load_step_preset() -> dict:
    if PRESET_PATH.exists():
        try:
            with open(PRESET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # complétion soft
            d = _default_preset()
            d["translation_mm"].update(data.get("translation_mm", {}))
            d["rotation_deg"].update(data.get("rotation_deg", {}))
            return d
        except Exception:
            return _default_preset()
    return _default_preset()


def save_step_preset(preset: dict) -> None:
    PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRESET_PATH, "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=2)


# ============================================================
# STEP parsing (robuste multi-lignes)
# ============================================================
def _split_step_sections(text: str) -> Tuple[List[str], List[str], List[str]]:
    lines = text.splitlines(True)
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
            m = re.match(r"^\s*#(\d+)\s*=\s*(.+);\s*$", s, flags=re.I | re.S)
            if m:
                eid = int(m.group(1))
                body = m.group(2).strip()
                entities[eid] = f"{body};"
        buf = parts[-1]
    return entities


def parse_step(step_bytes: bytes):
    txt = step_bytes.decode("utf-8", errors="ignore")
    header, data, tail = _split_step_sections(txt)
    ents = _collect_entities(data)
    return header, ents, tail


def max_entity_id(entities: Dict[int, str]) -> int:
    return max(entities.keys()) if entities else 0


def renumber_entities(entities: Dict[int, str], offset: int) -> Dict[int, str]:
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
    for kw in [
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        "SHAPE_REPRESENTATION",
    ]:
        eid = find_first_id_containing(entities, [kw])
        if eid is not None:
            return eid
    return None


def replace_shape_definition_representation_target(entities: Dict[int, str], new_shape_rep_id: int) -> bool:
    for eid in sorted(entities.keys()):
        body = entities[eid]
        if "SHAPE_DEFINITION_REPRESENTATION" in body.upper():
            m = re.match(
                r"^\s*SHAPE_DEFINITION_REPRESENTATION\s*\(\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;\s*$",
                body,
                flags=re.I
            )
            if m:
                a = m.group(1)
                entities[eid] = f"SHAPE_DEFINITION_REPRESENTATION({a},#{new_shape_rep_id});"
                return True
    return False


# ============================================================
# Math : rotation (Euler) -> axes (X,Y,Z) pour STEP
# ============================================================
def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _matmul(A, B):
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
         A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
         A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
         A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
         A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]],
        [A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
         A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
         A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]],
    ]


def _rot_matrix(rx_deg: float, ry_deg: float, rz_deg: float):
    rx = _deg2rad(rx_deg)
    ry = _deg2rad(ry_deg)
    rz = _deg2rad(rz_deg)

    Rx = [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(rx), -math.sin(rx)],
        [0.0, math.sin(rx),  math.cos(rx)],
    ]
    Ry = [
        [ math.cos(ry), 0.0, math.sin(ry)],
        [0.0,          1.0, 0.0],
        [-math.sin(ry), 0.0, math.cos(ry)],
    ]
    Rz = [
        [math.cos(rz), -math.sin(rz), 0.0],
        [math.sin(rz),  math.cos(rz), 0.0],
        [0.0,           0.0,          1.0],
    ]

    # Convention simple et standard : R = Rz * Ry * Rx
    return _matmul(_matmul(Rz, Ry), Rx)


def _axes_from_rot(rx_deg: float, ry_deg: float, rz_deg: float):
    R = _rot_matrix(rx_deg, ry_deg, rz_deg)

    # On prend les colonnes comme axes (X,Y,Z) transformés
    axis1 = (R[0][0], R[1][0], R[2][0])  # X'
    axis2 = (R[0][1], R[1][1], R[2][1])  # Y'
    axis3 = (R[0][2], R[1][2], R[2][2])  # Z'
    return axis1, axis2, axis3


# ============================================================
# Assemblage Option B : on garde A et on "place" B via MAPPED_ITEM
# ============================================================
def build_step_assembly_option_b(
    base_bytes: bytes,
    part_bytes: bytes,
    tx_mm: float,
    ty_mm: float,
    tz_mm: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> bytes:
    base_header, base_ents, _ = parse_step(base_bytes)
    _, part_ents, _ = parse_step(part_bytes)

    base_max = max_entity_id(base_ents)
    offset = base_max + 1000
    part_renum = renumber_entities(part_ents, offset)

    merged = dict(base_ents)
    merged.update(part_renum)

    ctx_id = find_first_id_containing(merged, ["GEOMETRIC_REPRESENTATION_CONTEXT"])
    if ctx_id is None:
        raise ValueError("GEOMETRIC_REPRESENTATION_CONTEXT introuvable dans le STEP base (Ensemble A).")

    base_shape_id = find_first_shape_representation_id(base_ents)
    if base_shape_id is None:
        raise ValueError("SHAPE_REPRESENTATION introuvable dans le STEP base (Ensemble A).")

    part_shape_id = find_first_shape_representation_id(part_renum)
    if part_shape_id is None:
        raise ValueError("SHAPE_REPRESENTATION introuvable dans le STEP ajouté (Ensemble B).")

    nid = max_entity_id(merged) + 1

    def add(entity: str) -> int:
        nonlocal nid
        e = entity.strip()
        if not e.endswith(";"):
            e += ";"
        merged[nid] = e
        nid += 1
        return nid - 1

    # ----------------
    # Base mapping (identité)
    # ----------------
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

    # ----------------
    # Part mapping (translation + rotation)
    # ----------------
    p_loc = add("CARTESIAN_POINT('',(0.,0.,0.))")
    p_axis = add("DIRECTION('',(0.,0.,1.))")
    p_ref  = add("DIRECTION('',(1.,0.,0.))")
    p_ax2  = add(f"AXIS2_PLACEMENT_3D('',#{p_loc},#{p_axis},#{p_ref})")
    p_map  = add(f"REPRESENTATION_MAP(#{p_ax2},#{part_shape_id})")

    axis1, axis2, axis3 = _axes_from_rot(rx_deg, ry_deg, rz_deg)

    p_a1 = add(f"DIRECTION('',({axis1[0]:.9g},{axis1[1]:.9g},{axis1[2]:.9g}))")
    p_a2 = add(f"DIRECTION('',({axis2[0]:.9g},{axis2[1]:.9g},{axis2[2]:.9g}))")
    p_a3 = add(f"DIRECTION('',({axis3[0]:.9g},{axis3[1]:.9g},{axis3[2]:.9g}))")
    p_org = add(f"CARTESIAN_POINT('',({tx_mm:.9g},{ty_mm:.9g},{tz_mm:.9g}))")
    p_xf  = add(f"CARTESIAN_TRANSFORMATION_OPERATOR_3D('',#{p_a1},#{p_a2},#{p_org},1.,#{p_a3})")
    p_mapped = add(f"MAPPED_ITEM(#{p_map},#{p_xf})")

    # Nouvelle shape rep = (base + part placé)
    new_shape_rep = add(f"SHAPE_REPRESENTATION('',(#{b_mapped},#{p_mapped}),#{ctx_id})")

    ok = replace_shape_definition_representation_target(merged, new_shape_rep)
    if not ok:
        # fallback (rare) : on ajoute une nouvelle SHAPE_DEFINITION_REPRESENTATION
        pd_id = find_first_id_containing(merged, ["PRODUCT_DEFINITION("])
        if pd_id is not None:
            add(f"SHAPE_DEFINITION_REPRESENTATION(#{pd_id},#{new_shape_rep})")

    # Export : on repart du header de A
    out = []
    out.extend(base_header)
    if not any("DATA;" in ln.upper() for ln in out):
        out.append("DATA;\n")

    out.append("\n")
    for eid in sorted(merged.keys()):
        out.append(f"#{eid}={merged[eid].rstrip()}\n")

    out.append("\nENDSEC;\n")
    out.append("END-ISO-10303-21;\n")
    return "".join(out).encode("utf-8")


# ============================================================
# STREAMLIT UI
# ============================================================
def render_step_lab_panel():
    st.subheader("🧊 STEP Lab – Positionnement manuel (mm) du 2-axes sur l’axe élevé")

    st.markdown(
        """
**Objectif (phase 1) :** positionner proprement l’Ensemble B (2-axes) sur l’Ensemble A (axe élevé),  
**sans CAO** et **avec réglage manuel** (translation + rotation).

➡️ *On ne supprime rien pour l’instant : on stabilise d’abord le placement.*
        """
    )

    preset = load_step_preset()

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Fichiers STEP")
        up_a = st.file_uploader("STEP – Ensemble A (axe élevé)", type=["stp", "step"], key="stp_a")
        up_b = st.file_uploader("STEP – Ensemble B (2-axes)", type=["stp", "step"], key="stp_b")

    with c2:
        st.markdown("### Preset")
        st.write(f"Fichier : `{PRESET_PATH.as_posix()}`")
        if PRESET_PATH.exists():
            st.success("Preset trouvé ✅")
        else:
            st.warning("Aucun preset : il sera créé au premier enregistrement.")

        if st.button("↩️ Réinitialiser aux valeurs du preset"):
            st.session_state["tx"] = float(preset["translation_mm"]["tx"])
            st.session_state["ty"] = float(preset["translation_mm"]["ty"])
            st.session_state["tz"] = float(preset["translation_mm"]["tz"])
            st.session_state["rx"] = float(preset["rotation_deg"]["rx"])
            st.session_state["ry"] = float(preset["rotation_deg"]["ry"])
            st.session_state["rz"] = float(preset["rotation_deg"]["rz"])
            st.rerun()

    a_bytes = up_a.read() if up_a else None
    b_bytes = up_b.read() if up_b else None

    st.markdown("---")
    st.markdown("### Réglages de placement (Ensemble B)")

    # init session_state
    if "tx" not in st.session_state:
        st.session_state["tx"] = float(preset["translation_mm"]["tx"])
    if "ty" not in st.session_state:
        st.session_state["ty"] = float(preset["translation_mm"]["ty"])
    if "tz" not in st.session_state:
        st.session_state["tz"] = float(preset["translation_mm"]["tz"])
    if "rx" not in st.session_state:
        st.session_state["rx"] = float(preset["rotation_deg"]["rx"])
    if "ry" not in st.session_state:
        st.session_state["ry"] = float(preset["rotation_deg"]["ry"])
    if "rz" not in st.session_state:
        st.session_state["rz"] = float(preset["rotation_deg"]["rz"])

    colT, colR = st.columns(2)

    with colT:
        st.markdown("#### Translation (mm)")
        st.number_input("Tx (mm)", key="tx", step=1.0, format="%.3f")
        st.number_input("Ty (mm)", key="ty", step=1.0, format="%.3f")
        st.number_input("Tz (mm)", key="tz", step=1.0, format="%.3f")

    with colR:
        st.markdown("#### Rotation (°)")
        st.number_input("Rx (°)", key="rx", step=1.0, format="%.3f")
        st.number_input("Ry (°)", key="ry", step=1.0, format="%.3f")
        st.number_input("Rz (°)", key="rz", step=1.0, format="%.3f")

    st.caption(
        "Astuce : commencez par régler **Tz**, puis **Ty**, puis **Tx**. "
        "Les rotations servent surtout si le 2-axes arrive tourné dans le STEP."
    )

    st.markdown("---")
    cA, cB = st.columns(2)

    with cA:
        if st.button("💾 Enregistrer ces valeurs comme preset par défaut"):
            new_preset = {
                "translation_mm": {"tx": float(st.session_state["tx"]), "ty": float(st.session_state["ty"]), "tz": float(st.session_state["tz"])},
                "rotation_deg": {"rx": float(st.session_state["rx"]), "ry": float(st.session_state["ry"]), "rz": float(st.session_state["rz"])},
            }
            save_step_preset(new_preset)
            st.success("Preset enregistré ✅")

    with cB:
        assemble_disabled = (a_bytes is None) or (b_bytes is None)
        if st.button("⚙️ Générer le STEP assemblé", type="primary", disabled=assemble_disabled):
            try:
                out = build_step_assembly_option_b(
                    base_bytes=a_bytes,
                    part_bytes=b_bytes,
                    tx_mm=float(st.session_state["tx"]),
                    ty_mm=float(st.session_state["ty"]),
                    tz_mm=float(st.session_state["tz"]),
                    rx_deg=float(st.session_state["rx"]),
                    ry_deg=float(st.session_state["ry"]),
                    rz_deg=float(st.session_state["rz"]),
                )
                st.success("STEP assemblé généré ✅")
                st.download_button(
                    "⬇️ Télécharger le STEP assemblé",
                    data=out,
                    file_name="STEP_LAB_ASSEMBLY.stp",
                    mime="application/step",
                )
            except Exception as e:
                st.error(f"Erreur génération STEP : {e}")

    if assemble_disabled:
        st.info("Chargez les deux STEP (Ensemble A + Ensemble B) pour activer la génération.")
