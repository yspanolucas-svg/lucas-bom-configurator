import re
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st


# ============================================================
# Preset
# ============================================================
PRESET_PATH = Path("presets/step_cantilever_default.json")


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
# Vecteurs
# ============================================================
def _norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def _normalize(v):
    n = _norm(v)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def _cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


# ============================================================
# Extraction : nom PRODUCT dans un STEP “simple”
# ============================================================
def extract_first_product_name(step_bytes: bytes) -> Optional[str]:
    _, ents, _ = parse_step(step_bytes)
    for eid in sorted(ents.keys()):
        body = ents[eid].strip()
        if body.upper().startswith("PRODUCT("):
            m = re.match(r"PRODUCT\(\s*'([^']+)'", body, flags=re.I)
            if m:
                return m.group(1)
    return None


# ============================================================
# Apprentissage : extraction du placement depuis référence assemblée
# Version robuste (ne dépend PAS du texte "Placement of ...")
# ============================================================
def extract_transform_from_reference_targeted(
    ref_bytes: bytes,
    base_name: str,
    part_name: str
) -> Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float]]:
    _, ents, _ = parse_step(ref_bytes)

    def _best_name_match(target: str, candidates: List[str]) -> Optional[str]:
        t = target.strip().lower()
        if not t:
            return None
        # exact
        for c in candidates:
            if c.lower() == t:
                return c
        # prefix
        for c in candidates:
            if c.lower().startswith(t) or t.startswith(c.lower()):
                return c
        # contains
        for c in candidates:
            if t in c.lower() or c.lower() in t:
                return c
        return None

    # 1) Index CARTESIAN_POINT / DIRECTION / AXIS2_PLACEMENT_3D
    points = {}
    dirs = {}
    axis2 = {}

    for eid, body in ents.items():
        up = body.upper().strip()

        if up.startswith("CARTESIAN_POINT"):
            m = re.search(r"\(\s*'[^']*'\s*,\s*\(\s*([^\)]+)\s*\)\s*\)", body, flags=re.I)
            if m:
                parts = m.group(1).replace(" ", "").split(",")
                if len(parts) == 3:
                    try:
                        points[eid] = tuple(float(p.replace("E", "e")) for p in parts)
                    except:
                        pass

        elif up.startswith("DIRECTION"):
            m = re.search(r"\(\s*'[^']*'\s*,\s*\(\s*([^\)]+)\s*\)\s*\)", body, flags=re.I)
            if m:
                parts = m.group(1).replace(" ", "").split(",")
                if len(parts) == 3:
                    try:
                        d = tuple(float(p.replace("E", "e")) for p in parts)
                        dirs[eid] = _normalize(d)
                    except:
                        pass

        elif up.startswith("AXIS2_PLACEMENT_3D"):
            m = re.match(
                r"AXIS2_PLACEMENT_3D\(\s*'[^']*'\s*,\s*(#\d+)\s*,\s*([^,]+)\s*,\s*([^\)]+)\s*\)\s*;",
                body.strip(),
                flags=re.I
            )
            if m:
                loc = int(m.group(1)[1:])
                axis_raw = m.group(2).strip()
                ref_raw = m.group(3).strip()
                axis_id = None if axis_raw == "$" else int(axis_raw[1:])
                ref_id = None if ref_raw == "$" else int(ref_raw[1:])
                axis2[eid] = (loc, axis_id, ref_id)

    # 2) Index PRODUCT id -> name
    product_id_by_name = {}
    for eid in sorted(ents.keys()):
        body = ents[eid].strip()
        if body.upper().startswith("PRODUCT("):
            m = re.match(r"PRODUCT\(\s*'([^']*)'", body, flags=re.I)
            if m:
                product_id_by_name[m.group(1)] = eid

    if not product_id_by_name:
        raise ValueError("Référence : aucun PRODUCT() trouvé, impossible d'apprendre le placement.")

    base_pick = _best_name_match(base_name, list(product_id_by_name.keys()))
    part_pick = _best_name_match(part_name, list(product_id_by_name.keys()))

    if not base_pick or not part_pick:
        raise ValueError(
            "Référence : impossible de retrouver les PRODUCT dans l'assemblage.\n"
            f"Base demandé: {base_name}\nPart demandé: {part_name}\n"
            "=> Les noms PRODUCT dans la référence sont probablement différents (tronqués/renommés)."
        )

    base_prod_id = product_id_by_name[base_pick]
    part_prod_id = product_id_by_name[part_pick]

    # 3) PRODUCT_DEFINITION_FORMATION -> PRODUCT
    pdf_to_product = {}
    for eid, body in ents.items():
        if body.upper().startswith("PRODUCT_DEFINITION_FORMATION"):
            m = re.match(
                r"PRODUCT_DEFINITION_FORMATION\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*\)\s*;",
                body, flags=re.I
            )
            if m:
                pdf_to_product[eid] = int(m.group(1)[1:])

    # 4) PRODUCT_DEFINITION -> PRODUCT_DEFINITION_FORMATION
    pd_to_pdf = {}
    for eid, body in ents.items():
        if body.upper().startswith("PRODUCT_DEFINITION("):
            m = re.match(
                r"PRODUCT_DEFINITION\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;",
                body, flags=re.I
            )
            if m:
                pd_to_pdf[eid] = int(m.group(1)[1:])

    def _find_pd_for_product(prod_id: int) -> Optional[int]:
        for pd_id, pdf_id in pd_to_pdf.items():
            p = pdf_to_product.get(pdf_id)
            if p == prod_id:
                return pd_id
        return None

    base_pd = _find_pd_for_product(base_prod_id)
    part_pd = _find_pd_for_product(part_prod_id)

    if base_pd is None or part_pd is None:
        raise ValueError("Référence : PRODUCT_DEFINITION introuvable pour la base ou la pièce.")

    # 5) PRODUCT_DEFINITION_SHAPE -> PRODUCT_DEFINITION (pour la pièce)
    pds_list = []
    for eid, body in ents.items():
        if body.upper().startswith("PRODUCT_DEFINITION_SHAPE"):
            m = re.match(
                r"PRODUCT_DEFINITION_SHAPE\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*\)\s*;",
                body, flags=re.I
            )
            if m and int(m.group(1)[1:]) == part_pd:
                pds_list.append(eid)

    if not pds_list:
        raise ValueError("Référence : aucun PRODUCT_DEFINITION_SHAPE lié à la pièce.")

    # 6) CONTEXT_DEPENDENT_SHAPE_REPRESENTATION( #rel , #pds_part )
    # puis #rel contient REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#idt)
    re_cdsr = re.compile(
        r"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION\(\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;",
        flags=re.I
    )
    re_rrwt = re.compile(
        r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\(\s*(#\d+)\s*\)",
        flags=re.I
    )
    re_idt = re.compile(
        r"ITEM_DEFINED_TRANSFORMATION\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;",
        flags=re.I
    )

    candidates = []
    for eid in sorted(ents.keys()):
        body = ents[eid].strip()
        if not body.upper().startswith("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
            continue
        m = re_cdsr.match(body)
        if not m:
            continue
        rel_id = int(m.group(1)[1:])
        pds_id = int(m.group(2)[1:])
        if pds_id not in pds_list:
            continue

        rel_body = ents.get(rel_id, "")
        m2 = re_rrwt.search(rel_body)
        if not m2:
            continue
        idt_id = int(m2.group(1)[1:])

        idt_body = ents.get(idt_id, "")
        m3 = re_idt.match(idt_body.strip())
        if not m3:
            continue

        ax2_b = int(m3.group(2)[1:])
        if ax2_b not in axis2:
            continue

        loc_id, z_id, x_id = axis2[ax2_b]
        origin = points.get(loc_id, (0.0, 0.0, 0.0))
        d_origin = _norm(origin)

        # On ne garde que les placements non-triviaux
        if d_origin < 1e-6:
            continue

        candidates.append((d_origin, ax2_b, origin, z_id, x_id, eid, rel_id, idt_id))

    if not candidates:
        raise ValueError(
            "Impossible d’extraire une transformation non-identité : "
            "aucun placement CONTEXT_DEPENDENT_SHAPE_REPRESENTATION exploitable trouvé pour la pièce.\n"
            "Astuce : si la référence a été exportée 'aplatie' (pas d'assemblage), il n'y aura pas de placements."
        )

    # 7) On choisit le placement avec la translation la plus significative
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, ax2_b, origin, z_id, x_id, _, _, _ = candidates[0]

    z = (0.0, 0.0, 1.0) if (z_id is None or z_id not in dirs) else dirs[z_id]
    x = (1.0, 0.0, 0.0) if (x_id is None or x_id not in dirs) else dirs[x_id]

    x = _normalize(x)
    z = _normalize(z)
    y = _normalize(_cross(z, x))
    x = _normalize(_cross(y, z))

    return x, y, z, origin


# ============================================================
# Assemblage Option B (réécriture STEP)
# ============================================================
def build_step_assembly_option_b(
    base_bytes: bytes,
    part_bytes: bytes,
    axis1: Tuple[float,float,float],
    axis2: Tuple[float,float,float],
    axis3: Tuple[float,float,float],
    origin: Tuple[float,float,float],
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
        raise ValueError("GEOMETRIC_REPRESENTATION_CONTEXT introuvable dans le STEP base.")

    base_shape_id = find_first_shape_representation_id(base_ents)
    if base_shape_id is None:
        raise ValueError("SHAPE_REPRESENTATION introuvable dans le STEP base.")

    part_shape_id = find_first_shape_representation_id(part_renum)
    if part_shape_id is None:
        raise ValueError("SHAPE_REPRESENTATION introuvable dans le STEP ajouté.")

    nid = max_entity_id(merged) + 1

    def add(entity: str) -> int:
        nonlocal nid
        e = entity.strip()
        if not e.endswith(";"):
            e += ";"
        merged[nid] = e
        nid += 1
        return nid - 1

    # Base mapping (identité)
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

    # Part mapping (preset)
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

    new_shape_rep = add(f"SHAPE_REPRESENTATION('',(#{b_mapped},#{p_mapped}),#{ctx_id})")
    ok = replace_shape_definition_representation_target(merged, new_shape_rep)
    if not ok:
        pd_id = find_first_id_containing(merged, ["PRODUCT_DEFINITION"])
        if pd_id is not None:
            add(f"SHAPE_DEFINITION_REPRESENTATION(#{pd_id},#{new_shape_rep})")

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
# DEBUG HELPERS
# ============================================================
def _debug_list_products(step_bytes: bytes) -> list[tuple[int, str]]:
    _, ents, _ = parse_step(step_bytes)
    out = []
    for eid in sorted(ents.keys()):
        body = ents[eid].strip()
        if body.upper().startswith("PRODUCT("):
            m = re.match(r"PRODUCT\(\s*'([^']*)'", body, flags=re.I)
            if m:
                out.append((eid, m.group(1)))
    return out


def _debug_find_candidate_transforms(ref_bytes: bytes, part_name: str) -> list[dict]:
    """
    Debug : liste les placements trouvés pour la pièce dans la référence,
    avec origins + ids STEP.
    """
    _, ents, _ = parse_step(ref_bytes)

    # Index points / dirs / axis2
    points, dirs, axis2 = {}, {}, {}

    for eid, body in ents.items():
        up = body.upper().strip()

        if up.startswith("CARTESIAN_POINT"):
            m = re.search(r"\(\s*'[^']*'\s*,\s*\(\s*([^\)]+)\s*\)\s*\)", body, flags=re.I)
            if m:
                parts = m.group(1).replace(" ", "").split(",")
                if len(parts) == 3:
                    try:
                        points[eid] = tuple(float(p.replace("E", "e")) for p in parts)
                    except:
                        pass

        elif up.startswith("DIRECTION"):
            m = re.search(r"\(\s*'[^']*'\s*,\s*\(\s*([^\)]+)\s*\)\s*\)", body, flags=re.I)
            if m:
                parts = m.group(1).replace(" ", "").split(",")
                if len(parts) == 3:
                    try:
                        d = tuple(float(p.replace("E", "e")) for p in parts)
                        dirs[eid] = _normalize(d)
                    except:
                        pass

        elif up.startswith("AXIS2_PLACEMENT_3D"):
            m = re.match(
                r"AXIS2_PLACEMENT_3D\(\s*'[^']*'\s*,\s*(#\d+)\s*,\s*([^,]+)\s*,\s*([^\)]+)\s*\)\s*;",
                body.strip(),
                flags=re.I
            )
            if m:
                loc = int(m.group(1)[1:])
                axis_raw = m.group(2).strip()
                ref_raw = m.group(3).strip()
                axis_id = None if axis_raw == "$" else int(axis_raw[1:])
                ref_id = None if ref_raw == "$" else int(ref_raw[1:])
                axis2[eid] = (loc, axis_id, ref_id)

    # products in ref
    products = _debug_list_products(ref_bytes)
    names = [n for _, n in products]

    # best-effort match
    t = (part_name or "").strip().lower()
    part_pick = None
    for n in names:
        if n.lower() == t:
            part_pick = n
            break
    if not part_pick:
        for n in names:
            if n.lower().startswith(t) or t.startswith(n.lower()):
                part_pick = n
                break
    if not part_pick:
        for n in names:
            if t in n.lower() or n.lower() in t:
                part_pick = n
                break

    if not part_pick:
        return [{"error": f"Part '{part_name}' non retrouvée dans les PRODUCT() de la référence."}]

    # find product id
    part_prod_id = None
    for eid in sorted(ents.keys()):
        b = ents[eid].strip()
        if b.upper().startswith("PRODUCT("):
            m = re.match(r"PRODUCT\(\s*'([^']*)'", b, flags=re.I)
            if m and m.group(1) == part_pick:
                part_prod_id = eid
                break

    if part_prod_id is None:
        return [{"error": "Part PRODUCT id introuvable."}]

    # PDF -> PRODUCT, PD -> PDF
    pdf_to_product = {}
    for eid, body in ents.items():
        if body.upper().startswith("PRODUCT_DEFINITION_FORMATION"):
            m = re.match(r"PRODUCT_DEFINITION_FORMATION\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*\)\s*;", body, flags=re.I)
            if m:
                pdf_to_product[eid] = int(m.group(1)[1:])

    pd_to_pdf = {}
    for eid, body in ents.items():
        if body.upper().startswith("PRODUCT_DEFINITION("):
            m = re.match(r"PRODUCT_DEFINITION\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;", body, flags=re.I)
            if m:
                pd_to_pdf[eid] = int(m.group(1)[1:])

    part_pd = None
    for pd_id, pdf_id in pd_to_pdf.items():
        if pdf_to_product.get(pdf_id) == part_prod_id:
            part_pd = pd_id
            break
    if part_pd is None:
        return [{"error": "PRODUCT_DEFINITION pour la pièce introuvable."}]

    # PDS list
    pds_list = []
    for eid, body in ents.items():
        if body.upper().startswith("PRODUCT_DEFINITION_SHAPE"):
            m = re.match(r"PRODUCT_DEFINITION_SHAPE\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*\)\s*;", body, flags=re.I)
            if m and int(m.group(1)[1:]) == part_pd:
                pds_list.append(eid)

    if not pds_list:
        return [{"error": "Aucun PRODUCT_DEFINITION_SHAPE pour la pièce."}]

    # Cdsr candidates
    re_cdsr = re.compile(
        r"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION\(\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;",
        flags=re.I
    )
    re_rrwt = re.compile(
        r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\(\s*(#\d+)\s*\)",
        flags=re.I
    )
    re_idt = re.compile(
        r"ITEM_DEFINED_TRANSFORMATION\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*(#\d+)\s*,\s*(#\d+)\s*\)\s*;",
        flags=re.I
    )

    candidates = []
    for eid in sorted(ents.keys()):
        body = ents[eid].strip()
        if not body.upper().startswith("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
            continue
        m = re_cdsr.match(body)
        if not m:
            continue
        rel_id = int(m.group(1)[1:])
        pds_id = int(m.group(2)[1:])
        if pds_id not in pds_list:
            continue

        rel_body = ents.get(rel_id, "")
        m2 = re_rrwt.search(rel_body)
        if not m2:
            continue
        idt_id = int(m2.group(1)[1:])
        idt_body = ents.get(idt_id, "")
        m3 = re_idt.match(idt_body.strip())
        if not m3:
            continue

        ax2_b = int(m3.group(2)[1:])
        if ax2_b not in axis2:
            continue

        loc_id, z_id, x_id = axis2[ax2_b]
        origin = points.get(loc_id, (0.0, 0.0, 0.0))
        d_origin = _norm(origin)

        candidates.append({
            "cdsr_entity_id": eid,
            "relationship_entity_id": rel_id,
            "idt_entity_id": idt_id,
            "axis2_placement_3d_id": ax2_b,
            "origin": origin,
            "origin_norm": d_origin,
            "z_dir_id": z_id,
            "x_dir_id": x_id,
        })

    candidates.sort(key=lambda x: x["origin_norm"], reverse=True)
    return candidates


# ============================================================
# STREAMLIT UI
# ============================================================
def render_step_lab_panel():
    st.subheader("🧊 STEP Lab – Assemblage cantilever (sans CAO)")

    st.markdown(
        """
**But :** assembler deux STEP (Ensemble A + Ensemble B) via un preset.

**Apprentissage :**
- On fournit Ensemble A + Ensemble B + un STEP assemblé de référence.
- On extrait le placement de **B** dans la référence via la structure d’assemblage STEP.
- On sauvegarde `presets/step_cantilever_default.json`.
        """
    )

    preset = load_step_preset()
    if preset:
        st.success("Preset STEP trouvé ✅")
    else:
        st.warning("Aucun preset STEP. Faites l’apprentissage avec une référence assemblée.")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1) Fichiers à assembler")
        step_a_up = st.file_uploader("STEP – Ensemble A (base)", type=["stp", "step"], key="step_a")
        step_b_up = st.file_uploader("STEP – Ensemble B (à placer)", type=["stp", "step"], key="step_b")

    with c2:
        st.markdown("### 2) Apprentissage (1 fois)")
        ref_up = st.file_uploader("STEP référence assemblé (cantilever correct)", type=["stp", "step"], key="step_ref")
        st.caption("Le fichier référence n’est pas stocké. On conserve uniquement la règle (axes + origin).")

    step_a_bytes = step_a_up.read() if step_a_up else None
    step_b_bytes = step_b_up.read() if step_b_up else None
    ref_bytes = ref_up.read() if ref_up else None

    # ---------------- DEBUG ----------------
    st.markdown("---")
    st.markdown("### 🐞 Debug (diagnostic)")
    debug_on = st.checkbox("Activer le mode debug", value=False)

    if debug_on:
        colA, colB, colR = st.columns(3)

        with colA:
            st.write("**PRODUCT() – Ensemble A**")
            if step_a_bytes:
                prods = _debug_list_products(step_a_bytes)
                st.write(f"{len(prods)} product(s)")
                st.dataframe([{"#id": i, "name": n} for i, n in prods], use_container_width=True)
            else:
                st.info("Chargez Ensemble A")

        with colB:
            st.write("**PRODUCT() – Ensemble B**")
            if step_b_bytes:
                prods = _debug_list_products(step_b_bytes)
                st.write(f"{len(prods)} product(s)")
                st.dataframe([{"#id": i, "name": n} for i, n in prods], use_container_width=True)
            else:
                st.info("Chargez Ensemble B")

        with colR:
            st.write("**PRODUCT() – Référence assemblée**")
            if ref_bytes:
                prods = _debug_list_products(ref_bytes)
                st.write(f"{len(prods)} product(s)")
                st.dataframe([{"#id": i, "name": n} for i, n in prods[:30]], use_container_width=True)
                if len(prods) > 30:
                    st.caption("Affichage limité aux 30 premiers.")
            else:
                st.info("Chargez la référence assemblée")

        if ref_bytes and step_b_bytes:
            st.markdown("#### Candidats placement trouvés pour Ensemble B dans la référence")
            part_name_guess = extract_first_product_name(step_b_bytes) or ""
            st.caption(f"Nom PRODUCT détecté dans Ensemble B : {part_name_guess!r}")

            cands = _debug_find_candidate_transforms(ref_bytes, part_name_guess)
            if cands and "error" in cands[0]:
                st.error(cands[0]["error"])
            else:
                st.write(f"{len(cands)} candidat(s) trouvé(s) (triés par translation |origin| décroissante).")
                st.dataframe(
                    [{
                        "origin_norm": round(c["origin_norm"], 6),
                        "origin": c["origin"],
                        "cdsr": f"#{c['cdsr_entity_id']}",
                        "rel": f"#{c['relationship_entity_id']}",
                        "idt": f"#{c['idt_entity_id']}",
                        "axis2": f"#{c['axis2_placement_3d_id']}",
                    } for c in cands[:50]],
                    use_container_width=True
                )
                if len(cands) > 50:
                    st.caption("Affichage limité aux 50 premiers.")

                st.info(
                    "Astuce : si origin_norm est très faible partout, la référence est peut-être 'aplatie' "
                    "(pas de vraie structure d'assemblage)."
                )

    # ---------------- Learning ----------------
    st.markdown("---")
    st.markdown("### Apprendre / mettre à jour le preset")

    learn_disabled = (ref_bytes is None) or (step_a_bytes is None) or (step_b_bytes is None)

    if st.button("📚 Apprendre depuis la référence et enregistrer le preset", disabled=learn_disabled):
        try:
            base_name = extract_first_product_name(step_a_bytes) or ""
            part_name = extract_first_product_name(step_b_bytes) or ""
            if not base_name or not part_name:
                raise ValueError("Impossible de lire le nom PRODUCT dans Ensemble A ou Ensemble B.")

            a1, a2, a3, org = extract_transform_from_reference_targeted(ref_bytes, base_name, part_name)
            save_step_preset(a1, a2, a3, org)

            st.success("Preset cantilever enregistré ✅")
            st.write(f"Base PRODUCT: `{base_name}`")
            st.write(f"Part  PRODUCT: `{part_name}`")
            st.json({"axis1": a1, "axis2": a2, "axis3": a3, "origin": org})
            st.info("Vous pouvez maintenant assembler sans recharger la référence.")
        except Exception as e:
            st.error(f"Erreur apprentissage : {e}")

    # ---------------- Preset view ----------------
    st.markdown("---")
    preset = load_step_preset()
    with st.expander("Voir le preset actuel", expanded=False):
        st.json(preset if preset else {})

    # ---------------- Assemble ----------------
    st.markdown("### Génération du STEP assemblé")
    if not step_a_bytes or not step_b_bytes:
        st.info("Chargez Ensemble A et Ensemble B pour activer l’assemblage.")
        return

    if st.button("⚙️ Assembler (avec le preset)", type="primary"):
        try:
            preset = load_step_preset()
            if not preset:
                raise ValueError("Aucun preset trouvé. Faites l’apprentissage une fois avec la référence assemblée.")

            out = build_step_assembly_option_b(
                base_bytes=step_a_bytes,
                part_bytes=step_b_bytes,
                axis1=tuple(preset["axis1"]),
                axis2=tuple(preset["axis2"]),
                axis3=tuple(preset["axis3"]),
                origin=tuple(preset["origin"]),
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
