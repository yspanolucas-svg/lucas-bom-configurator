import json
from pathlib import Path

PRESET_PATH = Path("presets/step_cantilever_default.json")


def load_step_preset():
    if PRESET_PATH.exists():
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_step_preset(axis1, axis2, axis3, origin):
    PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "axis1": list(axis1),
        "axis2": list(axis2),
        "axis3": list(axis3),
        "origin": list(origin),
    }
    with open(PRESET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
