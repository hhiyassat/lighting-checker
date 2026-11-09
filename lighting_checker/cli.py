"""Command line: python -m lighting_checker DRAWING.dwg|dxf --config config/project.json -o report.html"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cad
from .calc import evaluate_emergency, evaluate_room
from .config import load_project
from .parser import parse
from .report import write_html


def run(drawing: Path, config: Path, out_html: Path, out_json: Path | None = None, work_dir: Path | None = None):
    proj = load_project(config)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    drawing = Path(drawing)
    if drawing.suffix.lower() == ".dwg":
        drawing = cad.dwg_to_dxf(drawing, work_dir or drawing.parent)
    doc = cad.load_dxf(drawing)
    u = cad.units_to_m(doc, proj.units_to_m)
    dr = parse(doc, proj, u)
    results = [evaluate_room(r, proj, u) for r in dr.rooms]
    em = evaluate_emergency(dr, proj)
    write_html(out_html, dr, results, em, proj, drawing.name)
    summary = {
        "drawing": drawing.name,
        "rules_source": proj.rules.source,
        "rules_provisional_types": list(proj.rules.provisional_types),
        "rooms": [{
            "id": r.room.id, "floor": r.room.floor, "label": r.room.label, "type": r.room.type,
            "source": r.room.source, "area_m2": round(r.area_m2, 2), "luminaires": r.n_lum,
            "flux_lm": round(r.flux_lm), "e_lumen": r.e_lumen and round(r.e_lumen, 1),
            "em": r.e_avg and round(r.e_avg, 1), "emin": r.e_min and round(r.e_min, 1),
            "uo": r.uo and round(r.uo, 3), "ugr": r.ugr and round(r.ugr, 1),
            "spacing_d_m": r.spacing and round(r.spacing[0], 2), "spacing_h_m": r.spacing and round(r.spacing[1], 2),
            "spacing_ratio": r.spacing and round(r.spacing[2], 2),
            "checks": {k: {"value": v, "limit": lim, "ok": okk} for k, (v, lim, okk) in r.checks.items()},
            "required": r.requirement, "requirement_provisional": r.room.type in proj.rules.provisional_types,
            "verdict": r.verdict,
            "deficit_lx": r.deficit_lx and round(r.deficit_lx, 1), "extra_luminaires": r.extra_luminaires,
            "notes": r.notes,
        } for r in results],
        "emergency": {"present": em.present, "routes": em.routes, "anti_panic": em.anti_panic, "notes": em.notes},
        "warnings": dr.warnings,
        "unassigned": [{"block": lm.block, "sheet": lm.sheet, "x": lm.x, "y": lm.y} for lm in dr.unassigned if lm.sheet],
    }
    if out_json:
        Path(out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lighting compliance checker for DWG/DXF plans")
    ap.add_argument("drawing")
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--out", default="lighting_report.html")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    s = run(Path(a.drawing), Path(a.config), Path(a.out), Path(a.json) if a.json else None)
    from collections import Counter
    c = Counter(r["verdict"] for r in s["rooms"])
    print(f"{len(s['rooms'])} rooms: " + ", ".join(f"{k}={v}" for k, v in c.items()) + f" → {a.out}")
    return 0 if c.get("fail", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
