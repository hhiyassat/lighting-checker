"""DWG -> DXF conversion and DXF loading.

Converter order:
  1. ODA File Converter (free, closed source; ezdxf.addons.odafc) — recommended for production.
  2. LibreDWG `dwg2dxf` (GPL) — fine for prototyping; do not link/ship it in a closed product.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import ezdxf
from ezdxf import recover

# $INSUNITS code -> metres
INSUNITS_TO_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}


def dwg_to_dxf(dwg: Path, out_dir: Path) -> Path:
    dwg, out_dir = Path(dwg), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dxf = out_dir / (dwg.stem + ".dxf")
    if dxf.exists() and dxf.stat().st_mtime >= dwg.stat().st_mtime:
        return dxf
    try:
        from ezdxf.addons import odafc
        if odafc.is_installed():
            odafc.convert(str(dwg), str(dxf), version="R2018", replace=True)
            return dxf
    except Exception:  # noqa: BLE001 — fall through to LibreDWG
        pass
    exe = shutil.which("dwg2dxf")
    if exe:
        subprocess.run([exe, "-y", "-o", str(dxf), str(dwg)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dxf
    raise RuntimeError(
        "No DWG converter found. Install ODA File Converter, or `brew install libredwg` "
        "(prototype only), or export the drawing to DXF from AutoCAD (SAVEAS → DXF).")


def load_dxf(path: Path):
    path = Path(path)
    try:
        return ezdxf.readfile(path)
    except ezdxf.DXFStructureError:
        doc, _auditor = recover.readfile(path)
        return doc


def units_to_m(doc, override: float | None = None) -> float:
    if override:
        return float(override)
    code = doc.header.get("$INSUNITS", 0)
    if code not in INSUNITS_TO_M:
        raise ValueError(f"Drawing units $INSUNITS={code} not supported; set units_to_m in the project config.")
    return INSUNITS_TO_M[code]
