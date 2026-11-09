"""Unit and end-to-end tests. Analytical checks follow the spirit of CIE 171:2006 (point source cases)."""
import json
import math
from pathlib import Path

import ezdxf
import numpy as np
import pytest

from lighting_checker import photometry
from lighting_checker.calc import direct_illuminance, evaluate_room, grid_spacing, guth_position_index
from lighting_checker.cli import run
from lighting_checker.config import Rules, load_project
from lighting_checker.parser import Luminaire, parse


# ----------------------------------------------------------------------------- photometry
def test_generic_downlight_flux_and_peak():
    ph = photometry.generic("downlight", 1000)
    assert ph.total_flux() == pytest.approx(1000, rel=2e-3)
    assert float(ph.intensity(0, 0)) == pytest.approx(1000 / math.pi)
    assert float(ph.intensity(123, 95)) == 0.0


def _write_ies(path: Path):
    g = list(range(0, 91, 5))
    cd = [round(500 * math.cos(math.radians(a)), 3) for a in g]
    lines = ["IESNA:LM-63-2002", "[TEST] synthetic", "[LUMINAIRE] Test downlight", "TILT=NONE",
             f"1 -1 1 {len(g)} 1 1 2 0.2 0.2 0", "1 1 20",
             " ".join(map(str, g)), "0", " ".join(map(str, cd))]
    path.write_text("\n".join(lines))


def test_read_ies_absolute(tmp_path):
    p = tmp_path / "t.ies"
    _write_ies(p)
    ph = photometry.read_ies(p)
    assert float(ph.intensity(0, 0)) == pytest.approx(500)
    assert float(ph.intensity(200, 60)) == pytest.approx(250, rel=1e-3)
    assert ph.flux_lm == pytest.approx(500 * math.pi, rel=5e-3)     # ∫ I0 cosγ dΩ over hemisphere = π I0
    assert ph.width_m == pytest.approx(0.2)


def _write_ldt(path: Path, isym=1):
    g = list(range(0, 181, 10))
    vals = [round(318.31 * max(math.cos(math.radians(a)), 0), 2) for a in g]   # cd/klm, Lambertian
    mc = 1 if isym == 1 else 4
    c = [0] if isym == 1 else [0, 90, 180, 270]
    planes = 1 if isym == 1 else 4
    L = ["ACME", "1", str(isym), str(mc), "0" if isym == 1 else "90", str(len(g)), "10", "R1", "Test LDT",
         "T-1", "t.ldt", "2026", "300", "300", "50", "300", "300", "0", "0", "0", "0", "100", "100", "1", "0",
         "1", "1", "LED", "2000", "4000", "80", "20"] + ["0.5"] * 10 + [str(x) for x in c] + [str(x) for x in g]
    for _ in range(planes):
        L += [str(v) for v in vals]
    path.write_text("\n".join(L))


@pytest.mark.parametrize("isym", [1, 0])
def test_read_ldt(tmp_path, isym):
    p = tmp_path / "t.ldt"
    _write_ldt(p, isym)
    ph = photometry.read_ldt(p)
    assert ph.flux_lm == 2000
    assert float(ph.intensity(45, 0)) == pytest.approx(318.31 * 2, rel=1e-3)
    assert ph.total_flux() == pytest.approx(2000, rel=2e-2)   # 10° steps: trapezoid error ~1 %


def test_c_plane_symmetry_quadrant():
    c = np.array([0, 45, 90.0])
    cd = np.array([[1, 1], [2, 2], [3, 3.0]])
    cc, out = photometry._expand_c(c, cd)
    ph = photometry.Photometry(cc, np.array([0, 180.0]), out, 1)
    assert float(ph.intensity(135, 0)) == pytest.approx(2)     # mirrors 45
    assert float(ph.intensity(270, 0)) == pytest.approx(3)     # mirrors 90
    assert float(ph.intensity(315, 0)) == pytest.approx(2)


# ----------------------------------------------------------------------------- calculation
def test_point_source_inverse_square_cosine():
    """Lambertian downlight at height h: E(γ) = I0·cos⁴γ / h² (CIE 171 style analytical case)."""
    lm = Luminaire("X", "0", 0.0, 0.0, 0.0, {"lumens": 1000, "distribution": "downlight", "mounting": "ceiling"})
    h = 2.0
    xs = np.array([0.0, 2.0, 4.0])
    E = direct_illuminance(xs, np.zeros(3), np.zeros(3), [lm], 1.0, h, lambda p: p)
    I0 = 1000 / math.pi
    for x, e in zip(xs, E):
        cg = h / math.hypot(x, h)
        assert e == pytest.approx(I0 * cg ** 4 / h ** 2, rel=1e-3)


def test_grid_rule():
    assert grid_spacing(1) == pytest.approx(0.2)
    assert grid_spacing(10) == pytest.approx(1.0)
    assert grid_spacing(100) == pytest.approx(5.0)


def test_guth_on_axis_is_one():
    assert guth_position_index(0, 0) == pytest.approx(1.0)
    assert guth_position_index(30, 40) > 1.0


def test_classify_aliases():
    r = Rules({}, {"living": ["living room", "salon"], "bedroom": ["bedroom", "master bedroom"],
                   "kitchen": ["kitchen", "مطبخ"]}, {})
    assert r.classify("Master Bedroom") == "bedroom"
    assert r.classify("  SALON ") == "living"
    assert r.classify("مطبخ") == "kitchen"
    assert r.classify("Lift lobby") == "unknown"


# ----------------------------------------------------------------------------- end-to-end
def _project(tmp_path, rules_types=None):
    cat = {"DL": {"lumens": 2000, "watts": 18, "mounting": "ceiling", "distribution": "downlight",
                  "luminous_area_m2": 0.03, "ra": 80}}
    rules = {"source": "test", "types": rules_types or {"office": {"em_lx": 300, "uo": 0.4, "ugr_max": None, "ra": 80}},
             "aliases": {"office": ["office"]}, "emergency": {"emergency_block_regex": "^EM$"}}
    proj = {"sheets": {}, "layers": {"room_boundary": ["A-ROOM"], "room_tag": ["A-ROOM-TAG"], "walls": ["A-WALL"],
                                      "egress": ["E-EGRESS"]},
            "room_detection": {"pixel_m": 0.05, "seal_gap_m": 0.1, "max_room_area_m2": 150, "min_room_area_m2": 1},
            "geometry_defaults": {"ceiling_height_m": 3.0, "work_plane_m": 0.8, "wall_band_m": 0.5,
                                  "ceiling_height_regex": "\\bH\\s*=\\s*(?P<h>\\d+(?:\\.\\d+)?)"},
            "surfaces": {"rho_ceiling": 0.7, "rho_walls": 0.5, "rho_floor": 0.2}, "maintenance_factor": 0.8,
            "luminaire_catalog": "cat.json", "rules": "rules.json"}
    (tmp_path / "cat.json").write_text(json.dumps(cat))
    (tmp_path / "rules.json").write_text(json.dumps(rules))
    (tmp_path / "proj.json").write_text(json.dumps(proj))
    return tmp_path / "proj.json"


def _drawing(tmp_path, n_lum=4, boundary=True):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 6
    b = doc.blocks.new("DL")
    b.add_circle((0, 0), 0.15)
    msp = doc.modelspace()
    W, H = 6.0, 5.0
    if boundary:
        msp.add_lwpolyline([(0, 0), (W, 0), (W, H), (0, H)], close=True, dxfattribs={"layer": "A-ROOM"})
    else:   # walls with a 0.9 m door opening + door swing (leaf + arc)
        for a, b2 in [((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 2.9)), ((0, 2.0), (0, 0))]:
            msp.add_line(a, b2, dxfattribs={"layer": "A-WALL"})
        msp.add_line((0, 2.0), (0.9, 2.0), dxfattribs={"layer": "A-WALL"})
        msp.add_arc((0, 2.0), 0.9, 0, 90, dxfattribs={"layer": "A-WALL"})
        msp.add_line((-3, -1), (-3, 7), dxfattribs={"layer": "A-WALL"})  # something outside
    msp.add_text("OFFICE", dxfattribs={"layer": "A-ROOM-TAG", "height": 0.25}).set_placement((2.5, 2.5))
    msp.add_text("H=2.8", dxfattribs={"layer": "A-ROOM-TAG", "height": 0.2}).set_placement((2.5, 1.5))
    pts = [(1.5, 1.25), (4.5, 1.25), (1.5, 3.75), (4.5, 3.75)][:n_lum]
    for p in pts:
        msp.add_blockref("DL", p, dxfattribs={"layer": "E-LITE"})
    path = tmp_path / ("room.dxf" if boundary else "walls.dxf")
    doc.saveas(path)
    return path


def test_end_to_end_polyline_room(tmp_path):
    cfg = _project(tmp_path)
    dxf = _drawing(tmp_path)
    s = run(dxf, cfg, tmp_path / "r.html", tmp_path / "r.json")
    assert len(s["rooms"]) == 1
    r = s["rooms"][0]
    assert r["type"] == "office" and r["luminaires"] == 4
    assert r["area_m2"] == pytest.approx(30, rel=0.02)
    # lumen method by hand: K = 2A/(hm·P) = 60/(2.0·22) = 1.36 → UF≈0.55 ; E = 8000·0.55·0.8/30 ≈ 117
    assert r["e_lumen"] == pytest.approx(117, rel=0.05)
    assert r["verdict"] == "fail" and r["deficit_lx"] > 0
    assert (tmp_path / "r.html").stat().st_size > 10_000


def test_end_to_end_ceiling_height_from_text(tmp_path):
    cfg = _project(tmp_path)
    proj = load_project(cfg)
    doc = ezdxf.readfile(_drawing(tmp_path))
    dr = parse(doc, proj, 1.0)
    assert dr.rooms[0].ceiling_h_m == pytest.approx(2.8)


def test_walls_mode_door_sealed(tmp_path):
    cfg = _project(tmp_path)
    proj = load_project(cfg)
    doc = ezdxf.readfile(_drawing(tmp_path, boundary=False))
    dr = parse(doc, proj, 1.0)
    rooms = [r for r in dr.rooms if r.label == "OFFICE"]
    assert len(rooms) == 1 and rooms[0].status == "ok"
    a = rooms[0].area_du2
    assert 27 < a < 30.5            # 30 m² minus the door-swing quarter circle (~0.64 m²) and line pixels
    assert len(rooms[0].luminaires) == 4
    res = evaluate_room(rooms[0], proj, 1.0)
    assert res.e_avg > 0 and res.uo > 0


def test_rules_fallback_marks_provisional(tmp_path):
    from lighting_checker.config import load_rules
    (tmp_path / "fb.json").write_text(json.dumps({
        "source": "examples", "provisional": True,
        "types": {"bedroom": {"em_lx": 100, "uo": 0.3}, "kitchen": {"em_lx": 300, "uo": 0.4}},
        "aliases": {"bedroom": ["bedroom"], "kitchen": ["kitchen"]}}))
    (tmp_path / "jo.json").write_text(json.dumps({
        "source": "JO code", "fallback": "fb.json",
        "types": {"bedroom": {"em_lx": 150, "uo": 0.4, "clause": "T1"}, "kitchen": {"em_lx": None}}}))
    r = load_rules(tmp_path / "jo.json")
    assert r.requirement("bedroom")["em_lx"] == 150 and r.requirement("bedroom")["_source"] == "JO code"
    assert r.requirement("kitchen")["em_lx"] == 300
    assert r.provisional_types == ("kitchen",)
    assert r.classify("Kitchen") == "kitchen"


def test_spacing_ratio_nearest_neighbour():
    from lighting_checker.calc import spacing_ratio
    spec = {"lumens": 1000, "mounting": "ceiling"}
    lums = [Luminaire("X", "0", x, y, 0.0, spec) for x, y in [(0, 0), (3, 0), (0, 2), (3, 2)]]
    d, h, r = spacing_ratio(lums, 1.0, 3.0, 0.8)
    assert d == pytest.approx(2.0) and h == pytest.approx(2.2) and r == pytest.approx(2.0 / 2.2)
    wall = Luminaire("W", "0", 9, 9, 0.0, {"lumens": 1000, "mounting": "wall"})
    assert spacing_ratio([lums[0], wall], 1.0, 3.0, 0.8) is None      # wall-mounted units are not counted


def test_spacing_check_in_verdict(tmp_path):
    cfg = _project(tmp_path, {"office": {"em_lx": 50, "uo": None, "ugr_max": None, "ra": None}})
    rules = json.loads((tmp_path / "rules.json").read_text())
    rules["spacing"] = {"max_ratio": 1.0, "clause": "test"}
    (tmp_path / "rules.json").write_text(json.dumps(rules))
    s = run(_drawing(tmp_path), cfg, tmp_path / "r.html", tmp_path / "r.json")
    r = s["rooms"][0]
    assert r["spacing_ratio"] == pytest.approx(2.5 / 2.0, rel=1e-3)     # nn 2.5 m, H=2.8 → h=2.0
    assert r["checks"]["d/h"]["ok"] is False and r["verdict"] == "fail"


# ----------------------------------------------------------------------------- code PDF
import shutil  # noqa: E402

CODE_PDF = Path(__file__).resolve().parent.parent / "codes" / "egyptian-code.pdf"
needs_pdf = pytest.mark.skipif(not (CODE_PDF.exists() and shutil.which("pdftotext")),
                               reason="codes/egyptian-code.pdf or pdftotext missing")


def test_parse_m1_headings_and_lookup():
    from lighting_checker.codebook import lookup, parse_m1_group
    text = "\n".join(["شدة الإضاءة (لوكس)   المكان", "120  سلالم", "  غرفة معيشة :", "150  عام",
                      "120  غرفة نوم   المبانى السكنية", "  مطبخ :", "120  عام", "500 أسطح العمل"])
    rows = parse_m1_group(text, "المبانى السكنية")
    assert lookup(rows, "سلالم") == 120 and lookup(rows, "غرفة نوم") == 120
    assert lookup(rows, "غرفة معيشة/عام") == 150 and lookup(rows, "مطبخ/عام") == 120
    with pytest.raises(ValueError):
        lookup(rows, "عام")            # ambiguous without its heading


@needs_pdf
def test_egypt_308_m1_from_pdf():
    from lighting_checker.codebook import egypt_308_m1, lookup
    page, rows = egypt_308_m1(str(CODE_PDF))
    assert page == 343
    expect = {"غرفة معيشة/عام": 150, "غرفة طعام": 120, "غرفة نوم": 120, "مطبخ/عام": 120,
              "حمام": 300, "ممرات": 60, "سلالم": 120, "حجرة مكتب/عام": 300}
    assert {k: lookup(rows, k) for k in expect} == expect


@needs_pdf
def test_rules_value_must_match_pdf(tmp_path):
    from lighting_checker.config import load_rules
    rules = {"source": "t", "code_pdf": {"path": str(CODE_PDF), "table": "egypt_308_m1", "page": 343},
             "types": {"bedroom": {"em_lx": 150, "pdf_row": "غرفة نوم"}}}
    (tmp_path / "r.json").write_text(json.dumps(rules, ensure_ascii=False))
    with pytest.raises(ValueError, match="differs from the code PDF"):
        load_rules(tmp_path / "r.json")
    rules["types"]["bedroom"]["em_lx"] = None
    (tmp_path / "r.json").write_text(json.dumps(rules, ensure_ascii=False))
    assert load_rules(tmp_path / "r.json").requirement("bedroom")["em_lx"] == 120
