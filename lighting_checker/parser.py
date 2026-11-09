"""Extract sheets, rooms and luminaires from a DXF drawing.

Rooms come from one of three sources, per sheet:
  * 'polyline' — closed polylines on the room-boundary layer(s) (the preferred CAD standard);
  * 'walls'    — no boundaries drawn: walls are rasterised and the region around each
                 room-name text is flood-filled (door swings seal the openings);
  * 'seed'     — unlabelled rooms (e.g. bathrooms) found by flood-filling from a luminaire.
All geometry inside this module is in drawing units; `Room.to_m` converts.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from ezdxf import bbox, disassemble
from ezdxf.math import Vec3
from matplotlib.path import Path as MplPath
from scipy import ndimage

from .config import Project


@dataclass
class Sheet:
    name: str
    floor: str
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class Luminaire:
    block: str
    layer: str
    x: float
    y: float
    rotation: float
    spec: dict
    emergency: bool = False
    sheet: str | None = None
    room_id: str | None = None


@dataclass
class Room:
    id: str
    sheet: str
    floor: str
    label: str | None
    type: str
    source: str
    # raster description (drawing units): mask over a window starting at (ox, oy)
    mask: np.ndarray
    ox: float
    oy: float
    px: float
    ceiling_h_m: float
    status: str = "ok"
    notes: list = field(default_factory=list)
    luminaires: list = field(default_factory=list)

    @property
    def area_du2(self) -> float:
        return float(self.mask.sum()) * self.px * self.px

    def perimeter_du(self) -> float:
        """Boundary length from pixel edges (exact for walls parallel to the axes,
        overestimates diagonal walls by up to √2)."""
        m = np.pad(self.mask, 1)
        edges = np.count_nonzero(m[1:, :] != m[:-1, :]) + np.count_nonzero(m[:, 1:] != m[:, :-1])
        return float(edges) * self.px

    def pixel_centres(self):
        iy, ix = np.nonzero(self.mask)
        return self.ox + (ix + 0.5) * self.px, self.oy + (iy + 0.5) * self.px

    def contains(self, x: float, y: float, tol_px: int = 0) -> bool:
        j = int((x - self.ox) / self.px)
        i = int((y - self.oy) / self.px)
        h, w = self.mask.shape
        for di in range(-tol_px, tol_px + 1):
            for dj in range(-tol_px, tol_px + 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < h and 0 <= jj < w and self.mask[ii, jj]:
                    return True
        return False


@dataclass
class Drawing:
    sheets: list
    rooms: list
    luminaires: list
    egress: list           # list of [(x, y), ...] polylines, drawing units
    u: float               # drawing unit -> metre
    warnings: list = field(default_factory=list)
    unassigned: list = field(default_factory=list)


# ----------------------------------------------------------------------------- helpers
def _text_of(e) -> str:
    if e.dxftype() == "MTEXT":
        return e.plain_text().strip()
    if e.dxftype() in ("TEXT", "ATTRIB"):
        return (e.dxf.text or "").strip()
    return ""


def _centre(e):
    try:
        b = bbox.extents([e], fast=True)
        if b.has_data:
            return b.center.x, b.center.y
    except Exception:  # noqa: BLE001
        pass
    p = e.dxf.insert
    return p.x, p.y


def find_sheets(doc, proj: Project) -> list[Sheet]:
    msp = doc.modelspace()
    cfg = proj.sheets
    frame = cfg.get("frame_block")
    title_re = re.compile(cfg["title_regex"], re.I) if cfg.get("title_regex") else None
    sheets: list[Sheet] = []
    if frame and frame in doc.blocks:
        fb = bbox.extents(doc.blocks[frame])
        for ins in msp.query(f'INSERT[name=="{frame}"]'):
            sx, sy, p = abs(ins.dxf.xscale), abs(ins.dxf.yscale), ins.dxf.insert
            sheets.append(Sheet("", "", p.x + fb.extmin.x * sx, p.y + fb.extmin.y * sy,
                                p.x + fb.extmax.x * sx, p.y + fb.extmax.y * sy))
    if not sheets:
        b = bbox.extents(msp, fast=True)
        sheets = [Sheet("Model", "Model", b.extmin.x, b.extmin.y, b.extmax.x, b.extmax.y)]
        return sheets
    title_layers = set(cfg.get("title_layers") or [])
    for t in msp.query("TEXT MTEXT"):
        if title_layers and t.dxf.layer not in title_layers:
            continue
        s = _text_of(t)
        m = title_re.match(s) if title_re else None
        if not m:
            continue
        x, y = t.dxf.insert.x, t.dxf.insert.y
        for sh in sheets:
            if sh.contains(x, y) and not sh.name:
                sh.name, sh.floor = s, m.groupdict().get("floor", s).strip()
    if title_re:
        sheets = [s for s in sheets if s.name]
    for i, s in enumerate(sheets):
        s.name = s.name or f"Sheet {i + 1}"
        s.floor = s.floor or s.name
    return sheets


def _primitive_polylines(entities, flatten: float):
    """Yield flattened vertex lists (x, y) for every drawable primitive (blocks exploded)."""
    for prim in disassemble.to_primitives(disassemble.recursive_decompose(entities)):
        if prim.is_empty:
            continue
        path = prim.path
        if path is not None:
            for sub in path.sub_paths():
                pts = [(v.x, v.y) for v in sub.flattening(flatten)]
                if len(pts) >= 2:
                    yield pts
        else:
            mesh = prim.mesh
            if mesh is not None:
                for face in mesh.faces_as_vertices():
                    pts = [(v.x, v.y) for v in face]
                    yield pts + pts[:1]


def _keep_in_walls(ins, u: float, cache: dict) -> bool:
    """Keep doors (a ~90° swing arc of door size) and windows (thin, long blocks set in a wall);
    drop furniture and sanitary fixtures, which only fragment the floor."""
    try:
        b = bbox.extents([ins], fast=True)
    except Exception:  # noqa: BLE001
        return True
    if not b.has_data:
        return False
    w, h = sorted((b.size.x * u, b.size.y * u))
    if w <= 0.4 and h >= 0.5:                     # window / wall segment
        return True
    name = ins.dxf.name
    if name not in cache:
        door = False
        try:
            for e in ins.virtual_entities():
                if e.dxftype() == "ARC":
                    r = e.dxf.radius * u
                    sweep = (e.dxf.end_angle - e.dxf.start_angle) % 360
                    if 0.5 <= r <= 1.4 and 60 <= sweep <= 100:
                        door = True
                        break
        except Exception:  # noqa: BLE001
            pass
        cache[name] = door
    return cache[name]


def _raster_lines(shape, ox, oy, px, polylines) -> np.ndarray:
    h, w = shape
    img = np.zeros(shape, dtype=bool)
    step = px * 0.5
    for pts in polylines:
        a = np.asarray(pts, dtype=float)
        for (x0, y0), (x1, y1) in zip(a[:-1], a[1:]):
            n = int(math.hypot(x1 - x0, y1 - y0) / step) + 2
            xs = np.linspace(x0, x1, n)
            ys = np.linspace(y0, y1, n)
            j = ((xs - ox) / px).astype(int)
            i = ((ys - oy) / px).astype(int)
            ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
            img[i[ok], j[ok]] = True
    return img


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _polygon_mask(shape, ox, oy, px, pts) -> np.ndarray:
    h, w = shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    xs = ox + (jj + 0.5) * px
    ys = oy + (ii + 0.5) * px
    path = MplPath(np.asarray(pts))
    return path.contains_points(np.c_[xs.ravel(), ys.ravel()]).reshape(shape)


def _crop(mask, ox, oy, px, pad=2):
    ii, jj = np.nonzero(mask)
    if ii.size == 0:
        return mask, ox, oy
    i0, i1 = max(ii.min() - pad, 0), min(ii.max() + pad + 1, mask.shape[0])
    j0, j1 = max(jj.min() - pad, 0), min(jj.max() + pad + 1, mask.shape[1])
    return mask[i0:i1, j0:j1].copy(), ox + j0 * px, oy + i0 * px


# ----------------------------------------------------------------------------- main
def parse(doc, proj: Project, u: float, debug: dict | None = None) -> Drawing:
    msp = doc.modelspace()
    sheets = find_sheets(doc, proj)
    lay = proj.layers
    rd = proj.room_detection
    geo = proj.geometry
    em_cfg = proj.rules.emergency
    em_re = re.compile(em_cfg.get("emergency_block_regex", r"$^"), re.I)
    em_layers = set(em_cfg.get("emergency_layers", []))
    warnings: list[str] = []

    # --- luminaires -----------------------------------------------------------------
    lums: list[Luminaire] = []
    unknown_lighting = Counter()
    for ins in msp.query("INSERT"):
        name = ins.dxf.name
        spec = proj.catalog.get(name)
        is_em = ins.dxf.layer in em_layers or bool(em_re.search(name))
        if spec is None and not is_em:
            if re.search(r"LIGHT|LITE|LUM|LAMP|ECL", ins.dxf.layer, re.I):
                unknown_lighting[name] += 1
            continue
        spec = dict(spec or {})
        if spec.get("distribution") == "ignore":
            continue
        attrs = {a.dxf.tag.upper(): a.dxf.text for a in ins.attribs}
        for k_attr, k_spec in (("LUMENS", "lumens"), ("LM", "lumens"), ("WATT", "watts"), ("W", "watts"),
                               ("IES", "ies"), ("MODEL", "description")):
            if attrs.get(k_attr):
                v = attrs[k_attr]
                try:
                    spec[k_spec] = float(re.sub(r"[^\d.]", "", v)) if k_spec in ("lumens", "watts") else v
                except ValueError:
                    pass
        p = ins.dxf.insert
        lums.append(Luminaire(name, ins.dxf.layer, p.x, p.y, ins.dxf.rotation, spec, emergency=is_em))
    for name, n in unknown_lighting.most_common():
        warnings.append(f"Block '{name}' (x{n}) is on a lighting layer but not in the luminaire catalogue.")

    for lm in lums:
        for s in sheets:
            if s.contains(lm.x, lm.y):
                lm.sheet = s.name
                break

    # --- texts ------------------------------------------------------------------------
    tag_layers = set(lay.get("room_tag", []))
    texts = [(_text_of(t), *_centre(t)) for t in msp.query("TEXT MTEXT")
             if not tag_layers or t.dxf.layer in tag_layers]
    h_re = re.compile(geo.get("ceiling_height_regex", r"$^"), re.I)

    # --- egress polylines ---------------------------------------------------------------
    egress = []
    for e in msp.query("LWPOLYLINE POLYLINE LINE"):
        if e.dxf.layer in set(lay.get("egress", [])):
            pts = next(_primitive_polylines([e], 0.01 / u), None)
            if pts:
                egress.append(pts)

    # --- rooms per sheet ------------------------------------------------------------------
    rooms: list[Room] = []
    px = rd.get("pixel_m", 0.05) / u
    boundary_layers = set(lay.get("room_boundary", []))
    wall_layers = set(lay.get("walls", []))
    excl = re.compile(lay.get("wall_exclude_blocks_regex") or r"$^")
    default_h = float(geo.get("ceiling_height_m", 3.0))
    min_a = rd.get("min_room_area_m2", 1.0) / (u * u)
    max_a = rd.get("max_room_area_m2", 150) / (u * u)

    for sh in sheets:
        shape = (int((sh.y1 - sh.y0) / px) + 1, int((sh.x1 - sh.x0) / px) + 1)
        ox, oy = sh.x0, sh.y0
        sh_texts = [(s, x, y) for s, x, y in texts if sh.contains(x, y) and s]
        label_texts = [(s, x, y) for s, x, y in sh_texts if not h_re.search(s)]
        n_before = len(rooms)

        # (1) explicit boundaries
        polys = [e for e in msp.query("LWPOLYLINE POLYLINE")
                 if e.dxf.layer in boundary_layers and e.is_closed and sh.contains(*_centre(e))]
        for k, e in enumerate(polys):
            pts = next(_primitive_polylines([e], 0.01 / u))
            mask = _polygon_mask(shape, ox, oy, px, pts)
            path = MplPath(np.asarray(pts))
            inside = [t for t in label_texts if path.contains_point((t[1], t[2]))]
            label = inside[0][0] if inside else None
            hs = [float(m.group("h")) for s, x, y in sh_texts
                  if (m := h_re.search(s)) and path.contains_point((x, y))]
            m2, rox, roy = _crop(mask, ox, oy, px)
            rooms.append(Room(f"{sh.floor}/{label or 'room'}#{k + 1}", sh.name, sh.floor, label,
                              proj.rules.classify(label), "polyline", m2, rox, roy, px,
                              hs[0] if hs else default_h))

        # (2) walls + flood fill
        if not polys:
            door_cache: dict = {}
            ents = []
            for e in msp:
                if e.dxf.layer not in wall_layers or e.dxftype() in ("TEXT", "MTEXT", "ATTRIB", "DIMENSION",
                                                                     "POINT", "OLE2FRAME", "IMAGE"):
                    continue
                if e.dxftype() == "INSERT" and (excl.search(e.dxf.name) or not _keep_in_walls(e, u, door_cache)):
                    continue
                if e.dxftype() in ("LWPOLYLINE", "POLYLINE") and e.is_closed:
                    b = bbox.extents([e], fast=True)
                    ww, hh = sorted((b.size.x * u, b.size.y * u))
                    if hh < 1.0 and ww > 0.3:              # small closed outline: fixture / furniture
                        continue
                cx, cy = _centre(e)
                if sh.contains(cx, cy):
                    ents.append(e)
            walls = _raster_lines(shape, ox, oy, px, _primitive_polylines(ents, px * 0.5))
            seal_r = max(1, int(round(rd.get("seal_gap_m", 0.15) / u / px / 2)))
            # floor-finish hatches (tiles) outline wet rooms that fixtures fragment
            hatch_polys = []
            for e in ents:
                if e.dxftype() != "HATCH" or e.dxf.solid_fill:
                    continue
                for pts in _primitive_polylines([e], px * 0.5):
                    a = np.asarray(pts)
                    area = 0.5 * abs(np.dot(a[:-1, 0], a[1:, 1]) - np.dot(a[1:, 0], a[:-1, 1]))
                    if min_a <= area <= max_a:
                        hatch_polys.append((MplPath(a), area, pts))
            hatch_polys.sort(key=lambda t: t[1])
            used_hatch: set[int] = set()

            claimed = np.zeros(shape, dtype=bool)

            def fragment_room(x, y, hint=None, radius_m=1.8):
                """Merge the small free fragments fully inside a disk around (x, y) into one room."""
                r_px = int(radius_m / u / px)
                i, j = int((y - oy) / px), int((x - ox) / px)
                i0, i1 = max(i - r_px, 0), min(i + r_px + 1, shape[0])
                j0, j1 = max(j - r_px, 0), min(j + r_px + 1, shape[1])
                yy, xx = np.ogrid[i0:i1, j0:j1]
                disk = (yy - i) ** 2 + (xx - j) ** 2 <= r_px * r_px
                win = lab[i0:i1, j0:j1]
                inside = np.bincount(win[disk].ravel(), minlength=n + 1)
                total = sizes / (px * px)
                cand = [c for c in np.nonzero(inside)[0] if c and c not in border and c not in used
                        and inside[c] >= 0.98 * total[c] and sizes[c] < max_a]
                if not cand:
                    return None
                m = np.isin(lab, cand) & ~claimed
                m = ndimage.binary_closing(m, _disk(max(2, seal_r * 2)))
                m = ndimage.binary_fill_holes(m)
                lbl, _ = ndimage.label(m)
                k = lbl[min(max(i, 0), shape[0] - 1), min(max(j, 0), shape[1] - 1)]
                if not k:                                   # seed on a line: take the nearest piece
                    vals = lbl[i0:i1, j0:j1][disk]
                    vals = vals[vals > 0]
                    if not vals.size:
                        return None
                    k = np.bincount(vals).argmax()
                m = (lbl == k) & ~claimed
                if m.sum() * px * px < min_a:
                    return None
                for c in cand:
                    if (m & (lab == c)).any():
                        used[c] = "<fragment>"
                claimed[m] = True
                m2, rox, roy = _crop(m, ox, oy, px)
                n_ = sum(1 for r in rooms if r.sheet == sh.name) + 1
                r = Room(f"{sh.floor}/{hint or 'room'}#{n_}", sh.name, sh.floor, None, hint or "unknown",
                         "fragments", m2, rox, roy, px, default_h)
                r.notes.append("approximate boundary: assembled from fragments split by fixtures/partitions; "
                               "draw a closed room boundary for an exact check")
                return r

            def hatch_room(x, y, label, source, hint=None):
                for k, (path, area, pts) in enumerate(hatch_polys):
                    if k in used_hatch or not path.contains_point((x, y)):
                        continue
                    used_hatch.add(k)
                    mask = _polygon_mask(shape, ox, oy, px, pts)
                    claimed[mask] = True
                    m2, rox, roy = _crop(mask, ox, oy, px)
                    rtype = proj.rules.classify(label) if label else (hint or "unknown")
                    n_ = sum(1 for r in rooms if r.sheet == sh.name) + 1
                    r = Room(f"{sh.floor}/{label or (hint or 'room')}#{n_}", sh.name, sh.floor, label, rtype,
                             source, m2, rox, roy, px, default_h)
                    r.notes.append("boundary taken from the floor-finish hatch")
                    return r
                return None

            walls_d = ndimage.binary_dilation(walls, _disk(seal_r))
            free = ~walls_d
            lab, n = ndimage.label(free)
            border = set(np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])) - {0}
            sizes = ndimage.sum(free, lab, index=np.arange(n + 1)) * px * px
            # `own`: every pixel enclosed by a room-sized component (furniture islands included)
            own = np.zeros_like(lab)
            for c, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None or c in border or not (min_a <= sizes[c] <= max_a):
                    continue
                filled = ndimage.binary_fill_holes(lab[sl] == c)
                own[sl][filled & (own[sl] == 0)] = c
            used: dict[int, str] = {}
            if debug is not None:
                debug[sh.floor] = dict(lab=lab, own=own, walls=walls, sizes=sizes, border=border, ox=ox, oy=oy, px=px)

            def region_at(x, y, search_px=int(0.5 / u / px)):
                i, j = int((y - oy) / px), int((x - ox) / px)
                if not (0 <= i < shape[0] and 0 <= j < shape[1]):
                    return 0
                if own[i, j]:
                    return int(own[i, j])
                if lab[i, j] and (lab[i, j] in border or sizes[lab[i, j]] > max_a):
                    return int(lab[i, j])                      # open / leaking space
                if search_px <= 0:
                    return 0
                win = own[max(i - search_px, 0):i + search_px + 1, max(j - search_px, 0):j + search_px + 1]
                vals = win[win > 0]
                return int(np.bincount(vals).argmax()) if vals.size else 0

            def make_room(reg, label, source, hint=None, sub=None):
                mask = (own == reg) if (own == reg).any() else (lab == reg)
                mask = ndimage.binary_dilation(mask, _disk(seal_r)) & ~walls   # give back the sealing margin
                mask = ndimage.binary_fill_holes(mask)
                if sub is not None:
                    mask &= sub
                a = mask.sum() * px * px
                status, notes = "ok", []
                if reg in border or a > max_a:
                    status = "leak"
                    notes.append("flood fill escaped the room (open wall or missing door swing); "
                                 "draw a closed boundary on the room layer")
                elif a < min_a:
                    status = "too_small"
                claimed[mask] = True
                m2, rox, roy = _crop(mask, ox, oy, px)
                rtype = proj.rules.classify(label) if label else (hint or "unknown")
                hs = [float(m.group("h")) for s, x, y in sh_texts
                      if (m := h_re.search(s)) and region_at(x, y) == reg]
                k = sum(1 for r in rooms if r.sheet == sh.name) + 1
                return Room(f"{sh.floor}/{label or (hint or 'room')}#{k}", sh.name, sh.floor, label, rtype,
                            source, m2, rox, roy, px, hs[0] if hs else default_h, status, notes)

            groups: dict[int, list] = {}
            for s, x, y in label_texts:
                reg = region_at(x, y, search_px=0) or (0 if any(p.contains_point((x, y)) for p, *_ in hatch_polys)
                                                         else region_at(x, y))
                if not reg:
                    r = hatch_room(x, y, s, "hatch")
                    if r:
                        rooms.append(r)
                    else:
                        warnings.append(f"[{sh.floor}] label '{s}' is not inside any closed region")
                    continue
                g = groups.setdefault(reg, [])
                if any(s == t[0] and math.hypot(x - t[1], y - t[2]) < 2.0 / u for t in g):
                    continue                                   # duplicate label in the same room
                g.append((s, x, y))
            jj, ii = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            for reg, g in groups.items():
                used[reg] = g[0][0]
                if len(g) == 1:
                    rooms.append(make_room(reg, g[0][0], "walls"))
                    continue
                # open-plan space with several labels: split by nearest label (Voronoi)
                pts = np.array([((x - ox) / px, (y - oy) / px) for _, x, y in g])
                d2 = (jj[..., None] - pts[:, 0]) ** 2 + (ii[..., None] - pts[:, 1]) ** 2
                nearest = d2.argmin(axis=-1)
                for k, (s, x, y) in enumerate(g):
                    r = make_room(reg, s, "walls", sub=(nearest == k))
                    r.notes.append("open-plan space shared with " + ", ".join(t[0] for t in g if t[0] != s)
                                   + "; split by nearest label")
                    rooms.append(r)

            if rd.get("seed_unlabelled_rooms_from_luminaires", True):
                for lm in lums:
                    if lm.sheet != sh.name or lm.spec.get("mounting") == "wall":
                        continue
                    reg = region_at(lm.x, lm.y, search_px=0)
                    if not reg or sizes[reg] < min_a:
                        if not any(r.sheet == sh.name and r.contains(lm.x, lm.y) for r in rooms):
                            r = (hatch_room(lm.x, lm.y, None, "hatch", lm.spec.get("room_hint"))
                                 or fragment_room(lm.x, lm.y, lm.spec.get("room_hint")))
                            if r:
                                rooms.append(r)
                        continue
                    if reg in used or reg in border or not (min_a <= sizes[reg] <= max_a):
                        continue
                    used[reg] = f"<{lm.block}>"
                    rooms.append(make_room(reg, None, "seed", lm.spec.get("room_hint")))

        if len(rooms) == n_before:
            warnings.append(f"[{sh.floor}] no rooms found")

    # --- luminaire -> room ----------------------------------------------------------------
    unassigned = []
    by_sheet: dict[str, list[Room]] = {}
    for r in rooms:
        by_sheet.setdefault(r.sheet, []).append(r)
    for lm in lums:
        cands = by_sheet.get(lm.sheet, [])
        hit = next((r for r in cands if r.contains(lm.x, lm.y)), None)
        if hit is None:  # wall-mounted or drawn on the wall line: look around
            tol = int(0.6 / u / cands[0].px) if cands else 0
            hit = next((r for r in cands if r.contains(lm.x, lm.y, tol)), None)
        if hit is None:
            unassigned.append(lm)
            continue
        lm.room_id = hit.id
        hit.luminaires.append(lm)

    return Drawing(sheets, rooms, lums, egress, u, warnings, unassigned)
