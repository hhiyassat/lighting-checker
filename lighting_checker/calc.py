"""Illuminance calculations.

* Lumen method (quick check):           E = N·Φ·UF·MF / A
* Point-by-point direct (per grid point): E = Σ I(C,γ)·cosγ / d²  (= I·cos³γ / h²)
* Inter-reflections (approximation):     split-flux, E_ind = Φ·ρ̄ / (A_s·(1−ρ̄))  added uniformly
  → replace by Radiance / radiosity for a DIALux-grade result.
* UGR (CIE 117 formula with the Guth position index) — approximate, see `ugr()`.
* Emergency lighting (EN 1838): escape-route centre lines and anti-panic areas, direct light only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import photometry
from .config import Project
from .parser import Drawing, Luminaire, Room, _disk

# Generic utilisation factors vs room index K (direct downlights, reflectances ≈ 0.7/0.5/0.2).
# Replace by the manufacturer's UF table when available.
UF_TABLE = np.array([[0.6, 0.35], [0.8, 0.43], [1.0, 0.48], [1.25, 0.53], [1.5, 0.57],
                     [2.0, 0.62], [2.5, 0.66], [3.0, 0.68], [4.0, 0.71], [5.0, 0.73]])


def grid_spacing(d_m: float) -> float:
    """EN 12464-1 grid rule: p = 0.2 · 5^log10(d), d = longer dimension of the area (m), p ≤ 10 m."""
    d = max(d_m, 0.1)
    return min(0.2 * 5 ** math.log10(d), 10.0)


@dataclass
class RoomResult:
    room: Room
    area_m2: float
    n_lum: int
    flux_lm: float
    watts: float
    mount_h_m: float
    room_index: float | None
    uf: float | None
    e_lumen: float | None
    e_avg: float | None
    e_min: float | None
    e_max: float | None
    uo: float | None
    e_direct_avg: float | None
    e_indirect: float | None
    ugr: float | None
    lpd_w_m2: float | None
    grid: tuple | None = None          # (xs_m, ys_m, E) for the heat-map
    grid_p_m: float | None = None
    requirement: dict | None = None
    checks: dict = field(default_factory=dict)
    verdict: str = "n/a"
    deficit_lx: float | None = None
    extra_luminaires: int | None = None
    spacing: tuple | None = None       # (d_max_m, h_m, d/h) over ceiling/pendant luminaires
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------- geometry helpers
def luminaire_z(lm: Luminaire, ceiling_h: float) -> float:
    s = lm.spec
    mnt = s.get("mounting", "ceiling")
    if "mount_height_m" in s:
        return float(s["mount_height_m"])
    if mnt == "pendant":
        return ceiling_h - float(s.get("suspension_m", 0.5))
    if mnt == "wall":
        return min(2.2, ceiling_h - 0.2)
    return ceiling_h


def spacing_ratio(lums, u: float, ceiling_h: float, work_plane: float):
    """Spacing-to-height ratio: the largest nearest-neighbour horizontal distance between ceiling or pendant
    luminaires, over their mounting height above the work plane. None with fewer than two such luminaires."""
    pts = [lm for lm in lums if lm.spec.get("mounting", "ceiling") != "wall"]
    if len(pts) < 2:
        return None
    xy = np.array([(lm.x * u, lm.y * u) for lm in pts])
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    d_max = float(d.min(axis=1).max())
    h = max(float(np.mean([luminaire_z(lm, ceiling_h) for lm in pts])) - work_plane, 0.3)
    return d_max, h, d_max / h


def direct_illuminance(px_m, py_m, pz_m, lums, u, ceiling_h, resolve, normal=(0, 0, 1)) -> np.ndarray:
    """Illuminance on a plane with the given normal at points (px, py, pz) [m] from luminaires."""
    E = np.zeros_like(px_m, dtype=float)
    nx, ny, nz = normal
    for lm in lums:
        ph = photometry.for_luminaire(lm.spec, resolve)
        lx, ly, lz = lm.x * u, lm.y * u, luminaire_z(lm, ceiling_h)
        dx, dy, dz = px_m - lx, py_m - ly, pz_m - lz            # luminaire → point
        d2 = dx * dx + dy * dy + dz * dz
        d = np.sqrt(np.maximum(d2, 1e-9))
        gamma = np.degrees(np.arccos(np.clip(-dz / d, -1, 1)))  # 0 = straight down
        rot = math.radians(lm.rotation or 0.0)
        cx = dx * math.cos(rot) + dy * math.sin(rot)            # into the luminaire's local frame
        cy = -dx * math.sin(rot) + dy * math.cos(rot)
        C = np.degrees(np.arctan2(cy, cx))
        I = ph.intensity(C, gamma)
        cos_inc = -(dx * nx + dy * ny + dz * nz) / d             # incidence on the receiving plane
        E += np.where(cos_inc > 0, I * cos_inc / np.maximum(d2, 1e-9), 0.0)
    return E


def _room_dims_m(room: Room, u: float):
    ys, xs = np.nonzero(room.mask)
    w = (xs.max() - xs.min() + 1) * room.px * u
    h = (ys.max() - ys.min() + 1) * room.px * u
    return w, h


def calc_grid(room: Room, u: float, wall_band: float):
    """Calculation points on the work plane: EN 12464-1 spacing, 0.5 m wall band excluded."""
    L, W = _room_dims_m(room, u)
    p = grid_spacing(max(L, W))
    px_m = room.px * u
    band_px = int(round(wall_band / px_m))
    inner = ndimage.binary_erosion(room.mask, _disk(band_px)) if band_px > 0 else room.mask
    note = None
    if inner.sum() < 4:
        inner, note = room.mask, "room too small for the 0.5 m wall band; band not applied"
    step = max(1, int(round(p / px_m)))
    ii, jj = np.nonzero(inner)
    # regular lattice anchored at the mask's lower-left corner, offset by half a step
    i0, j0 = ii.min(), jj.min()
    sel = ((ii - i0) % step == step // 2) & ((jj - j0) % step == step // 2)
    if sel.sum() < 4:                                   # tiny room: ~25 evenly spread pixels
        k = max(1, len(ii) // 25)
        sel = np.zeros_like(ii, dtype=bool)
        sel[::k] = True
    xs = (room.ox + (jj[sel] + 0.5) * room.px) * u
    ys = (room.oy + (ii[sel] + 0.5) * room.px) * u
    return xs, ys, step * px_m, note


def indirect_component(flux_total, area, perim, height, rho_c, rho_w, rho_f):
    a_c, a_f, a_w = area, area, perim * height
    a_s = a_c + a_f + a_w
    rho = (rho_c * a_c + rho_f * a_f + rho_w * a_w) / a_s
    return flux_total * rho / (a_s * (1 - rho))


def utilisation_factor(k: float) -> float:
    return float(np.interp(k, UF_TABLE[:, 0], UF_TABLE[:, 1]))


# ----------------------------------------------------------------------------- UGR (CIE 117)
def guth_position_index(alpha_deg, beta_deg):
    a, b = alpha_deg, beta_deg
    return np.exp((35.2 - 0.31889 * a - 1.22 * np.exp(-2 * a / 9)) * 1e-3 * b
                  + (21 + 0.26667 * a - 0.002963 * a * a) * 1e-5 * b * b)


def ugr(room: Room, lums, u, ceiling_h, e_ind, resolve, eye_h=1.2):
    """Maximum UGR over four observers (mid-points of the room sides, 1 m in, looking across the room).
    Background luminance Lb = E_indirect / π. Approximate: real UGR tables come from the luminaire
    data (CIE 117 tabular method) — use those for formal compliance."""
    if not lums or e_ind <= 0:
        return None
    Lb = e_ind / math.pi
    ys, xs = np.nonzero(room.mask)
    cx, cy = (room.ox + (xs.mean() + 0.5) * room.px) * u, (room.oy + (ys.mean() + 0.5) * room.px) * u
    x0, x1 = (room.ox + xs.min() * room.px) * u, (room.ox + (xs.max() + 1) * room.px) * u
    y0, y1 = (room.oy + ys.min() * room.px) * u, (room.oy + (ys.max() + 1) * room.px) * u
    inset = min(1.0, 0.25 * min(x1 - x0, y1 - y0))
    observers = [((x0 + inset, cy), (1, 0)), ((x1 - inset, cy), (-1, 0)),
                 ((cx, y0 + inset), (0, 1)), ((cx, y1 - inset), (0, -1))]
    worst = None
    for (ex, ey), (fx, fy) in observers:
        s = 0.0
        for lm in lums:
            ph = photometry.for_luminaire(lm.spec, resolve)
            area = float(lm.spec.get("luminous_area_m2") or (ph.width_m * ph.length_m) or 0.0)
            if area <= 0:
                continue
            lx, ly, lz = lm.x * u, lm.y * u, luminaire_z(lm, ceiling_h)
            R = (lx - ex) * fx + (ly - ey) * fy          # along the line of sight
            T = -(lx - ex) * fy + (ly - ey) * fx         # lateral
            H = lz - eye_h                               # above the eye
            if R <= 0.1 or H <= 0:
                continue
            d = math.sqrt(R * R + T * T + H * H)
            gamma = math.degrees(math.acos(H / d))       # direction luminaire → eye, from nadir
            C = math.degrees(math.atan2(ey - ly, ex - lx)) - (lm.rotation or 0.0)
            I = float(ph.intensity(C, gamma))
            dist = lm.spec.get("distribution", "downlight")
            a_proj = area * (math.cos(math.radians(gamma)) if dist == "downlight" else 1.0)
            if a_proj <= 1e-6 or I <= 0:
                continue
            L = I / a_proj
            omega = a_proj / (d * d)
            alpha = math.degrees(math.atan2(abs(T), H))
            beta = math.degrees(math.acos(R / d))
            p = float(guth_position_index(alpha, beta))
            s += L * L * omega / (p * p)
        if s > 0:
            val = 8 * math.log10(0.25 / Lb * s)
            worst = val if worst is None else max(worst, val)
    return worst


# ----------------------------------------------------------------------------- per room
def evaluate_room(room: Room, proj: Project, u: float) -> RoomResult:
    geo, srf = proj.geometry, proj.surfaces
    mf = proj.maintenance_factor
    wp = float(geo.get("work_plane_m", 0.8))
    band = float(geo.get("wall_band_m", 0.5))
    H = room.ceiling_h_m
    A = room.area_du2 * u * u
    P = room.perimeter_du() * u
    lums = [lm for lm in room.luminaires if not lm.emergency]
    flux = sum(float(lm.spec.get("lumens", 0)) for lm in lums)
    watts = sum(float(lm.spec.get("watts", 0)) for lm in lums)
    req = proj.rules.requirement(room.type)
    res = RoomResult(room, A, len(lums), flux, watts, 0.0, None, None, None, None, None, None, None,
                     None, None, None, watts / A if A else None, requirement=req)
    res.notes += room.notes
    if room.status != "ok":
        res.verdict = "not assessed"
        res.notes.append(f"room boundary status: {room.status}")
        return res
    resolve = proj.resolve
    if lums:
        zs = [luminaire_z(lm, H) for lm in lums]
        hm = max(float(np.mean(zs)) - wp, 0.3)
        res.mount_h_m = hm
        K = 2 * A / (hm * P) if P > 0 else None
        res.room_index = K
        res.uf = utilisation_factor(K) if K else None
        res.e_lumen = flux * (res.uf or 0) * mf / A
    xs, ys, p, note = calc_grid(room, u, band)
    res.grid_p_m = p
    if note:
        res.notes.append(note)
    Ed = direct_illuminance(xs, ys, np.full_like(xs, wp), lums, u, H, resolve) if lums else np.zeros_like(xs)
    Ei = indirect_component(flux, A, P, H, srf.get("rho_ceiling", 0.7), srf.get("rho_walls", 0.5),
                            srf.get("rho_floor", 0.2)) if lums else 0.0
    E = (Ed + Ei) * mf
    res.grid = (xs, ys, E)
    res.e_direct_avg = float(Ed.mean() * mf)
    res.e_indirect = float(Ei * mf)
    res.e_avg, res.e_min, res.e_max = float(E.mean()), float(E.min()), float(E.max())
    res.uo = res.e_min / res.e_avg if res.e_avg > 0 else 0.0
    res.ugr = ugr(room, lums, u, H, Ei * mf, resolve) if lums else None

    if req is None:
        res.verdict = "no rule"
        res.notes.append(f"no requirement for room type '{room.type}' (label: {room.label or '—'})")
        return res
    ok = True
    em_req = float(req["em_lx"])
    res.checks["Em"] = (round(res.e_avg), em_req, res.e_avg >= em_req)
    ok &= res.e_avg >= em_req
    if req.get("uo") is not None:
        res.checks["Uo"] = (round(res.uo, 2), req["uo"], res.uo >= req["uo"])
        ok &= res.uo >= req["uo"]
    if req.get("ugr_max") is not None and res.ugr is not None:
        res.checks["UGR"] = (round(res.ugr, 1), req["ugr_max"], res.ugr <= req["ugr_max"])
        ok &= res.ugr <= req["ugr_max"]
    if req.get("ra") is not None:
        ras = [lm.spec.get("ra") for lm in lums if lm.spec.get("ra") is not None]
        if ras:
            res.checks["Ra"] = (min(ras), req["ra"], min(ras) >= req["ra"])
            ok &= min(ras) >= req["ra"]
    sp = proj.rules.spacing
    if sp.get("max_ratio") is not None:
        res.spacing = spacing_ratio(lums, u, H, wp)
        if res.spacing is None:
            res.notes.append("d/h: not applicable (fewer than two ceiling luminaires)")
        else:
            d_max, h, ratio = res.spacing
            okd = ratio <= float(sp["max_ratio"])
            res.checks["d/h"] = (round(ratio, 2), sp["max_ratio"], okd)
            ok &= okd
            if not okd:
                res.notes.append(f"luminaire spacing {d_max:.2f} m over mounting height {h:.2f} m = {ratio:.2f} "
                                 f"> {sp['max_ratio']} ({sp.get('clause', '')})")
    res.verdict = "pass" if ok else "fail"
    if res.e_avg < em_req:
        res.deficit_lx = em_req - res.e_avg
        per = (flux / len(lums)) if lums else 1000.0
        gain_per_lum = res.e_avg / len(lums) if lums and res.e_avg > 0 else per * (res.uf or 0.5) * mf / A
        res.extra_luminaires = math.ceil(res.deficit_lx / gain_per_lum) if gain_per_lum > 0 else None
    return res


# ----------------------------------------------------------------------------- emergency (EN 1838)
@dataclass
class EmergencyResult:
    present: bool
    n_luminaires: int
    routes: list = field(default_factory=list)      # (sheet, length_m, Emin, Emax, pass)
    anti_panic: list = field(default_factory=list)  # (room id, area, Emin, pass)
    notes: list = field(default_factory=list)


def evaluate_emergency(dr: Drawing, proj: Project) -> EmergencyResult:
    em = proj.rules.emergency
    u = dr.u
    lums = [lm for lm in dr.luminaires if lm.emergency]
    out = EmergencyResult(bool(lums), len(lums))
    if not lums:
        out.notes.append("No emergency luminaires found (layer/block convention not met) — EN 1838 not assessed.")
        return out
    h = float(proj.geometry.get("ceiling_height_m", 3.0))
    for pts in dr.egress:
        a = np.asarray(pts) * u
        seg = np.diff(a, axis=0)
        L = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        n = max(int(L / 0.25), 2)
        t = np.linspace(0, 1, n)
        cum = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))] / max(L, 1e-9)
        xs, ys = np.interp(t, cum, a[:, 0]), np.interp(t, cum, a[:, 1])
        E = direct_illuminance(xs, ys, np.zeros_like(xs), lums, u, h, proj.resolve)
        ok = E.min() >= em.get("escape_route_min_lx", 1.0) and \
            (E.max() / max(E.min(), 1e-9)) <= em.get("escape_route_uniformity_max_ratio", 40)
        out.routes.append((L, float(E.min()), float(E.max()), ok))
    if not dr.egress:
        out.notes.append("No escape-route polylines found — route check skipped.")
    for r in dr.rooms:
        A = r.area_du2 * u * u
        if r.status != "ok" or A < em.get("anti_panic_min_area_m2", 60):
            continue
        xs, ys, _, _ = calc_grid(r, u, 0.5)
        E = direct_illuminance(xs, ys, np.zeros_like(xs), lums, u, r.ceiling_h_m, proj.resolve)
        out.anti_panic.append((r.id, A, float(E.min()), float(E.min()) >= em.get("anti_panic_min_lx", 0.5)))
    return out
