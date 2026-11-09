"""Luminous intensity distributions: IES LM-63 and EULUMDAT (.ldt) readers + generic shapes.

Every distribution is stored on a full grid C ∈ [0, 360] (inclusive), γ ∈ [0, 180] in candela
for the luminaire's rated flux, and evaluated with bilinear interpolation.
γ = 0 is nadir (straight down); C = 0 is along the luminaire's local +x axis.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Photometry:
    c: np.ndarray            # (nc,) degrees, 0..360 inclusive
    g: np.ndarray            # (ng,) degrees, 0..180
    cd: np.ndarray           # (nc, ng) candela
    flux_lm: float           # luminaire output the table corresponds to
    name: str = ""
    width_m: float = 0.0
    length_m: float = 0.0
    meta: dict = field(default_factory=dict)

    def intensity(self, c_deg, g_deg) -> np.ndarray:
        c = np.mod(np.asarray(c_deg, float), 360.0)
        g = np.clip(np.asarray(g_deg, float), 0.0, 180.0)
        ic = np.clip(np.searchsorted(self.c, c, side="right") - 1, 0, len(self.c) - 2)
        ig = np.clip(np.searchsorted(self.g, g, side="right") - 1, 0, len(self.g) - 2)
        c0, c1 = self.c[ic], self.c[ic + 1]
        g0, g1 = self.g[ig], self.g[ig + 1]
        tc = np.where(c1 > c0, (c - c0) / np.where(c1 > c0, c1 - c0, 1), 0.0)
        tg = np.where(g1 > g0, (g - g0) / np.where(g1 > g0, g1 - g0, 1), 0.0)
        v00 = self.cd[ic, ig]
        v01 = self.cd[ic, ig + 1]
        v10 = self.cd[ic + 1, ig]
        v11 = self.cd[ic + 1, ig + 1]
        return (v00 * (1 - tc) * (1 - tg) + v01 * (1 - tc) * tg + v10 * tc * (1 - tg) + v11 * tc * tg)

    def scaled(self, flux_lm: float) -> "Photometry":
        k = flux_lm / self.flux_lm if self.flux_lm > 0 else 1.0
        return Photometry(self.c, self.g, self.cd * k, flux_lm, self.name, self.width_m, self.length_m, self.meta)

    def total_flux(self) -> float:
        """Numerically integrate the distribution (sanity check: should ≈ flux_lm·LOR)."""
        c = np.radians(self.c)
        g = np.radians(self.g)
        # ∫∫ I sinγ dγ dC
        inner = np.trapezoid(self.cd * np.sin(g)[None, :], g, axis=1)
        return float(np.trapezoid(inner, c))


# ----------------------------------------------------------------------------- generic shapes
def generic(kind: str, flux_lm: float, n_c: int = 2, n_g: int = 181) -> Photometry:
    """Fallback distributions when no photometric file is supplied.

    downlight: Lambertian into the lower hemisphere, I(γ) = Φ/π · cosγ
    diffuse:   isotropic sphere, I = Φ / 4π (globes, chandeliers, opal pendants)
    wall:      treated as diffuse (orientation of wall luminaires is not known from the plan)
    """
    g = np.linspace(0, 180, n_g)
    c = np.array([0.0, 360.0]) if n_c == 2 else np.linspace(0, 360, n_c)
    if kind == "downlight":
        prof = np.where(g <= 90, flux_lm / math.pi * np.cos(np.radians(g)), 0.0)
    elif kind in ("diffuse", "wall"):
        prof = np.full_like(g, flux_lm / (4 * math.pi))
    else:
        raise ValueError(f"unknown generic distribution '{kind}'")
    cd = np.tile(prof, (len(c), 1))
    return Photometry(c, g, cd, flux_lm, name=f"generic:{kind}")


# ----------------------------------------------------------------------------- helpers
def _expand_c(c: np.ndarray, cd: np.ndarray):
    """Expand a partial set of C-planes to the full 0..360 range using the usual symmetries:
    one plane (rotational), 0–90 (quadrant), 0–180 (about C0–C180), 90–270 (about C90–C270), 0–360."""
    c = np.asarray(c, float)
    cd = np.asarray(cd, float)
    if len(c) == 1:
        return np.array([0.0, 360.0]), np.vstack([cd[0], cd[0]])
    c0, c1 = c[0], c[-1]
    if math.isclose(c0, 0) and math.isclose(c1, 90):
        fold = lambda C: np.where(np.mod(C, 180) > 90, 180 - np.mod(C, 180), np.mod(C, 180))  # noqa: E731
    elif math.isclose(c0, 0) and math.isclose(c1, 180):
        fold = lambda C: np.where(C > 180, 360 - C, C)  # noqa: E731
    elif math.isclose(c0, 90) and math.isclose(c1, 270):
        fold = lambda C: np.where((C >= 90) & (C <= 270), C, np.mod(180 - C, 360))  # noqa: E731
    else:                                               # full (or nearly full) set
        if not math.isclose(c1, 360):
            c = np.r_[c, 360.0]
            cd = np.vstack([cd, cd[0]])
        return c, cd
    step = float(np.min(np.diff(c))) if len(c) > 1 else 90.0
    grid = np.linspace(0, 360, int(round(360 / step)) + 1)
    src = fold(grid)
    out = np.empty((len(grid), cd.shape[1]))
    for j in range(cd.shape[1]):
        out[:, j] = np.interp(src, c, cd[:, j])
    return grid, out


def _expand_g(g: np.ndarray, cd: np.ndarray):
    g = np.asarray(g, float)
    if g[0] > 0:
        g = np.r_[0.0, g]
        cd = np.hstack([cd[:, :1], cd])
    if g[-1] < 180:
        g = np.r_[g, 180.0]
        cd = np.hstack([cd, np.zeros((cd.shape[0], 1))])
    return g, cd


# ----------------------------------------------------------------------------- IES LM-63
def read_ies(path: str | Path) -> Photometry:
    text = Path(path).read_text(encoding="latin-1")
    lines = text.splitlines()
    meta = {}
    i = 0
    while i < len(lines) and not lines[i].strip().upper().startswith("TILT"):
        m = re.match(r"\[(\w+)\]\s*(.*)", lines[i].strip())
        if m:
            meta[m.group(1).upper()] = m.group(2)
        i += 1
    if i == len(lines):
        raise ValueError(f"{path}: TILT line not found — not an IES LM-63 file")
    tilt = lines[i].split("=", 1)[1].strip().upper()
    i += 1
    if tilt == "INCLUDE":
        i += 4                                          # skip the inline tilt table (geometry, n, angles, factors)
    nums = [float(x) for x in re.split(r"[\s,]+", " ".join(lines[i:])) if x]
    (n_lamps, lm_per_lamp, mult, n_v, n_h, ptype, units, width, length, _height) = nums[:10]
    _ballast, _future, watts = nums[10:13]
    n_v, n_h = int(n_v), int(n_h)
    k = 13
    v = np.array(nums[k:k + n_v]); k += n_v
    h = np.array(nums[k:k + n_h]); k += n_h
    cd = np.array(nums[k:k + n_v * n_h]).reshape(n_h, n_v) * mult
    if int(ptype) != 1:
        raise NotImplementedError("Only Type C photometry is supported (interior luminaires).")
    to_m = 0.3048 if int(units) == 1 else 1.0
    c, cd = _expand_c(h, cd)
    g, cd = _expand_g(v, cd)
    flux = n_lamps * lm_per_lamp if lm_per_lamp > 0 else 0.0
    ph = Photometry(c, g, cd, flux, meta.get("LUMINAIRE", Path(path).stem),
                    abs(width) * to_m, abs(length) * to_m, dict(meta, watts=watts))
    if lm_per_lamp <= 0:                                   # absolute photometry (LED): flux = integral
        ph.flux_lm = ph.total_flux()
    else:
        ph.meta["lor"] = ph.total_flux() / flux if flux else None
    return ph


# ----------------------------------------------------------------------------- EULUMDAT
def read_ldt(path: str | Path) -> Photometry:
    raw = Path(path).read_text(encoding="latin-1").splitlines()
    L = [s.strip() for s in raw]

    def f(i):
        return float(L[i].replace(",", "."))

    isym = int(f(2))
    mc, dc, ng, dg = int(f(3)), f(4), int(f(5)), f(6)
    name = L[8]
    length_mm, width_mm = f(12), f(13)
    lum_len, lum_wid = f(15), f(16)
    n_sets = int(f(25))
    k = 26
    lamps, flux = [], []
    for _ in range(n_sets):
        lamps.append(abs(f(k)))
        flux.append(f(k + 2))
        k += 6
    total_flux = sum(flux)
    k += 10                                           # direct ratios
    c_all = np.array([f(k + j) for j in range(mc)]); k += mc
    g = np.array([f(k + j) for j in range(ng)]); k += ng
    # planes actually stored in the file, by symmetry indicator
    mc1, mc2 = {0: (1, mc), 1: (1, 1), 2: (1, mc // 2 + 1),
                3: (mc // 4 + 1, 3 * mc // 4 + 1), 4: (1, mc // 4 + 1)}[isym]
    n_planes = mc2 - mc1 + 1
    vals = np.array([f(k + j) for j in range(n_planes * ng)]).reshape(n_planes, ng)
    cd = vals * total_flux / 1000.0                    # cd/klm → cd
    c, cd = _expand_c(c_all[mc1 - 1:mc2], cd)
    g, cd = _expand_g(g, cd)
    return Photometry(c, g, cd, total_flux, name,
                      (lum_wid or width_mm) / 1000.0, (lum_len or length_mm) / 1000.0,
                      {"isym": isym, "lamps": lamps})


def load(path: str | Path) -> Photometry:
    p = Path(path)
    if p.suffix.lower() == ".ies":
        return read_ies(p)
    if p.suffix.lower() == ".ldt":
        return read_ldt(p)
    raise ValueError(f"Unsupported photometric file: {p}")


_cache: dict = {}


def for_luminaire(spec: dict, resolve) -> Photometry:
    """Return the distribution for a catalogue entry, scaled to the catalogue lumens if given."""
    key = (spec.get("ies"), spec.get("distribution"), spec.get("lumens"))
    if key in _cache:
        return _cache[key]
    if spec.get("ies"):
        ph = load(resolve(spec["ies"]))
        if spec.get("lumens") and ph.flux_lm:
            ph = ph.scaled(float(spec["lumens"]))
    else:
        ph = generic(spec.get("distribution", "downlight"), float(spec.get("lumens", 0)))
    _cache[key] = ph
    return ph
