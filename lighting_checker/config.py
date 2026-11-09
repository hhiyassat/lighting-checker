"""Load the project configuration, luminaire catalogue and rules table."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _strip_comments(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_comments(v) for k, v in d.items() if not k.startswith("_")}
    return d


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return _strip_comments(json.load(f))


@dataclass
class Rules:
    types: dict
    aliases: dict
    emergency: dict
    source: str = ""
    provisional_types: tuple = ()
    spacing: dict = field(default_factory=dict)     # {"max_ratio": 1.5, "clause": "..."}; empty = not checked

    @property
    def provisional(self) -> bool:
        return bool(self.provisional_types)

    def classify(self, label: str | None) -> str:
        """Map a free-text room label (any language) to a rule type, or 'unknown'."""
        if not label:
            return "unknown"
        s = re.sub(r"\s+", " ", label.strip().lower())
        for t, names in self.aliases.items():           # exact match first
            if s in (n.lower() for n in names):
                return t
        best, best_len = "unknown", 0
        for t, names in self.aliases.items():           # then the longest alias contained in the label
            for n in names:
                n = n.lower()
                if re.search(r"(^|\W)" + re.escape(n) + r"($|\W)", s) and len(n) > best_len:
                    best, best_len = t, len(n)
        return best

    def requirement(self, room_type: str) -> dict | None:
        return self.types.get(room_type)


@dataclass
class Project:
    root: Path
    raw: dict
    catalog: dict
    rules: Rules
    sheets: dict = field(default_factory=dict)
    layers: dict = field(default_factory=dict)
    room_detection: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    surfaces: dict = field(default_factory=dict)
    maintenance_factor: float = 0.8
    units_to_m: float | None = None

    def resolve(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)


def _values_from_pdf(r: dict, base: Path) -> None:
    """Fill em_lx of every type that names a "pdf_row" from the code PDF. A value also written in the JSON
    must equal the PDF's, so a typo or a different edition is caught at run time."""
    from .codebook import READERS, lookup
    spec = r["code_pdf"]
    reader = READERS[spec["table"]]
    pdf = (base / spec["path"]).resolve()
    page, rows = reader(str(pdf), spec.get("group", "المبانى السكنية"), spec.get("page"))
    for t, req in r.get("types", {}).items():
        key = req.get("pdf_row")
        if not key:
            continue
        v = lookup(rows, key)
        if req.get("em_lx") is not None and float(req["em_lx"]) != v:
            raise ValueError(f"rules {t}: em_lx {req['em_lx']} differs from the code PDF ({key} = {v:g} lx, page {page})")
        req["em_lx"] = v
        req["clause"] = f"{req.get('clause') or ''} (PDF ص {page}: {key} = {v:g} lx)".strip()
    r["source"] = f"{r.get('source', '')} — القيم مقروءة من {pdf.name} ص {page}"


def load_rules(path: Path) -> Rules:
    """Load a rules table. A type whose em_lx is null (value not yet taken from the code) falls back to the
    file named in "fallback"; such types, and every type of a file with "provisional": true, are provisional."""
    r = _load(path)
    if r.get("code_pdf"):
        _values_from_pdf(r, path.parent)
    types, prov = {}, set()
    fb = None
    if r.get("fallback"):
        fb = load_rules(path.parent / r["fallback"])
    for t, req in r.get("types", {}).items():
        if req.get("em_lx") is None:
            if fb and t in fb.types:
                types[t] = {**fb.types[t], "_source": fb.source}
                prov.add(t)
            continue
        types[t] = {**req, "_source": r.get("source", "")}
        if r.get("provisional"):
            prov.add(t)
    if fb:
        for t, req in fb.types.items():
            if t not in types and t not in r.get("types", {}):
                types[t] = {**req, "_source": fb.source}
                prov.add(t)
        prov |= set(fb.provisional_types) & {t for t in types if types[t]["_source"] == fb.source}
    aliases = {**(fb.aliases if fb else {}), **r.get("aliases", {})}
    emergency = {**(fb.emergency if fb else {}), **r.get("emergency", {})}
    return Rules(types=types, aliases=aliases, emergency=emergency, source=r.get("source", ""),
                 provisional_types=tuple(sorted(prov)), spacing=r.get("spacing") or {})


def load_project(path: str | Path) -> Project:
    path = Path(path)
    raw = _load(path)
    root = path.parent
    catalog = _load(root / raw["luminaire_catalog"])
    rules = load_rules(root / raw["rules"])
    return Project(
        root=root, raw=raw, catalog=catalog, rules=rules,
        sheets=raw.get("sheets", {}), layers=raw.get("layers", {}),
        room_detection=raw.get("room_detection", {}),
        geometry=raw.get("geometry_defaults", {}),
        surfaces=raw.get("surfaces", {}),
        maintenance_factor=float(raw.get("maintenance_factor", 0.8)),
        units_to_m=raw.get("units_to_m"),
    )
