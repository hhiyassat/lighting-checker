"""Read requirement values straight from a code PDF at run time.

Supported table: Egyptian code 308 (2007), appendix M1, table M1 — required illuminance by space.
The PDF text uses Arabic presentation forms and bidi marks, so it is NFKC-normalised and cleaned first.
Text is extracted with `pdftotext -layout` (poppler); pypdf loses the word spacing of this file.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from functools import lru_cache
from pathlib import Path

_BIDI = re.compile("[‎‏‪-‮⁦-⁩]")


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _BIDI.sub("", s).replace("ـ", "")          # bidi marks, tatweel
    return re.sub(r"\s+", " ", s).strip()


def _page_text(pdf: Path, page: int) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext is required to read the code PDF (macOS: brew install poppler; "
                           "Debian/Ubuntu: apt install poppler-utils)")
    out = subprocess.run([exe, "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
                         capture_output=True, check=True)
    return out.stdout.decode("utf-8", "replace")


def _is_m1(text: str) -> bool:
    c = _clean(text)
    return "م1" in c and "شدة الإضاءة" in c and "غرفة نوم" in c and "المبانى السكنية" in c


def _find_m1_page(pdf: Path, hint: int | None) -> tuple[int, str]:
    if hint:
        t = _page_text(pdf, hint)
        if _is_m1(t):
            return hint, t
    exe = shutil.which("pdftotext")
    full = subprocess.run([exe, "-layout", str(pdf), "-"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
    for i, t in enumerate(full.split("\f"), 1):
        if _is_m1(t):
            return i, t
    raise ValueError(f"table M1 not found in {pdf}")


def parse_m1_group(text: str, group: str, next_groups=("المكاتب", "المكتبات", "المستشفيات", "الفنادق")) -> list[tuple[str, str, float]]:
    """Rows of one building group of table M1 as (heading, label, lux), in table order.

    A line ending with ':' is a heading (e.g. 'مطبخ :'); the rows under it carry it as their heading.
    The group's own name sits in the left column of one row and is removed from that row's label.
    The block ends where the first row of the next group starts (detected by the next group's name line)."""
    rows: list[tuple[str, str, float]] = []
    heading = ""
    started = False
    for raw in text.splitlines():
        line = _clean(raw)
        if not line or "PDF Created" in line:
            continue
        if not started:
            if "المكان" in line and "لوكس" in line:      # column header row
                started = True
            continue
        if any(g in line for g in next_groups) and not re.search(r"\d", line.replace(group, "")):
            break
        line = line.replace(group, "").strip()
        nums = re.findall(r"(?<![\d.])(\d{1,6})(?![\d.])", line)
        label = re.sub(r"\d+", "", line).replace("-", " ").strip()
        label = re.sub(r"\s+", " ", label)
        if not nums:
            if label.endswith(":"):
                heading = label.rstrip(":").strip()
            continue
        if label.endswith(":"):
            label = label.rstrip(":").strip()
        rows.append((heading, label, float(nums[0])))
    return rows


def lookup(rows, key: str) -> float:
    """key 'label' or 'heading/label'. A bare label must be unique within the group."""
    key = _clean(key)
    if "/" in key:
        h, lab = (k.strip() for k in key.split("/", 1))
        hits = [v for hh, ll, v in rows if ll == lab and hh == h]
    else:
        hits = [v for _, ll, v in rows if ll == key]
        if len(hits) > 1:
            raise ValueError(f"row '{key}' is ambiguous in the code table; write it as 'heading/{key}'")
    if not hits:
        known = ", ".join(f"{h}/{l}" if h else l for h, l, _ in rows)
        raise ValueError(f"row '{key}' not found in the code table (rows: {known})")
    return hits[0]


@lru_cache(maxsize=8)
def egypt_308_m1(pdf: str, group: str = "المبانى السكنية", page_hint: int | None = 343):
    """(page, rows) for one group of Egyptian code 308 table M1."""
    p = Path(pdf)
    if not p.exists():
        raise FileNotFoundError(f"code PDF not found: {p}")
    page, text = _find_m1_page(p, page_hint)
    rows = parse_m1_group(text, group)
    if not rows:
        raise ValueError(f"no rows for group '{group}' in table M1 (page {page})")
    return page, tuple(rows)


READERS = {"egypt_308_m1": egypt_308_m1}
