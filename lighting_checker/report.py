"""Self-contained HTML report: summary, per-floor plan with verdict colours, per-room heat-maps."""
from __future__ import annotations

import base64
import html
import io
from collections import Counter, defaultdict
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VERDICT_COL = {"pass": "#2e9d5b", "fail": "#d1453b", "no rule": "#8a8f98", "not assessed": "#c9a227", "n/a": "#8a8f98"}
VERDICT_AR = {"pass": "مطابق", "fail": "غير مطابق", "no rule": "لا قاعدة", "not assessed": "لم يُقيَّم", "n/a": "—"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _floor_plan(results, lums, u, title):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for r in results:
        room = r.room
        h, w = room.mask.shape
        rgba = np.zeros((h, w, 4))
        col = matplotlib.colors.to_rgb(VERDICT_COL.get(r.verdict, "#888"))
        rgba[room.mask] = [*col, 0.55]
        ext = (room.ox * u, (room.ox + w * room.px) * u, room.oy * u, (room.oy + h * room.px) * u)
        ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest")
        ys, xs = np.nonzero(room.mask)
        label = (room.label or room.type) + (f"\n{r.e_avg:.0f} lx" if r.e_avg is not None else "")
        ax.text((room.ox + (xs.mean() + .5) * room.px) * u, (room.oy + (ys.mean() + .5) * room.px) * u,
                label, ha="center", va="center", fontsize=6.5)
    for lm in lums:
        ax.plot(lm.x * u, lm.y * u, marker="o" if lm.room_id else "x", ms=3 if lm.room_id else 6,
                color="#222" if lm.room_id else "#d1453b", lw=0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("m")
    ax.autoscale()
    return _png(fig)


def _heatmap(r, u, vmax):
    xs, ys, E = r.grid
    room = r.room
    h, w = room.mask.shape
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    outline = np.zeros((h, w, 4))
    outline[room.mask] = [0, 0, 0, 0.07]
    ax.imshow(outline, origin="lower", interpolation="nearest",
              extent=(room.ox * u, (room.ox + w * room.px) * u, room.oy * u, (room.oy + h * room.px) * u))
    sc = ax.scatter(xs, ys, c=E, cmap="inferno", vmin=0, vmax=vmax, s=max(8, 900 / max(len(xs), 1)), marker="s")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="lx")
    for lm in r.room.luminaires:
        ax.plot(lm.x * u, lm.y * u, "c^", ms=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return _png(fig)


def _fmt(v, nd=0):
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def write_html(path, drawing, results, emergency, proj, source_name):
    u = drawing.u
    by_floor = defaultdict(list)
    for r in results:
        by_floor[r.room.floor].append(r)
    cnt = Counter(r.verdict for r in results)

    css = """
    :root{--bg:#fff;--fg:#1d2330;--mut:#646b78;--line:#e3e6eb;--card:#f7f8fa}
    @media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e8eaee;--mut:#9aa1ad;--line:#2a2f37;--card:#1b1f26}}
    body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Tahoma,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}
    h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
    .mut{color:var(--mut)} .chips span{display:inline-block;padding:3px 10px;border-radius:12px;margin:4px 6px 0 0;color:#fff;font-size:13px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}
    th,td{border-bottom:1px solid var(--line);padding:5px 6px;text-align:start;vertical-align:top}
    th{background:var(--card);position:sticky;top:0}
    td.num{text-align:end;font-variant-numeric:tabular-nums}
    .v{padding:1px 8px;border-radius:10px;color:#fff;white-space:nowrap}
    .grid{display:flex;flex-wrap:wrap;gap:12px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;width:360px}
    .card img{width:100%} img.plan{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
    .warn{background:var(--card);border-inline-start:3px solid #c9a227;padding:8px 12px;margin:8px 0}
    .note{font-size:12px;color:var(--mut)}
    """
    out = [f"<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>تقرير الإنارة</title><style>{css}</style></head><body>"]
    out.append(f"<h1>تقرير التحقق من الإنارة — {html.escape(source_name)}</h1>")
    out.append(f"<div class='mut'>{date.today().isoformat()} · مرجع المتطلبات: {html.escape(proj.rules.source or '—')}"
               f" · معامل الصيانة MF = {proj.maintenance_factor}</div>")
    out.append("<div class='chips'>" + "".join(
        f"<span style='background:{VERDICT_COL[k]}'>{VERDICT_AR[k]}: {cnt.get(k, 0)}</span>"
        for k in ("pass", "fail", "not assessed", "no rule")) + "</div>")

    if proj.rules.provisional:
        prov = ", ".join(proj.rules.provisional_types)
        out.append("<div class='warn' style='border-color:#d1453b'><b>قيم المتطلبات مؤقتة:</b> "
                   f"لم تُنقل بعد قيم الكود المعتمد لأنواع الغرف التالية: {html.escape(prov)}. "
                   "الحكم على هذه الغرف مبنيّ على قيم أمثلة، فلا يُعتمد قبل إدخال قيم الكود في ملف المتطلبات.</div>")
    out.append("<div class='warn'><b>حدود هذا الحساب:</b> الإضاءة المباشرة محسوبة نقطةً نقطة؛ "
               "الانعكاسات مقدَّرة بطريقة التدفق المنقسم (split-flux) وتضاف بانتظام، فالنتائج تقريبية مقارنةً "
               "بـ DIALux (radiosity). حين لا يوجد ملف IES/LDT يُستعمل توزيع عام (لامبرتي أو كروي) بلومنات الكتالوج. "
               "UGR محسوب بمعادلة CIE 117 مع مؤشر Guth لأربعة مراقبين، وهو تقديري.</div>")

    if drawing.warnings or drawing.unassigned:
        out.append("<h2>تنبيهات القراءة</h2><ul>")
        for w in drawing.warnings:
            out.append(f"<li>{html.escape(w)}</li>")
        ua = Counter((lm.sheet, lm.block) for lm in drawing.unassigned if lm.sheet)
        for (sheet, block), n in ua.most_common():
            out.append(f"<li>{n} × <code>{html.escape(block)}</code> في «{html.escape(sheet)}» خارج أي غرفة "
                       f"(خارجي، أو مساحة مفتوحة لم تُغلق حدودها).</li>")
        out.append("</ul>")

    # summary table
    out.append("<h2>ملخّص الغرف</h2><div style='overflow-x:auto'><table><tr>"
               "<th>الطابق</th><th>الغرفة</th><th>النوع</th><th>المصدر</th><th>المساحة م²</th><th>الوحدات</th>"
               "<th>لومن</th><th>W/م²</th><th>E طريقة اللومن</th><th>Em</th><th>Emin</th><th>Uo</th><th>UGR</th><th>d/h</th>"
               "<th>المطلوب Em / Uo</th><th>الحكم</th><th>العجز</th><th>ملاحظات</th></tr>")
    for floor, rs in by_floor.items():
        for r in rs:
            req = r.requirement or {}
            reqs = f"{_fmt(req.get('em_lx'))} / {_fmt(req.get('uo'), 2)}" if req else "—"
            if req and r.room.type in proj.rules.provisional_types:
                reqs += " <span class='note'>(مؤقت)</span>"
            elif req and req.get("clause"):
                reqs += f" <span class='note'>{html.escape(req['clause'])}</span>"
            failed = [k for k, (_, _, okk) in r.checks.items() if not okk and k != "Em"]
            deficit = "<br>".join(([f"{r.deficit_lx:.0f} lx (~{r.extra_luminaires} وحدة إضافية)"] if r.deficit_lx else [])
                                  + [f"{k} خارج الحد" for k in failed])
            out.append(
                f"<tr><td>{html.escape(floor)}</td><td>{html.escape(r.room.label or '—')}<div class='note'>{html.escape(r.room.id)}</div></td>"
                f"<td>{r.room.type}</td><td>{r.room.source}</td><td class='num'>{r.area_m2:.1f}</td>"
                f"<td class='num'>{r.n_lum}</td><td class='num'>{r.flux_lm:.0f}</td><td class='num'>{_fmt(r.lpd_w_m2, 1)}</td>"
                f"<td class='num'>{_fmt(r.e_lumen)}</td><td class='num'>{_fmt(r.e_avg)}</td><td class='num'>{_fmt(r.e_min)}</td>"
                f"<td class='num'>{_fmt(r.uo, 2)}</td><td class='num'>{_fmt(r.ugr, 1)}</td><td class='num'>{_fmt(r.spacing and r.spacing[2], 2)}</td><td>{reqs}</td>"
                f"<td><span class='v' style='background:{VERDICT_COL.get(r.verdict)}'>{VERDICT_AR.get(r.verdict, r.verdict)}</span></td>"
                f"<td>{deficit}</td><td class='note'>{'<br>'.join(html.escape(n) for n in r.notes)}</td></tr>")
    out.append("</table></div>")

    # per floor
    for floor, rs in by_floor.items():
        sheet = rs[0].room.sheet
        lums = [lm for lm in drawing.luminaires if lm.sheet == sheet]
        out.append(f"<h2>{html.escape(floor)}</h2>")
        out.append(f"<img class='plan' alt='plan' src='{_floor_plan(rs, lums, u, sheet)}'>")
        vmax = max([r.e_max for r in rs if r.e_max] + [1])
        out.append("<div class='grid'>")
        for r in rs:
            if r.grid is None:
                continue
            out.append(f"<div class='card'><b>{html.escape(r.room.label or r.room.type)}</b> "
                       f"<span class='v' style='background:{VERDICT_COL.get(r.verdict)}'>{VERDICT_AR.get(r.verdict)}</span>"
                       f"<div class='note'>Em {r.e_avg:.0f} lx · Emin {r.e_min:.0f} · Uo {r.uo:.2f} · "
                       f"شبكة {r.grid_p_m:.2f} م · ارتفاع التركيب فوق مستوى العمل {r.mount_h_m:.2f} م</div>"
                       f"<img alt='heatmap' src='{_heatmap(r, u, vmax)}'></div>")
        out.append("</div>")

    # emergency
    out.append("<h2>إنارة الطوارئ (EN 1838)</h2>")
    if not emergency.present:
        out.append(f"<p>{html.escape(' '.join(emergency.notes))}</p>")
    else:
        out.append(f"<p>{emergency.n_luminaires} وحدة طوارئ.</p><table><tr><th>المسار</th><th>الطول م</th><th>Emin</th><th>Emax</th><th>الحكم</th></tr>")
        for k, (L, emin, emax, ok) in enumerate(emergency.routes, 1):
            out.append(f"<tr><td>{k}</td><td>{L:.1f}</td><td>{emin:.2f}</td><td>{emax:.2f}</td><td>{'مطابق' if ok else 'غير مطابق'}</td></tr>")
        out.append("</table><table><tr><th>منطقة مكافحة الذعر</th><th>المساحة</th><th>Emin</th><th>الحكم</th></tr>")
        for rid, A, emin, ok in emergency.anti_panic:
            out.append(f"<tr><td>{html.escape(rid)}</td><td>{A:.0f}</td><td>{emin:.2f}</td><td>{'مطابق' if ok else 'غير مطابق'}</td></tr>")
        out.append("</table>")
        for n in emergency.notes:
            out.append(f"<p class='note'>{html.escape(n)}</p>")
    out.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
