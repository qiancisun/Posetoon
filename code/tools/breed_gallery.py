import os
import csv
import json
import argparse

import numpy as np

from character_style import build_rig_styled, rest_world_transforms

OUT_DIR = "outputs"
TEMPLATES = os.path.join(OUT_DIR, "breed_templates.json")
CANVAS_BG = "#F0F0F0"


def _ribbon(rig, world):
    meta = rig["_spine_meta"]
    verts = np.array([world[f"spine{i}"]["pos"] for i in range(1, 5)]
                     + [world["neck"]["pos"]])
    w_up = np.asarray(meta.get("w_up_k", [meta["w_up"]] * 5), dtype=float)
    w_dn = np.asarray(meta.get("w_dn_k", [meta["w_dn"]] * 5), dtype=float)

    dirs = np.zeros_like(verts)
    dirs[:-1] = verts[1:] - verts[:-1]
    dirs[-1] = dirs[-2]
    n = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(n, 1e-9)
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)

    top = [verts[i] - normals[i] * w_up[i] for i in range(len(verts))]
    bot = [verts[i] + normals[i] * w_dn[i] for i in range(len(verts))]
    neck_ext = verts[-1] + dirs[-1] * (meta["L_seg"] * 0.15)
    rump_c = (top[0] + bot[0]) / 2 - dirs[0] * (meta["L_seg"] * 0.25)

    poly = [tuple(top[0])]
    poly += [tuple(p) for p in top]
    poly.append(tuple(neck_ext - normals[-1] * w_up[-1]))
    poly.append(tuple(neck_ext + normals[-1] * w_dn[-1]))
    poly += [tuple(p) for p in bot[::-1]]
    poly.append(tuple(rump_c))
    return poly


def draw_template(ax, rig, world, title):
    from matplotlib.patches import Polygon
    pal = rig["_style"]["palette"]
    line = pal.get("outline", "#222222")

    ax.add_patch(Polygon(_ribbon(rig, world), closed=True,
                         facecolor=pal.get("body", "#3D3D3D"),
                         edgecolor=line,
                         linewidth=rig["_style"].get("line_w_body", 1) * 0.9,
                         zorder=1))

    drawable = [(n, b) for n, b in rig.items()
                if not n.startswith("_") and b.get("mesh") is not None]
    for name, b in sorted(drawable, key=lambda kv: kv[1]["layer"]):
        wt = world[name]
        c, s = np.cos(wt["angle"]), np.sin(wt["angle"])
        px, py = wt["pos"]
        poly = [(x * c - y * s + px, x * s + y * c + py) for (x, y) in b["mesh"]]
        ax.add_patch(Polygon(poly, closed=True, facecolor=b["color"],
                             edgecolor=b["outline"] or line,
                             linewidth=b.get("line_w", 1) * 0.9,
                             zorder=2 + b["layer"] / 100.0))

    hw, hl = world["head"], rig["head"]["length"]
    hh = rig["_style"]["h_head"]
    c, s = np.cos(hw["angle"]), np.sin(hw["angle"])

    def hp(x, y):
        return (x * c - y * s + hw["pos"][0], x * s + y * c + hw["pos"][1])

    from matplotlib.patches import Ellipse, Circle
    lw_face = max(0.8, hh * 0.055)

    fs = min(hh, hl * 0.95)
    mz = hp(hl * 0.98, hh * 0.125)
    ax.add_patch(Ellipse(mz, fs * 1.08, fs * 0.73,
                         angle=np.degrees(hw["angle"]),
                         facecolor=pal.get("muzzle", "#5A5A5A"),
                         edgecolor=line, linewidth=lw_face, zorder=90))

    nr = max(2.5, fs * 0.20)
    nsp = hp(hl * 1.16, hh * 0.083)
    ax.add_patch(Ellipse(nsp, nr * 2, nr * 1.48,
                         angle=np.degrees(hw["angle"]),
                         facecolor="#1A1A1A", edgecolor="none", zorder=93))
    ax.add_patch(Circle(hp(hl * 1.16 - nr * 0.30, hh * 0.083 - nr * 0.34),
                        nr * 0.26, facecolor="#8A8A8A", edgecolor="none",
                        zorder=94))

    er = max(3.0, hh * 0.217)
    eye = hp(hl * 0.58, -hh * 0.33)
    ax.add_patch(Circle(eye, er, facecolor="white", edgecolor="#222222",
                        linewidth=0.9, zorder=96))
    ax.add_patch(Circle(eye, er * 0.55, facecolor="#1A1A1A", edgecolor="none",
                        zorder=97))
    ax.add_patch(Circle(hp(hl * 0.58 + er * 0.19, -hh * 0.33 - er * 0.29),
                        er * 0.26, facecolor="white", edgecolor="none",
                        zorder=98))
    ax.set_title(title, fontsize=8.5, linespacing=1.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coat", default=None,
                    help="sampled coat hex, e.g. '#E8E8E6'; omit for breed colours")
    ap.add_argument("--complexity", default="fine", choices=["fine", "coarse"])
    ap.add_argument("--retint", type=float, default=0.75)
    ap.add_argument("--out", default="breed_templates_gallery")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(TEMPLATES, "r") as f:
        data = json.load(f)
    ref = float(data.get("target_spine", 200.0))
    templates = data["templates"]

    try:
        from breed_markings import breed_palette
    except ImportError:
        breed_palette = None
        print("!! breed_markings.py not importable -- default greys only")

    try:
        import breeds as _breeds
        _stale = []
        for _n, _e in templates.items():
            _live = _breeds.appearance_of(_n)
            if _live is None:
                continue
            for _k, _v in _live.items():
                _cached = _e.get("appearance", {}).get(_k)
                if _cached != _v:
                    _stale.append(f"{_n}.{_k}: json {_cached} vs breeds.py {_v}")
        if _stale:
            print(f"!! {os.path.basename(TEMPLATES)} is OUT OF DATE with "
                  f"breeds.py ({len(_stale)} field(s) differ).")
            for _line in _stale[:8]:
                print(f"     {_line}")
            if len(_stale) > 8:
                print(f"     ... and {len(_stale) - 8} more")
            print("   Re-run 0_build_breed_templates.py, or this gallery shows "
                  "the OLD appearance.\n")
    except ImportError:
        pass

    def ratio(m):
        return 0.5 * ((m["hum"] + m["rad"]) + (m["fem"] + m["tib"])) / ref

    names = sorted(templates, key=lambda n: ratio(templates[n]["measure"]))

    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()

    table = []
    for ax, name in zip(axes, names):
        entry = templates[name]
        pal = None
        if breed_palette is not None:
            pal = breed_palette(name, sampled_hex=args.coat,
                                use_sampled=bool(args.coat), retint=args.retint,
                                lightness_match=1.0 if args.coat else None)
        _app = dict(entry["appearance"])
        try:
            import breeds as _bm
            _live = _bm.appearance_of(name)
            if _live:
                for _k, _v in _live.items():
                    if _app.get(_k) is None:
                        _app[_k] = _v
        except ImportError:
            pass
        rig = build_rig_styled(entry["measure"], _app, tier=name,
                               ref=ref, complexity=args.complexity, palette=pal)
        world = rest_world_transforms(rig)
        st = rig["_style"]
        r = ratio(entry["measure"])
        draw_template(ax, rig, world,
                      f"{name}\nratio {r:.3f}   build {st['build_factor']:.2f}   "
                      f"ear {st['ear_type']}\n"
                      f"head {rig['head']['length']:.0f}px   "
                      f"chest {rig['_spine_meta']['w_dn_k'][3]:.0f}px   "
                      f"tail {entry['appearance']['tail_len']:.2f}")
        table.append({
            "breed": name, "leg_to_spine": round(r, 3),
            "build_factor": round(st["build_factor"], 2),
            "neck_px": round(rig["neck"]["length"], 1),
            "head_px": round(rig["head"]["length"], 1),
            "chest_depth_px": round(rig["_spine_meta"]["w_dn_k"][3], 1),
            "ear_type": st["ear_type"],
            "ear_len": entry["appearance"]["ear_len"],
            "tail_len": entry["appearance"]["tail_len"],
            "tail_curl": entry["appearance"]["tail_curl"],
            "coat": entry["appearance"]["coat"],
            "body_hex": st["palette"]["body"],
            "outline_hex": st["palette"]["outline"],
        })

    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_facecolor(CANVAS_BG)
        ax.relim(); ax.autoscale_view()
    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.patch.set_facecolor(CANVAS_BG)
    sub = (f"retinted toward a sampled coat {args.coat}" if args.coat
           else "breed-typical colours")
    plt.suptitle(f"Breed templates: {len(names)} characters, rest pose, "
                 f"{args.complexity} — {sub}\n"
                 f"same bone hierarchy and same solver throughout; "
                 f"proportions, part geometry, ears, tail and markings differ",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, args.out + ".png")
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor=CANVAS_BG)
    print(f"Saved {png}")

    csv_path = os.path.join(OUT_DIR, args.out + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"Saved {csv_path}")

    print()
    _h = (f"{'breed':16} {'ratio':>6} {'build':>6} {'neck':>6} {'head':>6} "
          f"{'chest':>6} {'ear':11} {'earLen':>7} {'tail':>6} {'curl':>6} "
          f"{'coat':7} body")
    print(_h)
    print("-" * len(_h))
    for row in table:
        print(f"{row['breed']:16} {row['leg_to_spine']:6.3f} "
              f"{row['build_factor']:6.2f} {row['neck_px']:6.1f} "
              f"{row['head_px']:6.1f} {row['chest_depth_px']:6.1f} "
              f"{row['ear_type']:11} {row['ear_len']:7.2f} "
              f"{row['tail_len']:6.2f} {row['tail_curl']:6.2f} "
              f"{row['coat']:7} {row['body_hex']}")

    print()
    for key in ("leg_to_spine", "build_factor", "neck_px", "head_px",
                "chest_depth_px", "ear_len", "tail_len", "tail_curl"):
        v = [row[key] for row in table]
        print(f"  {key:15} {min(v):7.3f} .. {max(v):7.3f}   "
              f"x{max(v)/max(min(v), 1e-9):.2f}")
    ears = sorted({row["ear_type"] for row in table})
    bodies = {row["body_hex"] for row in table}
    print(f"  {'ear_type':15} {', '.join(ears)}")
    print(f"  {'distinct body colours':15} {len(bodies)} of {len(table)}")


if __name__ == "__main__":
    main()
