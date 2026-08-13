import os
import json
import numpy as np

OUT_DIR = "outputs"
TEMPLATES_V2 = os.path.join(OUT_DIR, "dog_templates_v2.json")

EAR_DROOP = {"floppy": 1.50, "semi_erect": 0.45, "erect": -0.95}

def tail_curl_rel(curl):
    return 0.25 + 1.10 * float(curl)

BUILD_FACTOR = {
    "Basset Hound": 1.25,
    "Cardigan Corgi": 1.20,
    "Pug": 1.22,
    "Pekingese": 1.18,
    "Beagle": 1.08,
    "Labrador": 1.05,
    "German Shepherd": 1.00,
    "Great Dane": 0.90,
    "Whippet": 0.82,
    "small": 1.15, "medium": 1.00, "large": 0.92,
}


def build_factor_for(name, measure=None, ref=200.0):
    try:
        from breeds import build_of, resolve
        b = build_of(resolve(name) or name)
        if b is not None:
            return b
    except ImportError:
        pass
    if name in BUILD_FACTOR:
        return BUILD_FACTOR[name]
    if measure is None:
        return 1.0
    ratio = 0.5 * ((measure["hum"] + measure["rad"])
                    + (measure["fem"] + measure["tib"])) / ref
    t = (ratio - 0.37) / (0.66 - 0.37)
    return float(np.clip(1.22 + t * (0.85 - 1.22), 0.85, 1.22))

NECK_FRACTION = 0.32

DEFAULT_PALETTE = {
    "body": "#3D3D3D", "neck": "#424242", "head": "#4A4A4A",
    "ear_upper": "#363636", "ear_lower": "#2E2E2E", "tail": "#2A2A2A",
    "near_upper": "#444444", "near_mid": "#3C3C3C", "near_lower": "#2A2A2A",
    "far_upper": "#666666", "far_mid": "#5E5E5E", "far_lower": "#4E4E4E",
    "outline": "#222222", "outline_dark": "#1A1A1A",
    "muzzle": "#5A5A5A",
}


def build_rig_styled(measure, appearance, tier="medium", ref=200.0,
                      complexity="fine", palette=None):
    pal = dict(DEFAULT_PALETTE)
    if palette:
        pal.update({k: v for k, v in palette.items() if v})

    fine = (complexity == "fine")
    bf = build_factor_for(tier, measure, ref)

    L_low, L_up = measure["lower"], measure["upper"]
    L_spine_total = L_low + L_up
    L_seg = (L_spine_total / 4.0) * 0.82

    head_total = float(measure.get("head") or appearance["muzzle_len"] * ref)
    L_neck = float(np.clip(head_total * NECK_FRACTION, ref * 0.10, ref * 0.22))
    L_head = float(np.clip(head_total - L_neck, ref * 0.14, ref * 0.42))

    h_head = ref * 0.115 * bf

    L_hum, L_rad = measure["hum"], measure["rad"]
    L_fem, L_tib = measure["fem"], measure["tib"]
    L_paw_f, L_paw_h = ref * 0.052 * bf, ref * 0.056 * bf

    ear_total = appearance["ear_len"] * ref
    L_e1, L_e2 = ear_total * 0.48, ear_total * 0.52
    ear_w = appearance["ear_width"] * ref

    tail_total = appearance["tail_len"] * ref
    L_t1, L_t2 = tail_total * 0.55, tail_total * 0.45

    COARSE_W = ref * 0.038

    def taper(L, w0, w1, back=0.0):
        if fine:
            return [(-back, -w0), (L, -w1), (L, w1), (-back, w0)]
        w = COARSE_W
        return [(0, -w), (L, -w), (L, w), (0, w)]

    def paw_shape(L, w):
        if fine:
            return [(-L * 0.34, -w * 0.95), (L * 0.70, -w * 0.85), (L, -w * 0.45),
                    (L, w * 0.55), (L * 0.55, w * 0.95), (-L * 0.34, w * 0.90)]
        return None

    rig = {}

    def bone(name, parent, offset, length, mesh, color, outline=None, layer=0):
        rig[name] = {"parent": parent, "offset": np.array(offset, dtype=float),
                     "length": length, "mesh": mesh, "color": color,
                     "outline": outline, "layer": layer}

    bone("root", None, (0, 0), 0.0, None, None, layer=0)

    for i in range(1, 5):
        parent = "root" if i == 1 else f"spine{i-1}"
        off = (0, 0) if i == 1 else (L_seg, 0)
        bone(f"spine{i}", parent, off, L_seg, None, color="#3D3D3D", layer=30 + i)

    bone("neck", "spine4", (L_seg, 0), L_neck,
         mesh=[(-ref * 0.03, -ref * 0.082 * bf), (L_neck, -ref * 0.090 * bf),
               (L_neck, ref * 0.078 * bf), (-ref * 0.03, ref * 0.074 * bf)],
         color=pal["neck"], layer=36)

    if fine:
        head_mesh = [(-ref * 0.02, -h_head * 1.00), (L_head * 0.30, -h_head * 1.21),
                     (L_head * 0.70, -h_head * 1.13), (L_head * 0.98, -h_head * 0.67),
                     (L_head * 1.14, -h_head * 0.04), (L_head * 1.08, h_head * 0.50),
                     (L_head * 0.66, h_head * 0.92), (L_head * 0.26, h_head * 0.83),
                     (-ref * 0.02, h_head * 0.75)]
    else:
        head_mesh = [(0, -h_head * 0.62), (L_head * 1.05, -h_head * 0.62),
                     (L_head * 1.05, h_head * 0.62), (0, h_head * 0.62)]
    bone("head", "neck", (L_neck, 0), L_head, head_mesh,
         color=pal["head"], outline=pal["outline"], layer=37)

    ear_anchor = (L_head * 0.28, -h_head * 0.78)
    if fine:
        bone("ear_upper", "head", ear_anchor, L_e1,
             taper(L_e1, ear_w * 0.85, ear_w), color=pal["ear_upper"],
             outline=pal["outline"], layer=38)
        bone("ear_lower", "ear_upper", (L_e1, 0), L_e2,
             taper(L_e2, ear_w, ear_w * 0.55, back=ref * 0.010),
             color=pal["ear_lower"], outline=pal["outline_dark"], layer=39)
    else:
        bone("ear_upper", "head", ear_anchor, L_e1 + L_e2,
             taper(L_e1 + L_e2, ear_w, ear_w * 0.7), color=pal["ear_upper"],
             outline=pal["outline"], layer=38)

    TAIL_ROOT_BACK = 0.03
    bone("tail_1", "spine1", (-ref * TAIL_ROOT_BACK, ref * 0.02), L_t1,
         taper(L_t1, ref * 0.028 * bf, ref * 0.020 * bf, back=ref * 0.075),
         color=pal["tail"], layer=20)
    bone("tail_2", "tail_1", (L_t1, 0), L_t2,
         taper(L_t2, ref * 0.020 * bf, ref * 0.010 * bf, back=ref * 0.010),
         color=pal["tail"], layer=21)

    bone("shoulder", "spine4", (0, 0), 0.0, None, None, layer=0)
    bone("hip", "spine1", (0, 0), 0.0, None, None, layer=0)

    for side, base, cu, cm, cl in [
            ("L", 40, pal["near_upper"], pal["near_mid"], pal["near_lower"]),
            ("R", 10, pal["far_upper"], pal["far_mid"], pal["far_lower"])]:
        LIMB_ROOT_SPREAD = 0.022
        _lat = ref * LIMB_ROOT_SPREAD * (1.0 if side == "L" else -1.0)
        bone(f"humerus_{side}", "shoulder", (_lat, _lat * 0.35), L_hum,
             taper(L_hum, ref * 0.044 * bf, ref * 0.028 * bf, back=ref * 0.045),
             color=cu, layer=base + 1)
        bone(f"radius_{side}", f"humerus_{side}", (L_hum, 0), L_rad,
             taper(L_rad, ref * 0.028 * bf, ref * 0.018 * bf, back=ref * 0.026),
             color=cm, layer=base + 2)
        bone(f"paw_front_{side}", f"radius_{side}", (L_rad, 0), L_paw_f,
             paw_shape(L_paw_f, ref * 0.021 * bf), color=cl,
             outline=pal["outline_dark"], layer=base + 3)
        bone(f"femur_{side}", "hip", (_lat, _lat * 0.35), L_fem,
             taper(L_fem, ref * 0.048 * bf, ref * 0.029 * bf, back=ref * 0.045),
             color=cu, layer=base + 4)
        bone(f"tibia_{side}", f"femur_{side}", (L_fem, 0), L_tib,
             taper(L_tib, ref * 0.029 * bf, ref * 0.018 * bf, back=ref * 0.027),
             color=cm, layer=base + 5)
        bone(f"paw_hind_{side}", f"tibia_{side}", (L_tib, 0), L_paw_h,
             paw_shape(L_paw_h, ref * 0.021 * bf), color=cl,
             outline=pal["outline_dark"], layer=base + 6)

    rig["_spine_meta"] = {"L_seg": L_seg,
                          "w_up": ref * 0.125 * bf, "w_dn": ref * 0.150 * bf,
                          "w_up_end": ref * 0.075 * bf, "w_dn_end": ref * 0.090 * bf}

    rig["_style"] = {
        "tier": tier,
        "complexity": complexity,
        "ear_droop": EAR_DROOP[appearance["ear_type"]],
        "tail_curl_rel": tail_curl_rel(appearance["tail_curl"]),
        "build_factor": bf,
        "h_head": h_head,
        "L_neck": L_neck,
        "ear_type": appearance["ear_type"],
        "coat": appearance["coat"],
        "palette": pal,
    }
    return rig


REST_WORLD = {
    "root": 0.0, "spine1": 0.0, "spine2": 0.0, "spine3": 0.0, "spine4": 0.0,
    "shoulder": 0.0, "hip": 0.0,
    "neck": -0.42, "head": -0.10,
    "tail_1": -0.75,
    "humerus_L": 1.75, "radius_L": 1.95, "paw_front_L": 2.30,
    "femur_L": 1.35, "tibia_L": 1.85, "paw_hind_L": 2.30,
    "humerus_R": 1.75, "radius_R": 1.95, "paw_front_R": 2.30,
    "femur_R": 1.35, "tibia_R": 1.85, "paw_hind_R": 2.30,
}


def rest_world_transforms(rig):
    style = rig["_style"]
    world = {}

    def resolve(name):
        if name in world:
            return world[name]
        b = rig[name]
        if b["parent"] is None:
            world[name] = {"pos": np.zeros(2), "angle": REST_WORLD.get(name, 0.0)}
            return world[name]
        pw = resolve(b["parent"])
        c, s = np.cos(pw["angle"]), np.sin(pw["angle"])
        off = b["offset"]
        pos = pw["pos"] + np.array([off[0] * c - off[1] * s,
                                     off[0] * s + off[1] * c])
        if name == "ear_upper":
            angle = resolve("head")["angle"] + style["ear_droop"]
        elif name == "ear_lower":
            angle = resolve("ear_upper")["angle"] + 0.25
        elif name == "tail_2":
            angle = resolve("tail_1")["angle"] + style["tail_curl_rel"]
        else:
            angle = REST_WORLD.get(name, pw["angle"])
        world[name] = {"pos": pos, "angle": angle}
        return world[name]

    for name in rig:
        if not name.startswith("_"):
            resolve(name)
    return world


def draw_rest(ax, rig, world, title):
    from matplotlib.patches import Polygon

    drawable = [(n, b) for n, b in rig.items()
                if not n.startswith("_") and b.get("mesh") is not None]
    for name, b in sorted(drawable, key=lambda kv: kv[1]["layer"]):
        wt = world[name]
        c, s = np.cos(wt["angle"]), np.sin(wt["angle"])
        px, py = wt["pos"]
        poly = [(x * c - y * s + px, x * s + y * c + py) for (x, y) in b["mesh"]]
        ax.add_patch(Polygon(poly, closed=True, facecolor=b["color"],
                              edgecolor=b["outline"] or "#222222", linewidth=1.0))

    spine_pts = [world[f"spine{i}"]["pos"] for i in range(1, 5)]
    spine_pts.append(world["neck"]["pos"])
    sp = np.array(spine_pts)
    ax.plot(sp[:, 0], sp[:, 1], "-", color="#3D3D3D", linewidth=14,
            solid_capstyle="round", zorder=0)

    ax.set_title(title, fontsize=10)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(TEMPLATES_V2, "r") as f:
        data = json.load(f)
    ref = data["target_spine"]

    tiers = ["small", "medium", "large"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, tier in enumerate(tiers):
        measure = data["templates"][tier]["measure"]
        appearance = data["templates"][tier]["appearance"]
        for row, complexity in enumerate(["fine", "coarse"]):
            rig = build_rig_styled(measure, appearance, tier=tier,
                                    ref=ref, complexity=complexity)
            world = rest_world_transforms(rig)
            draw_rest(axes[row, col], rig, world,
                      f"{tier.upper()} / {complexity}\n"
                      f"ear={appearance['ear_type']}  "
                      f"head={rig['head']['length']:.0f}px  "
                      f"tail={appearance['tail_len']:.2f}")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.relim()
        ax.autoscale_view()

    plt.suptitle("Rest-pose part geometry: 3 tiers x 2 complexity levels\n"
                 "same bone hierarchy and same solver throughout; "
                 "only part meshes and rest angles differ", fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "character_style_preview.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Saved {out_png}")

    for tier in tiers:
        measure = data["templates"][tier]["measure"]
        appearance = data["templates"][tier]["appearance"]
        rig = build_rig_styled(measure, appearance, tier=tier, ref=ref)
        st = rig["_style"]
        print(f"[{tier:6}] neck={rig['neck']['length']:5.1f}px  "
              f"head={rig['head']['length']:6.1f}px  h_head={st['h_head']:5.1f}px  "
              f"ear={st['ear_type']:10} droop={st['ear_droop']:+.2f}rad  "
              f"tail_curl_rel={st['tail_curl_rel']:.2f}  bf={st['build_factor']:.2f}")


if __name__ == "__main__":
    main()
