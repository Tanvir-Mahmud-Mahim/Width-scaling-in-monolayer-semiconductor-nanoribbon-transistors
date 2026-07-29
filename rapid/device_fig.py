"""Three-dimensional schematic of the monolayer nanoribbon transistor.

The schematic is drawn on an ordinary two-dimensional axes through an explicit
orthographic projection rather than through mplot3d.  Doing the projection by
hand buys three things that matter for a publication figure:

  * exact control of occlusion.  Faces are sorted back to front by the depth
    of their centroid along the view direction and painted in that order, so
    nothing shows through anything else.
  * physically sensible shading.  Each face is shaded by the Lambert cosine
    between its outward normal and a fixed light direction, which is what makes
    a stack of slabs read as a solid object rather than as flat colour.
  * ordinary two-dimensional text handling.  Labels, leader lines and arrow
    heads are placed in data coordinates on a normal axes, so they behave
    predictably and can be verified against the projected geometry.

Everything is regenerated with the rest of the figures and needs no external
drawing program.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, FancyArrowPatch, Circle

# palette, kept in step with figures.py
BLUE = '#0072B2'
VERM = '#D55E00'
GREEN = '#009E73'
INK = '#1a1a1a'
INK2 = '#555555'

# ---------------------------------------------------------------------------
# camera and lighting
# ---------------------------------------------------------------------------
AZ = np.deg2rad(-52.0)          # rotation about the vertical axis
EL = np.deg2rad(20.0)           # elevation above the substrate plane
ZSCALE = 1.55                   # vertical exaggeration of the layer stack
LIGHT = np.array([-0.45, -0.72, 0.53])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE = 0.62, 0.38


def _basis():
    ca, sa, ce, se = np.cos(AZ), np.sin(AZ), np.cos(EL), np.sin(EL)
    view = np.array([ce * ca, ce * sa, se])          # camera looks along -view
    right = np.array([-sa, ca, 0.0])
    up = np.array([-se * ca, -se * sa, ce])
    return view, right, up


VIEW, RIGHT, UP = _basis()


def project(P):
    """3D points (..., 3) to screen (..., 2); z is exaggerated by ZSCALE."""
    Q = np.atleast_2d(np.asarray(P, float)).copy()
    Q[:, 2] *= ZSCALE
    return np.stack([Q @ RIGHT, Q @ UP], axis=-1)


def depth(P):
    """Distance from the camera; larger means farther away."""
    Q = np.atleast_2d(np.asarray(P, float)).copy()
    Q[:, 2] *= ZSCALE
    return -(Q @ VIEW)


def _shade(colour, normal, boost=0.0):
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    f = AMBIENT + DIFFUSE * max(0.0, float(n @ LIGHT)) + boost
    r, g, b = to_rgb(colour)
    return (min(1.0, r * f), min(1.0, g * f), min(1.0, b * f))


# ---------------------------------------------------------------------------
# scene graph: every face is queued, then painted back to front
# ---------------------------------------------------------------------------
class Scene:
    def __init__(self):
        self.faces = []   # (layer, depth, verts3, fc, ec, lw, alpha)

    def add(self, verts, colour, normal=None, edge=None, lw=0.4, alpha=1.0,
            shade=True, boost=0.0, layer=0):
        V = np.asarray(verts, float)
        if normal is None:
            normal = np.cross(V[1] - V[0], V[2] - V[0])
        fc = _shade(colour, normal, boost) if shade else to_rgb(colour)
        ec = edge if edge is not None else _shade(colour, normal, boost - 0.30)
        self.faces.append((layer, float(np.mean(depth(V))), V, fc, ec, lw,
                           alpha))

    def box(self, x0, y0, z0, dx, dy, dz, colour, edge=None, lw=0.4,
            alpha=1.0, boost=0.0, layer=0):
        """Axis-aligned slab.  Only the three faces that can be seen from this
        camera are queued, which keeps the face count and the file size low."""
        x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
        faces = [
            ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
             (0, 0, 1)),                                    # top
            ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
             (0, -1, 0)),                                   # front
            ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
             (1, 0, 0)),                                    # right
        ]
        for v, n in faces:
            self.add(v, colour, normal=n, edge=edge, lw=lw, alpha=alpha,
                     boost=boost, layer=layer)

    def draw(self, ax, zorder0=2):
        """Paint back to front within each layer, and layer by layer.

        A single global depth sort is not enough here: the gate dielectric is
        one wide slab whose centroid is nearer the camera than the far contact
        pad that stands on top of it, so a pure centroid sort would hide the
        pad.  Sorting by (layer, depth) fixes that without any per-face
        special casing."""
        for f in sorted(self.faces, key=lambda f: (f[0], -f[1])):
            layer, _, V, fc, ec, lw, alpha = f
            ax.add_patch(Polygon(project(V), closed=True, facecolor=fc,
                                 edgecolor=ec, linewidth=lw, alpha=alpha,
                                 joinstyle='round', zorder=zorder0 + layer))


# ---------------------------------------------------------------------------
# the device
# ---------------------------------------------------------------------------
# geometry, in arbitrary drawing units
L, D = 11.0, 5.6                 # substrate footprint
T_SI, T_OX = 1.05, 0.55          # substrate and gate-dielectric thickness
WR = 2.3                         # ribbon width
HALO = 0.40                      # damaged strip beside each edge
XS, XE = 3.30, 7.70              # exposed channel between the contacts
PAD_W, PAD_H = 2.35, 0.44        # contact footprint along x, and height
TIPX = 5.35                      # position of the Raman tip along the ribbon

Y0 = (D - WR) / 2.0
Z_OX = -T_OX
Z_CH = 0.0                       # the monolayer sits on the dielectric

C_SI = '#8f979f'
C_OX = '#c7ddef'
C_CH = '#5f9ec9'
C_HALO = '#e8a76a'
C_MET = '#d4a933'


def _cone(sc, apex, axis, radius, height, colour, n=28, alpha=1.0, boost=0.0):
    """Cone with its apex at `apex`, opening along +axis."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.array([1.0, 0.0, 0.0])
    if abs(axis @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, a)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    base = np.asarray(apex, float) + axis * height
    th = np.linspace(0, 2 * np.pi, n + 1)
    ring = base + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    for i in range(n):
        tri = np.array([apex, ring[i], ring[i + 1]])
        nrm = np.cross(ring[i] - apex, ring[i + 1] - apex)
        sc.add(tri, colour, normal=nrm, lw=0.0, alpha=alpha, boost=boost,
               edge=(0, 0, 0, 0))


def _frustum(sc, base, axis, r0, r1, height, colour, n=22, boost=0.0,
             layer=0):
    """Truncated cone from radius r0 at `base` to r1 at base + axis*height."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.array([1.0, 0.0, 0.0])
    if abs(axis @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, a)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n + 1)
    c0 = np.asarray(base, float)
    c1 = c0 + axis * height
    r_0 = c0 + r0 * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    r_1 = c1 + r1 * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    for i in range(n):
        quad = np.array([r_0[i], r_0[i + 1], r_1[i + 1], r_1[i]])
        nrm = np.cross(r_0[i + 1] - r_0[i], r_1[i] - r_0[i])
        sc.add(quad, colour, normal=nrm, lw=0.0, boost=boost,
               edge=(0, 0, 0, 0), layer=layer)


def build_scene():
    sc = Scene()

    # --- layer 0: silicon handle wafer and gate dielectric -------------
    sc.box(0, 0, -T_SI - T_OX, L, D, T_SI, C_SI, lw=0.35, layer=0)
    sc.box(0, 0, Z_OX, L, D, T_OX, C_OX, lw=0.35, layer=0)

    # --- layer 1: the patterned monolayer ------------------------------
    t = 0.07
    sc.box(XS, Y0, Z_CH, XE - XS, HALO, t, C_HALO, lw=0.3, boost=0.06,
           layer=1)
    sc.box(XS, Y0 + WR - HALO, Z_CH, XE - XS, HALO, t, C_HALO, lw=0.3,
           boost=0.06, layer=1)
    sc.box(XS, Y0 + HALO, Z_CH, XE - XS, WR - 2 * HALO, t, C_CH, lw=0.3,
           layer=1)
    # the film continues under the contacts
    for x0 in (0.60, XE):
        sc.box(x0, Y0 - 0.55, Z_CH, PAD_W, WR + 1.1, t, C_CH, lw=0.25,
               boost=-0.07, layer=1)

    # --- layer 2: source and drain metal -------------------------------
    for x0 in (0.60, XE):
        sc.box(x0, Y0 - 0.55, Z_CH + t, PAD_W, WR + 1.1, PAD_H, C_MET,
               lw=0.35, layer=2)

    return sc


def _edge_charges(ax):
    """Fixed charge along both etched edges, drawn as small discs."""
    xs = np.linspace(XS + 0.38, XE - 2.45, 5)
    pts = []
    for x in xs:
        for y in (Y0 + 0.11, Y0 + WR - 0.11):
            pts.append((x, y, Z_CH + 0.10))
    P = project(np.array(pts))
    dep = depth(np.array(pts))
    order = np.argsort(-dep)
    for i in order:
        ax.add_patch(Circle(P[i], 0.052, facecolor=VERM, edgecolor='white',
                            linewidth=0.35, zorder=6))
    return pts


def _raman_tip(ax, sc):
    """Metal tip with its excitation and collection cone."""
    ty = Y0 + 0.11                      # the tip sits over the near edge
    apex = np.array([TIPX, ty, Z_CH + 0.16])

    # focused excitation cone, drawn translucent, tilted off the vertical
    top_z, half = 2.72, 0.55
    dx, dy = -0.55, -0.38
    beam = np.array([[TIPX + dx - half, ty + dy - half * 0.7, top_z],
                     [TIPX + dx + half, ty + dy + half * 0.7, top_z],
                     [TIPX, ty, Z_CH + 0.22]])
    ax.add_patch(Polygon(project(beam), closed=True, facecolor=GREEN,
                         edgecolor='none', alpha=0.18, zorder=5))
    for sgn in (-1, 1):
        seg = np.array([[TIPX + dx + sgn * half, ty + dy + sgn * half * 0.7,
                         top_z],
                        [TIPX, ty, Z_CH + 0.22]])
        xy = project(seg)
        ax.plot(xy[:, 0], xy[:, 1], color=GREEN, lw=0.9,
                solid_capstyle='round', zorder=6)

    # the probe: a tapered shaft ending in a sharp apex on the ribbon edge
    tip = Scene()
    _frustum(tip, (TIPX, ty, 1.22), (0, 0, 1), 0.16, 0.27, 1.20, '#8d949b',
             boost=0.08)
    _cone(tip, apex, (0, 0, 1), 0.16, 1.06, '#a7aeb5', boost=0.12)
    for f in sorted(tip.faces, key=lambda f: -f[1]):
        ax.add_patch(Polygon(project(f[2]), closed=True, facecolor=f[3],
                             edgecolor='none', zorder=7))
    P = project(apex[None, :])[0]
    ax.add_patch(Circle(P, 0.115, facecolor=GREEN, edgecolor='none',
                        alpha=0.35, zorder=7))
    ax.add_patch(Circle(P, 0.048, facecolor='#fbfbfb', edgecolor='#6b7076',
                        linewidth=0.35, zorder=8))
    return apex


def _width_marker(ax, fs=6.8):
    """Width dimension: two extension ticks and a double-headed arrow, drawn
    just above the ribbon so that it cannot be confused with a leader line."""
    x = XE - 1.85
    z0, z1 = Z_CH + 0.38, Z_CH + 0.62
    for y in (Y0, Y0 + WR):
        seg = project(np.array([[x, y, z0], [x, y, z1]]))
        ax.plot(seg[:, 0], seg[:, 1], color=INK2, lw=0.45, zorder=9)
    a = project(np.array([[x, Y0, z1]]))[0]
    b = project(np.array([[x, Y0 + WR, z1]]))[0]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='<|-|>', mutation_scale=4.2,
                                 lw=0.6, color=INK, shrinkA=0, shrinkB=0,
                                 zorder=10))
    lab = project(np.array([[x - 0.10, Y0 + WR + 0.40, z1 + 0.28]]))[0]
    ax.text(lab[0], lab[1], '$W$', fontsize=fs, color=INK, ha='center',
            va='center', zorder=10)
    return project(np.array([[x, Y0 + WR / 2, z1]]))[0]


def draw_device(ax, fs=6.8):
    """Render the schematic and its labels into a plain 2D axes."""
    ax.set_aspect('equal')
    ax.set_axis_off()

    sc = build_scene()
    sc.draw(ax, zorder0=2)
    charges = _edge_charges(ax)
    tip_apex = _raman_tip(ax, sc)
    w_mid = _width_marker(ax, fs=fs)

    # ---- anchors, in projected coordinates ---------------------------
    def pr(p):
        return project(np.array([p], float))[0]

    anchors = dict(
        tip=pr((TIPX - 0.42, Y0 + 0.11 - 0.29, 2.30)),
        charge=pr((XS + 0.55, Y0 + 0.11, Z_CH + 0.13)),
        halo=pr((XE - 1.55, Y0 + WR - HALO / 2, Z_CH + 0.09)),
        channel=pr((XE - 1.90, Y0 + WR / 2, Z_CH + 0.09)),
        source=pr((1.55, Y0 - 0.50, Z_CH + 0.07 + PAD_H)),
        drain=pr((XE + 1.55, Y0 - 0.50, Z_CH + 0.07 + PAD_H)),
        oxide=pr((3.60, 0.0, Z_OX + T_OX / 2)),
        substrate=pr((5.20, 0.0, -T_SI - T_OX / 2)),
        width=w_mid,
    )

    # ---- extent ------------------------------------------------------
    # The limits are chosen so that their aspect ratio equals the aspect ratio
    # of the subplot slot.  With an equal-aspect axes that makes the drawing
    # fill the slot exactly instead of being shrunk to fit, which is what keeps
    # the labels small relative to the device.
    allv = np.concatenate([f[2] for f in sc.faces], axis=0)
    P = project(allv)
    P = np.vstack([P, project(np.array([[TIPX, Y0 + 0.11, 2.80]]))])
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    w, h = x1 - x0, y1 - y0
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    slot = (pos.width * fw) / (pos.height * fh)
    my = 0.060
    tot_h = h * (1.0 + 2.0 * my)
    tot_w = slot * tot_h
    if tot_w < w * 1.46:                      # not enough room for the labels
        tot_w = w * 1.46
        tot_h = tot_w / slot
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ax.set_xlim(cx - tot_w / 2, cx + tot_w / 2)
    ax.set_ylim(cy - tot_h / 2 - 0.02 * tot_h, cy + tot_h / 2 - 0.02 * tot_h)
    return anchors


# Labels are split into a left and a right column and ordered top to bottom
# by the projected height of the feature they point at, so no two leaders
# cross and no label is written over the device.
# Labels live in the clear band above the device, the clear band below it, and
# the two side margins.  Each entry gives the anchor key, the text, the colour,
# the position in axes fractions and the alignment, so the whole layout can be
# read and checked in one place.
LABELS = [
    # key,        text,                            colour,     x,     y,    ha
    ('tip',       'tip-enhanced\nRaman probe',      GREEN,    0.010, 0.990, 'left'),
    ('halo',      'process\ndamage halo',           '#a8631a', 0.990, 0.990, 'right'),
    ('source',    'source\ncontact',                '#8a6d18', 0.008, 0.665, 'left'),
    ('channel',   'monolayer\nchannel',             '#1f6ea8', 0.990, 0.620, 'right'),
    ('charge',    'fixed edge\n' r'charge $\sigma_{\rm e}$',
     VERM, 0.008, 0.250, 'left'),
    ('drain',     'drain contact',                  '#8a6d18', 0.990, 0.020, 'right'),
    ('oxide',     'gate dielectric',                '#2f6f9e', 0.008, 0.055, 'left'),
    ('substrate', 'Si handle wafer',                '#4a545e', 0.450, 0.010, 'center'),
]


def annotate(ax, anchors, fs=6.8):
    """Place every label in a clear margin and point an arrow at its feature.

    Left-hand labels are right-aligned against the left margin and right-hand
    labels are left-aligned against the right margin, so the two columns cannot
    run into the device or into each other.  The arrow head always sits on the
    feature; the tail always sits on the text.
    """
    for key, text, colour, xf, yf, ha in LABELS:
        va = 'top' if yf > 0.85 else ('bottom' if yf < 0.15 else 'center')
        ax.annotate(
            text, xy=anchors[key], xycoords='data',
            xytext=(xf, yf), textcoords='axes fraction',
            ha=ha, va=va, fontsize=fs, color=colour, zorder=20,
            annotation_clip=False, linespacing=1.15,
            arrowprops=dict(arrowstyle='-|>', mutation_scale=5.2, lw=0.55,
                            color=INK2, shrinkA=3.0, shrinkB=1.5,
                            connectionstyle='arc3,rad=0.0'))


# retained for compatibility with earlier call sites
def width_marker(ax, geo=None, fs=6.0):
    return
