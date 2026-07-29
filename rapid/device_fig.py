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

What the drawing asserts about the device is meant to be right, not merely
suggestive:

  * the monolayer is one ribbon of constant width running the whole length of
    the device and passing under both contacts.  The etch that defines it
    leaves a damaged strip beside each edge along that whole length, so the
    halo is drawn along the full ribbon and not only in the exposed channel.
  * the stack is gate / gate dielectric / monolayer, with a gate terminal, so
    the object is a transistor and not a resistor on a wafer.  The same
    picture covers a doped-silicon back gate and a local metal gate; only the
    dielectric thickness changes between them.
  * the fixed charge sits on the etched sidewalls, which is where the
    dangling bonds are, and is drawn there rather than on the top surface.
  * the exposed channel carries the 1H lattice, drawn as a real honeycomb of
    metal and chalcogen sublattices, so the channel reads as a crystal one
    atom thick.

Layer thicknesses are exaggerated vertically by ZSCALE; the caption says so.
The inset cross-section carries the detail that cannot be seen in projection.

Everything is regenerated with the rest of the figures and needs no external
drawing program.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import (Circle, FancyArrowPatch, Polygon, Rectangle,
                                Wedge)

# palette, kept in step with figures.py
BLUE = '#0072B2'
VERM = '#D55E00'
GREEN = '#009E73'
INK = '#1a1a1a'
INK2 = '#555555'

# ---------------------------------------------------------------------------
# camera and lighting
# ---------------------------------------------------------------------------
AZ = np.deg2rad(-54.0)          # rotation about the vertical axis
EL = np.deg2rad(22.0)           # elevation above the substrate plane
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
T_SI, T_GM, T_OX = 0.80, 0.26, 0.55   # gate body, gate metal, dielectric
WR = 1.70                        # ribbon width
HALO = 0.30                      # damaged strip beside each edge
PAD_W, PAD_H = 2.30, 0.46        # contact footprint along x, and height
PAD_X0, PAD_X1 = 0.80, 7.45      # near and far contact, leading edges
XS, XE = PAD_X0 + PAD_W, PAD_X1  # exposed channel between the contacts
RIB_X0, RIB_X1 = 0.65, 9.90      # the ribbon overhangs both contacts
TIPX = 4.55                      # position of the Raman tip along the ribbon

Y0 = (D - WR) / 2.0
Z_GM = -T_OX - T_GM
Z_OX = -T_OX
Z_CH = 0.0                       # the monolayer sits on the dielectric
T_MONO = 0.075

C_SI = '#8f979f'
C_GM = '#6b7480'
C_OX = '#c7ddef'
C_CH = '#4f93c2'
C_HALO = '#e8a76a'
C_MET = '#d4a933'


def _cone(sc, apex, axis, radius, height, colour, n=28, alpha=1.0, boost=0.0,
          layer=0):
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
               edge=(0, 0, 0, 0), layer=layer)


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

    # --- layer 0: gate body, gate metal and gate dielectric ------------
    sc.box(0, 0, -T_SI - T_GM - T_OX, L, D, T_SI, C_SI, lw=0.35, layer=0)
    sc.box(0, 0, Z_GM, L, D, T_GM, C_GM, lw=0.35, layer=0)
    sc.box(0, 0, Z_OX, L, D, T_OX, C_OX, lw=0.35, layer=0)

    # --- layer 1: the patterned monolayer ------------------------------
    # One ribbon of constant width over the whole length of the device: the
    # etch defines it everywhere, so the damaged strip beside each edge runs
    # the whole length too and is not confined to the exposed channel.
    dx = RIB_X1 - RIB_X0
    sc.box(RIB_X0, Y0, Z_CH, dx, HALO, T_MONO, C_HALO, lw=0.25, boost=0.06,
           layer=1)
    sc.box(RIB_X0, Y0 + WR - HALO, Z_CH, dx, HALO, T_MONO, C_HALO, lw=0.25,
           boost=0.06, layer=1)
    sc.box(RIB_X0, Y0 + HALO, Z_CH, dx, WR - 2 * HALO, T_MONO, C_CH, lw=0.25,
           layer=1)

    # --- layer 2: source and drain metal -------------------------------
    # The pads land on the ribbon and overhang it slightly, as a patterned
    # metal on a narrower ribbon does.
    for x0 in (PAD_X0, PAD_X1):
        sc.box(x0, Y0 - 0.42, Z_CH + T_MONO, PAD_W, WR + 0.84, PAD_H, C_MET,
               lw=0.32, layer=2)

    return sc


# ---------------------------------------------------------------------------
# surface detail
# ---------------------------------------------------------------------------
def _lattice(ax, x0, x1, y0, y1, z, a=0.262, zorder=3.4):
    """1H honeycomb on the exposed channel, both sublattices and the bonds.

    The net is generated in the plane of the monolayer and then projected, so
    it follows the perspective of the slab it sits on instead of being a flat
    overlay.  The index range is obtained by mapping the four corners of the
    exposed rectangle through the inverse of the lattice basis, which is what
    guarantees the net covers the rectangle completely.
    """
    d = a / np.sqrt(3.0)                 # nearest-neighbour distance
    a1 = np.array([a, 0.0])
    a2 = np.array([0.5 * a, 0.5 * np.sqrt(3.0) * a])
    B = np.column_stack([a1, a2])
    Binv = np.linalg.inv(B)
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    ij = (Binv @ (corners - np.array([x0, y0])).T).T
    i0, i1 = int(np.floor(ij[:, 0].min())) - 1, int(np.ceil(ij[:, 0].max())) + 1
    j0, j1 = int(np.floor(ij[:, 1].min())) - 1, int(np.ceil(ij[:, 1].max())) + 1

    nn = [np.array([0.0, d]),
          np.array([0.5 * np.sqrt(3.0) * d, -0.5 * d]),
          np.array([-0.5 * np.sqrt(3.0) * d, -0.5 * d])]

    def inside(p):
        return (x0 <= p[0] <= x1) and (y0 <= p[1] <= y1)

    subA, subB, segs = [], [], []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            A = np.array([x0, y0]) + i * a1 + j * a2
            inA = inside(A)
            if inA:
                subA.append(A)
            for v in nn:
                Bp = A + v
                if inA and inside(Bp):
                    segs.append([A, Bp])
                    if v[1] > 0:
                        subB.append(Bp)
    if segs:
        S3 = [np.array([[p[0], p[1], z] for p in s_]) for s_ in segs]
        P = [project(s_) for s_ in S3]
        ax.add_collection(LineCollection(P, colors='#245f86', linewidths=0.22,
                                         alpha=0.85, zorder=zorder))
    for pts, r, col in ((subA, 0.021, '#14527a'), (subB, 0.013, '#a8d6ee')):
        if not pts:
            continue
        Q = project(np.array([[p[0], p[1], z] for p in pts]))
        for xy in Q:
            ax.add_patch(Circle(xy, r, facecolor=col, edgecolor='none',
                                zorder=zorder + 0.05))


def _edge_charges(ax):
    """Fixed charge on the two etched sidewalls, drawn where it actually sits.

    The discs are placed on the vertical faces of the ribbon rather than on its
    top surface, and each is given a minus sign, so the picture says the charge
    is trapped at the cut edge.
    """
    xs = np.linspace(XS + 0.34, XE - 0.34, 5)
    pts, faces = [], []
    for x in xs:
        pts.append((x, Y0 - 0.006, Z_CH + 0.55 * T_MONO))          # near edge
        faces.append('near')
        pts.append((x, Y0 + WR + 0.006, Z_CH + 0.55 * T_MONO))     # far edge
        faces.append('far')
    A = np.array(pts)
    P = project(A)
    dep = depth(A)
    for i in np.argsort(-dep):
        vis = faces[i] == 'near'
        ax.add_patch(Circle(P[i], 0.049 if vis else 0.038, facecolor=VERM,
                            edgecolor='white', linewidth=0.30,
                            alpha=1.0 if vis else 0.55,
                            zorder=9 if vis else 3.6))
        if vis:
            ax.plot([P[i][0] - 0.022, P[i][0] + 0.022], [P[i][1], P[i][1]],
                    color='white', lw=0.45, solid_capstyle='round', zorder=10)
    return pts


def _terminals(ax, fs=6.8):
    """Gate and drain terminals, so the object reads as a transistor."""
    out = {}
    # drain bias on the far pad
    top = Z_CH + T_MONO + PAD_H
    p0 = np.array([XE + PAD_W - 0.55, Y0 - 0.42, top])
    seg = project(np.array([p0, p0 + np.array([0.0, -0.95, 0.72])]))
    ax.plot(seg[:, 0], seg[:, 1], color=INK2, lw=0.7, solid_capstyle='round',
            zorder=11)
    ax.add_patch(Circle(seg[-1], 0.055, facecolor='white', edgecolor=INK2,
                        linewidth=0.55, zorder=12))
    out['drain_term'] = seg[-1]
    # gate bias on the buried metal
    g0 = np.array([L - 0.02, D * 0.18, Z_GM + T_GM / 2])
    seg = project(np.array([g0, g0 + np.array([0.95, -0.35, -0.20])]))
    ax.plot(seg[:, 0], seg[:, 1], color=INK2, lw=0.7, solid_capstyle='round',
            zorder=11)
    ax.add_patch(Circle(seg[-1], 0.055, facecolor='white', edgecolor=INK2,
                        linewidth=0.55, zorder=12))
    out['gate_term'] = seg[-1]
    # ground on the near pad
    s0 = np.array([RIB_X0 - 0.45, Y0 - 0.42, top])
    seg = project(np.array([s0, s0 + np.array([0.0, -0.85, 0.30])]))
    ax.plot(seg[:, 0], seg[:, 1], color=INK2, lw=0.7, solid_capstyle='round',
            zorder=11)
    e = seg[-1]
    for k, hw in enumerate((0.105, 0.070, 0.036)):
        ax.plot([e[0] - hw, e[0] + hw], [e[1] - 0.030 * k, e[1] - 0.030 * k],
                color=INK2, lw=0.55, solid_capstyle='round', zorder=12)
    out['source_term'] = e
    return out


def _raman_tip(ax):
    """Metal tip with a true excitation cone converging on its apex."""
    ty = Y0 - 0.02                      # the tip sits over the near edge
    apex = np.array([TIPX, ty, Z_CH + T_MONO + 0.05])

    # excitation cone: a real cone about the tip axis, painted translucent
    cone = Scene()
    _cone(cone, apex, (0, 0, 1), 0.46, 2.30, GREEN, n=34, alpha=0.16)
    for f in sorted(cone.faces, key=lambda f: -f[1]):
        ax.add_patch(Polygon(project(f[2]), closed=True, facecolor=f[3],
                             edgecolor='none', alpha=0.16, zorder=5))
    for sgn in (-1, 1):
        seg = project(np.array([[TIPX + sgn * 0.46, ty, Z_CH + 2.35],
                                apex + np.array([0.0, 0.0, 0.04])]))
        ax.plot(seg[:, 0], seg[:, 1], color=GREEN, lw=0.8,
                solid_capstyle='round', zorder=6)

    # the probe: a tapered shaft ending in a sharp apex on the ribbon edge
    tip = Scene()
    _frustum(tip, (TIPX, ty, 1.30), (0, 0, 1), 0.14, 0.25, 1.15, '#8d949b',
             boost=0.08)
    _cone(tip, apex, (0, 0, 1), 0.14, 1.14, '#a7aeb5', boost=0.12)
    for f in sorted(tip.faces, key=lambda f: -f[1]):
        ax.add_patch(Polygon(project(f[2]), closed=True, facecolor=f[3],
                             edgecolor='none', zorder=7))

    # the hot spot under the apex, and the scattered light leaving it
    P = project(apex[None, :])[0]
    ax.add_patch(Circle(P, 0.130, facecolor=GREEN, edgecolor='none',
                        alpha=0.32, zorder=7))
    ax.add_patch(Circle(P, 0.048, facecolor='#fbfbfb', edgecolor='#6b7076',
                        linewidth=0.35, zorder=8))
    q0 = apex + np.array([-0.34, -0.30, 0.42])
    q1 = apex + np.array([-1.55, -1.05, 1.95])
    a, b = project(np.array([q0, q1]))
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=4.6,
                                 lw=0.7, color=VERM, shrinkA=0, shrinkB=0,
                                 zorder=11))
    return apex, project(np.array([[TIPX - 0.46, ty, Z_CH + 2.26]]))[0], b


def _dimensions(ax, fs=6.8):
    """Width across the ribbon and channel length between the contacts."""
    out = {}
    # W, lifted just above the ribbon
    x = XS + 0.62
    z1 = Z_CH + 0.50
    for y in (Y0, Y0 + WR):
        seg = project(np.array([[x, y, Z_CH + T_MONO], [x, y, z1]]))
        ax.plot(seg[:, 0], seg[:, 1], color=INK2, lw=0.40, zorder=9)
    a = project(np.array([[x, Y0, z1]]))[0]
    b = project(np.array([[x, Y0 + WR, z1]]))[0]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='<|-|>', mutation_scale=4.0,
                                 lw=0.55, color=INK, shrinkA=0, shrinkB=0,
                                 zorder=10))
    lab = project(np.array([[x, Y0 + WR + 0.46, z1 + 0.20]]))[0]
    ax.text(lab[0], lab[1], '$W$', fontsize=fs, color=INK, ha='center',
            va='bottom', zorder=10)
    out['width'] = 0.5 * (a + b)
    return out


# ---------------------------------------------------------------------------
# inset: the cross-section that projection cannot show
# ---------------------------------------------------------------------------
def draw_cross_section(ax, rect=(0.000, 0.010, 0.335, 0.375), fs=5.4):
    """True cross-section through the ribbon, looking along the channel.

    This is where the detail lives: the halo strip beside each edge, the fixed
    charge on the two sidewalls and the band bending it produces, and the
    layers of the gate stack in their real order.  Nothing here is in
    perspective, so widths and thicknesses can be read off directly.
    """
    cax = ax.inset_axes(rect)
    cax.set_xlim(-1.18, 1.14)
    cax.set_ylim(-0.86, 1.16)
    cax.set_axis_off()
    cax.set_facecolor('none')

    w = 1.0
    h_ox, h_gm = 0.30, 0.13
    y_ch = 0.0
    t = 0.085
    wh = 0.16                                     # halo width, in units of W

    # gate stack
    cax.add_patch(Rectangle((-0.10, y_ch - h_ox - h_gm - 0.16), w + 0.20,
                            0.16, facecolor=C_SI, edgecolor='#6e767e',
                            lw=0.30))
    cax.add_patch(Rectangle((-0.10, y_ch - h_ox - h_gm), w + 0.20, h_gm,
                            facecolor=C_GM, edgecolor='#4e565f', lw=0.30))
    cax.add_patch(Rectangle((-0.10, y_ch - h_ox), w + 0.20, h_ox,
                            facecolor=C_OX, edgecolor='#8fb2ce', lw=0.30))
    # the monolayer, with a damaged strip beside each edge
    cax.add_patch(Rectangle((0.0, y_ch), wh, t, facecolor=C_HALO,
                            edgecolor='#b57433', lw=0.30))
    cax.add_patch(Rectangle((w - wh, y_ch), wh, t, facecolor=C_HALO,
                            edgecolor='#b57433', lw=0.30))
    cax.add_patch(Rectangle((wh, y_ch), w - 2 * wh, t, facecolor=C_CH,
                            edgecolor='#2b6a92', lw=0.30))

    # fixed charge on the two sidewalls
    for x in (0.0, w):
        for dz in (0.30, 0.70):
            cax.add_patch(Circle((x, y_ch + dz * t), 0.036, facecolor=VERM,
                                 edgecolor='white', lw=0.25, zorder=6))
    # the depletion each edge produces, sketched as a shaded wedge
    for x, sgn in ((0.0, +1), (w, -1)):
        cax.add_patch(Polygon([[x, y_ch + t], [x + sgn * 0.30, y_ch + t],
                               [x, y_ch + t + 0.20]], closed=True,
                              facecolor=VERM, edgecolor='none', alpha=0.16,
                              zorder=5))

    # dielectric thickness, marked where it can be read
    cax.annotate('', xy=(0.86, y_ch), xytext=(0.86, y_ch - h_ox),
                 arrowprops=dict(arrowstyle='<|-|>', lw=0.42, color='#2f6f9e',
                                 mutation_scale=2.8, shrinkA=0, shrinkB=0))
    cax.text(0.90, y_ch - 0.5 * h_ox, r'$t_{\rm ox}$', fontsize=fs,
             color='#2f6f9e', ha='left', va='center')

    # width dimension, and the halo width
    yv = y_ch + t + 0.40
    cax.annotate('', xy=(0.0, yv), xytext=(w, yv),
                 arrowprops=dict(arrowstyle='<|-|>', lw=0.5, color=INK,
                                 mutation_scale=3.6, shrinkA=0, shrinkB=0))
    cax.text(0.5 * w, yv + 0.03, '$W$', fontsize=fs, color=INK, ha='center',
             va='bottom')
    for x in (0.0, w):
        cax.plot([x, x], [y_ch + t, yv], color=INK2, lw=0.30, ls=':')
    cax.annotate('', xy=(wh, y_ch + t + 0.22), xytext=(0.0, y_ch + t + 0.22),
                 arrowprops=dict(arrowstyle='<|-|>', lw=0.45, color='#a8631a',
                                 mutation_scale=3.0, shrinkA=0, shrinkB=0))
    cax.text(wh + 0.035, y_ch + t + 0.22, r'$w_{\rm h}$', fontsize=fs,
             color='#a8631a', ha='left', va='center')

    # layer labels, kept inside the inset
    cax.text(-0.16, y_ch + 0.5 * t, 'monolayer', fontsize=fs, color='#1f6ea8',
             ha='right', va='center')
    cax.text(-0.16, y_ch - 0.52 * h_ox, 'dielectric', fontsize=fs,
             color='#2f6f9e', ha='right', va='center')
    cax.text(-0.16, y_ch - h_ox - 0.55 * h_gm, 'gate', fontsize=fs,
             color='#3f4750', ha='right', va='center')
    cax.text(0.5 * w, y_ch - h_ox - h_gm - 0.34, 'section A--A', fontsize=fs,
             color=INK2, ha='center', va='center', style='italic')
    return cax


# ---------------------------------------------------------------------------
def _cut_plane(ax, fs=6.8, x=None):
    """Mark where the inset cross-section is taken."""
    if x is None:
        x = XE - 0.58
    a = project(np.array([[x, Y0 - 0.80, Z_CH + T_MONO]]))[0]
    b = project(np.array([[x, Y0 + WR + 0.80, Z_CH + T_MONO]]))[0]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.55, ls=(0, (3.2, 1.6)),
            zorder=11)
    for p, t in ((a, 'A'), (b, 'A')):
        ax.text(p[0], p[1] - 0.055, t, fontsize=fs - 0.9, color=INK,
                ha='center', va='top', zorder=11)
    return a, b


def draw_device(ax, fs=6.8):
    """Render the schematic and its labels into a plain 2D axes."""
    ax.set_aspect('equal')
    ax.set_axis_off()

    sc = build_scene()
    sc.draw(ax, zorder0=2)
    _lattice(ax, XS + 0.06, XE - 0.06, Y0 + HALO + 0.03,
             Y0 + WR - HALO - 0.03, Z_CH + T_MONO + 0.001)
    _edge_charges(ax)
    tip_apex, beam_top, scat = _raman_tip(ax)
    dims = _dimensions(ax, fs=fs)
    terms = _terminals(ax, fs=fs)
    _cut_plane(ax, fs=fs)

    # ---- anchors, in projected coordinates ---------------------------
    def pr(p):
        return project(np.array([p], float))[0]

    anchors = dict(
        tip=beam_top,
        scatter=scat,
        charge=pr((XS + 0.30, Y0, Z_CH + 0.55 * T_MONO)),
        halo=pr((XE - 0.55, Y0 + WR - HALO / 2, Z_CH + T_MONO)),
        channel=pr((XS + 2.20, Y0 + WR / 2, Z_CH + T_MONO)),
        source=pr((RIB_X0 + 0.35, Y0 - 0.42, Z_CH + T_MONO + PAD_H)),
        drain=pr((XE + 1.20, Y0 - 0.42, Z_CH + T_MONO + PAD_H)),
        oxide=pr((2.05, 0.0, Z_OX + T_OX / 2)),
        gatemetal=pr((2.55, 0.0, Z_GM + T_GM / 2)),
        substrate=pr((3.60, 0.0, -T_SI - T_GM - T_OX / 2)),
        **dims, **terms,
    )

    # ---- extent ------------------------------------------------------
    # The limits are chosen so that their aspect ratio equals the aspect ratio
    # of the subplot slot.  With an equal-aspect axes that makes the drawing
    # fill the slot exactly instead of being shrunk to fit, which is what keeps
    # the labels small relative to the device.
    allv = np.concatenate([f[2] for f in sc.faces], axis=0)
    P = project(allv)
    P = np.vstack([P, project(np.array([[TIPX, Y0, Z_CH + 2.40]])),
                   np.array([beam_top, scat, terms['gate_term'],
                             terms['drain_term'], terms['source_term']])])
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    w, h = x1 - x0, y1 - y0
    # asymmetric margins: a wide band below and to the left, which is where the
    # cross-section inset and the bottom labels live, and thin bands above and
    # to the right, which only have to clear the label columns.
    ml, mr, mb, mt = 0.07, 0.07, 0.18, 0.17
    win_x0, win_x1 = x0 - ml * w, x1 + mr * w
    win_y0, win_y1 = y0 - mb * h, y1 + mt * h
    ww, hh = win_x1 - win_x0, win_y1 - win_y0
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    slot = (pos.width * fw) / (pos.height * fh)
    if ww / hh < slot:
        # Too narrow for the slot.  The slack is split unevenly, more of it to
        # the left, because the left band carries the cross-section inset and
        # the left label column while the right band only carries text.
        need = slot * hh - ww
        win_x0 -= 0.52 * need
        win_x1 += 0.48 * need
    else:                                  # too short: deepen downwards
        need = ww / slot - hh
        win_y0 -= need
    ax.set_xlim(win_x0, win_x1)
    ax.set_ylim(win_y0, win_y1)
    return anchors


# Labels live in the clear band above the device, the clear band below it and
# the two side margins.  Each entry gives the anchor key, the text, the colour,
# the position in axes fractions and the alignment, so the whole layout can be
# read and checked in one place.
LABELS = [
    # key,          text,                        colour,     x,     y,    ha
    ('tip',         '532 nm\nexcitation',        GREEN,    0.335, 0.995, 'center'),
    ('channel',     '1H monolayer\nchannel',     '#1f6ea8', 0.985, 0.995, 'right'),
    ('halo',        'damage halo',               '#a8631a', 0.985, 0.720, 'right'),
    ('drain',       'drain',                     '#8a6d18', 0.985, 0.500, 'right'),
    ('source',      'source',                    '#8a6d18', 0.008, 0.830, 'left'),
    ('scatter',     'Raman\nscattering',         VERM,     0.008, 0.995, 'left'),
    ('gate_term',   r'$V_{\rm G}$',              INK2,     0.985, 0.170, 'right'),
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
            annotation_clip=False, linespacing=1.12,
            arrowprops=dict(arrowstyle='-|>', mutation_scale=5.0, lw=0.50,
                            color=INK2, shrinkA=2.5, shrinkB=1.5,
                            connectionstyle='arc3,rad=0.0'))


# retained for compatibility with earlier call sites
def width_marker(ax, geo=None, fs=6.0):
    return
