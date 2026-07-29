"""Main-text figures.

Colours use a fixed, colour-vision-deficiency-safe categorical order
(blue, vermillion, bluish green, purple) assigned to the four materials and
never cycled; sequential fields use a single-hue ramp and signed fields a
two-hue diverging ramp with a neutral midpoint.  Series identity is carried by
a legend or by a direct label with a leader line, never by colour alone.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from . import datasets as D, device_fig, transport
from .materials import all_materials, MATERIALS

ROOT = os.path.join(os.path.dirname(__file__), '..')
FIGS = os.path.join(ROOT, 'figs')
os.makedirs(FIGS, exist_ok=True)

CAT = ['#0072B2', '#D55E00', '#009E73', '#7A5195', '#CC79A7', '#8C6D1F']
CMAT = dict(zip(MATERIALS, CAT))
LAB = {'MoS2': r'MoS$_2$', 'WS2': r'WS$_2$', 'MoSe2': r'MoSe$_2$',
       'WSe2': r'WSe$_2$'}
INK, INK2, MUTED, GRID = '#1a1a1a', '#5c5c5c', '#8a8a8a', '#dedede'
SEQ = LinearSegmentedColormap.from_list(
    'seq', ['#f2f7fb', '#a8cee4', '#4f97c4', '#0072B2', '#01456d'])
DIV = LinearSegmentedColormap.from_list(
    'div', ['#01456d', '#4f97c4', '#c8d3d8', '#e79a5c', '#8c3d00'])

FS_ANN = 6.4          # in-panel annotation
FS_LEG = 6.4          # legend
FS_TAG = 8.4          # panel letter

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 7.2, 'axes.labelsize': 7.5, 'axes.titlesize': 7.6,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': FS_LEG,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
    'xtick.minor.size': 1.4, 'ytick.minor.size': 1.4,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.3, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'grid.color': GRID, 'grid.linewidth': 0.45, 'legend.frameon': False,
    'legend.handlelength': 1.1, 'legend.handletextpad': 0.5,
    'legend.labelspacing': 0.30, 'legend.borderpad': 0.15,
    'legend.borderaxespad': 0.25, 'legend.columnspacing': 0.9,
    'figure.dpi': 400, 'savefig.dpi': 400, 'savefig.bbox': 'standard',
})

FULL_W = 6.85


def _panel(ax, tag, dx=-0.020, dy=1.045, color=INK):
    """Panel letter, placed outside the axes above the top-left corner.

    Keeping every panel letter out of the data area removes any chance of a
    collision with a curve, a marker or a spine.
    """
    t = ax.text(dx, dy, tag, transform=ax.transAxes, fontweight='bold',
                fontsize=FS_TAG, va='bottom', ha='left', color=color,
                zorder=30)
    t.set_clip_on(False)
    return t


def _note(ax, x, y, s, color=INK2, **kw):
    """In-panel note in axes fractions, with sensible defaults."""
    kw.setdefault('va', 'top')
    kw.setdefault('ha', 'left')
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=FS_ANN,
                   color=color, zorder=25, **kw)


def _load():
    with open(os.path.join(ROOT, 'results.json')) as fh:
        return json.load(fh)


def _lorentz(x, x0, g, a):
    return a * (g / 2.0) ** 2 / ((x - x0) ** 2 + (g / 2.0) ** 2)


# ===========================================================================
def fig_abstract(res):
    """Figure 1: the device, the two phonon channels, the separation and the
    resulting width limit."""
    kv = res['krayev']
    rb = res['ribbons']
    fig = plt.figure(figsize=(FULL_W, 4.28))
    # The top row gives the schematic more width than the spectra; the bottom
    # row is split evenly, so the two ordinary plots stay the same size.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.18, 1.00], hspace=0.40,
                             left=0.078, right=0.985, top=0.945, bottom=0.086)
    gtop = outer[0].subgridspec(1, 2, width_ratios=[1.34, 1.00], wspace=0.20)
    gbot = outer[1].subgridspec(1, 2, width_ratios=[1.00, 1.00], wspace=0.30)

    # ---------------- (a) device schematic -----------------------------
    ax = fig.add_subplot(gtop[0, 0])
    anchors = device_fig.draw_device(ax, fs=6.3)
    device_fig.draw_cross_section(ax, rect=(0.012, 0.012, 0.404, 0.372),
                                  fs=5.2)
    device_fig.annotate(ax, anchors, fs=6.3)
    t = ax.text(0.0, 1.0, '(a)', transform=ax.transAxes, fontweight='bold',
                fontsize=FS_TAG, va='bottom', ha='left', color=INK)
    t.set_clip_on(False)

    # ---------------- (b) the two phonon channels ----------------------
    ax = fig.add_subplot(gtop[0, 1])
    w = np.linspace(382, 474, 1500)
    for i, (col, lab) in enumerate([(CAT[0], 'ribbon centre'),
                                    (CAT[1], 'ribbon edge')]):
        y = (_lorentz(w, 403.0 - 0.5 * i, 5.2, 1.0) +
             _lorentz(w, 455.2, 13.0, 0.55))
        ax.plot(w, y + 1.30 * i, color=col, lw=1.4, label=lab)
    ax.set_xlabel(r'Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_yticks([])
    ax.set_xlim(382, 474)
    ax.set_ylim(-0.10, 3.85)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99),
              handlelength=1.0, labelspacing=0.20)
    ax.annotate(r"$A_1^{\prime}$ shifts by 0.5 cm$^{-1}$",
                xy=(403.6, 2.36), xytext=(0.42, 0.965),
                textcoords='axes fraction', ha='left', va='top',
                fontsize=FS_ANN, color=CAT[1],
                arrowprops=dict(arrowstyle='-', lw=0.5, color=CAT[1],
                                shrinkA=2, shrinkB=3))
    ax.annotate('2LA(M) does not', xy=(455.2, 1.88), xytext=(0.985, 0.66),
                textcoords='axes fraction', ha='right', va='center',
                fontsize=FS_ANN, color=CAT[1],
                arrowprops=dict(arrowstyle='-', lw=0.5, color=CAT[1],
                                shrinkA=2, shrinkB=3))
    _panel(ax, '(b)', dx=-0.135)

    # ---------------- (c) separation of charge from strain -------------
    ax = fig.add_subplot(gbot[0, 0])
    eps = np.linspace(-0.09, 0.30, 60)
    for col, dwA, dw2, lab in ((CAT[1], D.KRAYEV['edge_dwA'],
                                D.KRAYEV['edge_dw2LA'], 'ribbon edge'),
                               (CAT[2], D.KRAYEV['spot_dwA'],
                                D.KRAYEV['spot_dw2LA'], 'interior spot')):
        nA = (dwA - kv['dwA_deps'] * eps) / kv['dwA_dn']
        ax.plot(eps, nA * 10, color=col, lw=1.4, label=lab)
        ax.axvline(dw2 / kv['dw2LA_deps'], color=col, ls='--', lw=1.1)
    ax.plot(kv['edge']['strain_pct'], kv['edge']['n_1e13'] * 10, 'o', ms=6.0,
            color=CAT[1], mec=INK, mew=0.8, zorder=6)
    ax.plot(kv['spot']['strain_pct'], kv['spot']['n_1e13'] * 10, 's', ms=5.6,
            color=CAT[2], mec=INK, mew=0.8, zorder=6)
    ax.annotate('edge: charge,\nno strain',
                xy=(kv['edge']['strain_pct'], kv['edge']['n_1e13'] * 10),
                xytext=(0.30, 0.985), textcoords='axes fraction',
                ha='left', va='top', fontsize=FS_ANN, color=CAT[1],
                arrowprops=dict(arrowstyle='-', lw=0.5, color=CAT[1],
                                shrinkA=2, shrinkB=4))
    ax.annotate('interior spot:\nstrain, no charge',
                xy=(kv['spot']['strain_pct'], kv['spot']['n_1e13'] * 10),
                xytext=(0.985, 0.34), textcoords='axes fraction',
                ha='right', va='top', fontsize=FS_ANN, color=CAT[2],
                arrowprops=dict(arrowstyle='-', lw=0.5, color=CAT[2],
                                shrinkA=2, shrinkB=4))
    ax.set_xlabel('Biaxial strain (%)')
    ax.set_ylabel(r'Electron density ($10^{12}$ cm$^{-2}$)')
    ax.set_xlim(-0.09, 0.30)
    ax.set_ylim(-1.90, 5.40)
    ax.legend(loc='lower left', bbox_to_anchor=(0.01, -0.01),
              handlelength=1.0, labelspacing=0.20)
    _panel(ax, '(c)', dx=-0.150)

    # ---------------- (d) the resulting width limit --------------------
    ax = fig.add_subplot(gbot[0, 1])
    place = {'HfO2_EOT1p5': ((0.42, 0.235), 'left', 'center'),
             'SiO2_90nm': ((0.985, 0.315), 'right', 'center')}
    for tag, col, lab in (('HfO2_EOT1p5', CAT[2], r'thin high-$\kappa$ gate'),
                          ('SiO2_90nm', CAT[1], r'90 nm SiO$_2$ gate')):
        Wl = np.array(rb['cox'][tag]['W'])
        # normalised on the same wide-channel reference (W_REF_NM, outside
        # the plotted range) that defines W_c, so the 50 % level of the dotted
        # line is exactly the critical-width criterion
        I = np.array(rb['cox'][tag]['I']) / rb['cox'][tag]['I_ref']
        ax.semilogx(Wl, I, color=col, lw=1.6, label=lab)
        Wc = rb['cox'][tag]['Wc']
        ax.plot([Wc], [0.5], 'o', ms=5.4, color=col, mec='white', mew=0.9,
                zorder=6)
        xy_t, ha, va = place[tag]
        ax.annotate(r'$W_{\rm c}=%.0f$ nm' % Wc, xy=(Wc, 0.5), xytext=xy_t,
                    textcoords='axes fraction', ha=ha, va=va,
                    fontsize=FS_ANN, color=col,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color=col,
                                    shrinkA=2, shrinkB=4))
    ax.axhline(0.5, color=MUTED, ls=':', lw=0.9)
    ax.set_xlabel('Nanoribbon width (nm)')
    ax.set_ylabel(r'$I_{\rm on}/W$, normalised')
    ax.set_xlim(8, 1000)
    ax.set_ylim(0, 1.18)
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, -0.01),
              handlelength=1.0, labelspacing=0.20)
    _panel(ax, '(d)', dx=-0.150)

    fig.savefig(os.path.join(FIGS, 'fig_abstract.pdf'))
    plt.close(fig)


# ===========================================================================
def fig1(res):
    """Figure 2: first-principles deformation response."""
    mats = all_materials()
    fig = plt.figure(figsize=(FULL_W, 2.26))
    gs = fig.add_gridspec(1, 4, wspace=0.58, left=0.080, right=0.992,
                          top=0.895, bottom=0.270)
    x = np.arange(4)
    xt = [LAB[m] for m in MATERIALS]

    # ---- (a) Grueneisen parameters ------------------------------------
    ax = fig.add_subplot(gs[0])
    gE = [mats[m].gE for m in MATERIALS]
    gA = [mats[m].gA for m in MATERIALS]
    gL = [mats[m].dft.get('gamma_LA', np.nan) for m in MATERIALS]
    ax.bar(x - 0.26, gE, 0.25, color=CAT[0], lw=0, label=r"$E^{\prime}$")
    ax.bar(x, gA, 0.25, color=CAT[1], lw=0, label=r"$A_1^{\prime}$")
    ax.bar(x + 0.26, gL, 0.25, color=CAT[2], lw=0, label='LA(M)')
    mb = D.MICHAIL_BIAX
    # Direct biaxial-strain measurements exist for both optical modes of
    # MoS2 and for the A1' and the 2LA overtone of WSe2 (Michail 2024);
    # nothing is plotted where no measurement exists.
    pts = [(0 - 0.26, mb['MoS2']['gE'], mb['MoS2']['gE_e']),
           (0.0, mb['MoS2']['gA'], mb['MoS2']['gA_e']),
           (3.0, mb['WSe2']['gA'], mb['WSe2']['gA_e']),
           (3 + 0.26, mb['WSe2']['gLA'], mb['WSe2']['gLA_e'])]
    for px, py, pe in pts:
        ax.errorbar([px], [py], yerr=[pe], fmt='o', ms=3.3, color=INK,
                    mfc='white', mew=0.9, lw=0.9, zorder=6)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker='o', ls='', ms=3.3, color=INK,
                    mfc='white', mew=0.9))
    l.append('measured')
    ax.set_xticks(x)
    ax.set_xticklabels(xt, rotation=30, ha='right')
    ax.set_ylabel(r'Grüneisen parameter $\gamma$')
    ax.set_ylim(0, 4.6)
    ax.legend(h, l, loc='upper right', bbox_to_anchor=(0.99, 0.99),
              handlelength=0.85, handletextpad=0.35, labelspacing=0.18)
    _panel(ax, '(a)')

    # ---- (b) gap deformation potential vs exciton gauge factor ---------
    ax = fig.add_subplot(gs[1])
    lim = np.array([40, 200])
    ax.plot(lim, lim, color=MUTED, ls='--', lw=0.8, zorder=1)
    off = {'MoS2': (8, -4), 'WS2': (7, 3), 'MoSe2': (8, -4),
           'WSe2': (-8, 2)}
    ha = {'MoS2': 'left', 'WS2': 'left', 'MoSe2': 'left', 'WSe2': 'right'}
    for m in MATERIALS:
        d = mats[m].dft.get('dgap_deps')
        g, ge = D.GAUGE[m]
        if d is None:
            continue
        ax.errorbar([abs(d)], [abs(g)], yerr=[ge], fmt='o', ms=5.0,
                    color=CMAT[m], mec=INK, mew=0.7, lw=0.9, zorder=6)
        ax.annotate(LAB[m], (abs(d), abs(g)), textcoords='offset points',
                    xytext=off[m], fontsize=FS_ANN, color=CMAT[m], ha=ha[m],
                    va='center')
    ax.set_xlabel(r'DFT $|{\rm d}E_{\rm g}/{\rm d}\varepsilon|$ (meV/%)')
    ax.set_ylabel(r'measured $|\Xi_{\rm A}|$ (meV/%)')
    ax.set_xlim(40, 200)
    ax.set_ylim(40, 200)
    _note(ax, 0.045, 0.96, 'dashed line:\nequality')
    _panel(ax, '(b)')

    # ---- (c) the two strain lever arms --------------------------------
    ax = fig.add_subplot(gs[2])
    kv = res['krayev']
    vals = [abs(kv['dwA_deps']), abs(kv['dw2LA_deps'])]
    ax.bar([0, 1], vals, 0.5, color=[CAT[0], CAT[1]], lw=0)
    ax.errorbar([0], [2.5], yerr=[0.3], fmt='o', ms=3.4, color=INK,
                mfc='white', mew=0.9, lw=0.9, zorder=6, label='measured')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$A_1^{\prime}$", '2LA(M)'])
    ax.set_ylabel(r'$|{\rm d}\omega/{\rm d}\varepsilon|$ (cm$^{-1}$ per %)')
    ax.set_ylim(0, max(vals) * 1.72)
    ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.99),
              handletextpad=0.35)
    _note(ax, 0.05, 0.82, 'computed lever arms\ndiffer by %.1f times'
          % (vals[1] / vals[0]))
    _panel(ax, '(c)')

    # ---- (d) disorder-activated Raman calibration constant ------------
    ax = fig.add_subplot(gs[3])
    C = [mats[m].C_A for m in MATERIALS]
    ax.bar(x, C, 0.52, color=[CMAT[m] for m in MATERIALS], lw=0)
    ax.errorbar([0], [D.MIGNUZZI['C_A']], yerr=[D.MIGNUZZI['C_A_err']],
                fmt='o', ms=3.6, color=INK, mfc='white', mew=1.0, lw=1.0,
                zorder=6)
    top = max(C)
    ax.set_ylim(0, top * 1.40)
    for i in range(4):
        ax.text(i, C[i] + top * 0.035, 'meas.' if i == 0 else 'pred.',
                ha='center', va='bottom', fontsize=5.9, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(xt, rotation=30, ha='right')
    ax.set_ylabel(r'$C_{\rm A}$ (nm$^2$)')
    _panel(ax, '(d)')

    fig.savefig(os.path.join(FIGS, 'fig1.pdf'))
    plt.close(fig)


# ===========================================================================
def fig2(res):
    """Figure 3: edge charge and its electrostatic signature."""
    mats = all_materials()
    rb = res['ribbons']
    fig = plt.figure(figsize=(FULL_W, 2.30))
    gs = fig.add_gridspec(1, 4, wspace=0.60, left=0.080, right=0.992,
                          top=0.895, bottom=0.250)

    # ---- (a) robustness of the assignment -----------------------------
    ax = fig.add_subplot(gs[0])
    kr = res['krayev_robust']
    g = np.array([d['gamma'] for d in kr])
    ne = np.array([d['edge_n'] for d in kr]) / 1e12
    ee = np.array([d['edge_eps'] for d in kr])
    ax.plot(g, ne, color=CAT[1], lw=1.5,
            label=r'charge ($10^{12}$ cm$^{-2}$)')
    ax.plot(g, ee * 10, color=CAT[0], lw=1.5, ls='--', label='strain (0.1 %)')
    ax.set_xlabel(r'assumed $\gamma_{\rm LA}$')
    ax.set_ylabel('recovered edge state')
    ax.set_xlim(0.3, 3.3)
    ax.set_ylim(-0.8, 5.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.99),
              handlelength=0.9, labelspacing=0.18, fontsize=5.9)
    # dotted lines: the value measured for WSe2 and the value computed for
    # MoS2; both are named in the caption rather than in the panel, which is
    # too narrow to carry two labels without a collision
    ax.axvline(D.MICHAIL_BIAX['WSe2']['gLA'], color=INK2, lw=0.7, ls=':')
    ax.axvline(mats['MoS2'].dft.get('gamma_LA', np.nan), color=INK2,
               lw=0.7, ls=':')
    _panel(ax, '(a)')

    # ---- (b) threshold shift versus width -----------------------------
    ax = fig.add_subplot(gs[1])
    W = np.geomspace(8, 1000, 200)
    names = {'SiO2_300nm': r'300 nm SiO$_2$', 'SiO2_90nm': r'90 nm SiO$_2$',
             'SiO2_30nm': r'30 nm SiO$_2$',
             'HfO2_EOT1p5': r'HfO$_2$, EOT 1.5 nm'}
    for i, (tag, cox) in enumerate(transport.COX.items()):
        ax.loglog(W, transport.threshold_shift(rb['sigma_line_cm'], W, cox),
                  color=CAT[i], lw=1.3, label=names[tag])
    ax.axhline(0.15, color=MUTED, ls=':', lw=0.9)
    dvt200 = transport.threshold_shift(rb['sigma_line_cm'], 200.0,
                                       transport.COX['SiO2_300nm'])
    ax.plot([200], [dvt200], '*', ms=8.5, color=INK, mfc='white', mew=0.9,
            zorder=6)
    ax.set_xlabel('Nanoribbon width (nm)')
    ax.set_ylabel(r'$\Delta V_{\rm T}$ from edge charge (V)')
    ax.set_ylim(2e-3, 5e3)
    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99),
              handlelength=0.9, labelspacing=0.16)
    ax.annotate('measured onset', xy=(200, dvt200), xytext=(0.965, 0.640),
                textcoords='axes fraction', ha='right', va='center',
                fontsize=FS_ANN, color=INK2,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=INK2,
                                shrinkA=2, shrinkB=4))
    _panel(ax, '(b)')

    # ---- (c) on-current against width ---------------------------------
    ax = fig.add_subplot(gs[2])
    # the self-consistent solve including the measured contact resistance,
    # which is what the text quotes; the contact is revisited separately as a
    # resolved barrier
    sc = res['self_consistent']['curves']
    Wl = np.array(sc['MoS2']['W'])
    for m in ['WS2', 'MoS2', 'WSe2']:
        ax.semilogx(Wl, np.array(sc[m]['I']), color=CMAT[m], lw=1.4,
                    label=LAB[m])
    # Only the high-kappa-gated measurements are shown, because the curves are
    # computed for that stack at the 1.5 V overdrive it is operated at.  The
    # back-gated devices of the same work sit on a 96 nm SiO2 oxide at an
    # overdrive of tens of volts that is not reported per device, so they
    # cannot be placed on this axis without assuming a bias.
    mk = {'MoS2': 'o', 'WS2': 's', 'WSe2': '^'}
    for name, mname, car, w, L, stack, meas in D.PENA['devices']:
        ax.plot([w], [meas], mk[mname], ms=4.4, color=CMAT[mname],
                mfc='white', mew=1.1, zorder=6)
    ax.set_xlabel('Nanoribbon width (nm)')
    ax.set_ylabel(r'$I_{\rm on}/W$ ($\mu$A $\mu$m$^{-1}$)')
    ax.set_xlim(9, 1000)
    ax.set_ylim(0, 700)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker='o', ls='', ms=4.0, color=INK,
                    mfc='white', mew=1.0))
    l.append('measured')
    ax.legend(h, l, loc='upper left', ncol=2, bbox_to_anchor=(0.01, 0.99),
              handlelength=0.85, columnspacing=0.6, labelspacing=0.16)
    _panel(ax, '(c)')

    # ---- (d) inverted defect density against measured mobility --------
    ax = fig.add_subplot(gs[3])
    rows = res['nattoo']['rows']
    nd = np.array([r['nd'] for r in rows])
    mu = np.array([r['mu_meas'] if r['mu_meas'] is not None else np.nan
                   for r in rows], dtype=float)
    ndx = np.geomspace(8e11, 8e14, 40)
    ax.loglog(ndx, [transport.sheet_mobility(mats['MoS2'], n)[0]
                    for n in ndx], color=CAT[0], lw=1.4,
              label='point-defect ceiling')
    ax.loglog(nd, mu, 'o', ms=4.6, color=INK, mfc='white', mew=1.0, zorder=6,
              ls='', label='measured films')
    lbl = {0: ('ALD as-grown', (7, -6), 'left'),
           1: ('ALD annealed', (7, 6), 'left'),
           2: ('sputtered', (7, -2), 'left')}
    for i, (xx, yy) in enumerate(zip(nd, mu)):
        if i in lbl and np.isfinite(yy):
            t, o, hh = lbl[i]
            ax.annotate(t, (xx, yy), textcoords='offset points', xytext=o,
                        fontsize=5.9, color=INK2, ha=hh, va='center')
    ax.annotate('', xy=(nd[1] * 1.9, mu[1] * 1.6), xytext=(nd[1] * 1.9, 24.0),
                arrowprops=dict(arrowstyle='<->', lw=0.7, color=MUTED))
    _note(ax, 0.020, 0.60, 'grain\nboundary\ndeficit')
    ax.set_xlabel(r'inverted $n_{\rm d}$ (cm$^{-2}$)')
    ax.set_ylabel(r'$\mu$ (cm$^2$V$^{-1}$s$^{-1}$)')
    ax.set_xlim(8e11, 8e14)
    ax.set_ylim(8e-3, 4e3)
    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99),
              handlelength=0.9, labelspacing=0.16)
    _panel(ax, '(d)')

    fig.savefig(os.path.join(FIGS, 'fig2.pdf'))
    plt.close(fig)


# ===========================================================================
def fig3(res):
    """Figure 4: what sets the width limit."""
    mats = all_materials()
    rb = res['ribbons']
    fig = plt.figure(figsize=(FULL_W, 2.36))
    gs = fig.add_gridspec(1, 4, wspace=0.80, left=0.082, right=0.938,
                          top=0.890, bottom=0.245)

    # ---- (a) critical width against damage halo -----------------------
    ax = fig.add_subplot(gs[0])
    halos = np.array(rb['halos'])
    names = {'SiO2_300nm': r'300 nm SiO$_2$', 'SiO2_90nm': r'90 nm SiO$_2$',
             'SiO2_30nm': r'30 nm SiO$_2$', 'HfO2_EOT1p5': r'HfO$_2$'}
    for i, (tag, vals) in enumerate(rb['Wc_halo'].items()):
        ax.loglog(halos, vals, color=CAT[i], lw=1.3, label=names[tag])
    # the two patterning processes are marked with short tags and spelled out
    # in the caption, which keeps the labels clear of the curves
    for xh, txt, off, hh in ((D.HALO['GENTLE_nm'], r'XeF$_2$', (5.5, -7.0), 'left'),
                             (D.HALO['HIM_nm'], 'HIM', (-2.5, -15.0), 'right')):
        yv = rb['halo']['GENTLE' if xh < 50 else 'HIM']['Wc']
        ax.plot(xh, yv, 'o', ms=4.6, color=INK, mfc='white', mew=1.0,
                zorder=6)
        ax.annotate(txt, xy=(xh, yv), xytext=off,
                    textcoords='offset points', ha=hh, va='top',
                    fontsize=FS_ANN, color=INK)
    ax.set_xlabel('Damage halo width (nm)')
    ax.set_ylabel(r'Critical width $W_{\rm c}$ (nm)')
    ax.set_xlim(0.9, 260)
    ax.set_ylim(3, 3e4)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.99),
              handlelength=0.9, labelspacing=0.16, title='gate stack',
              title_fontsize=FS_LEG)
    _panel(ax, '(a)')

    # ---- (b) design map -----------------------------------------------
    ax = fig.add_subplot(gs[1])
    Wg = np.array(rb['map']['W'])
    ndg = np.array(rb['map']['nd'])
    Z = np.array(rb['map']['I'])
    pcm = ax.pcolormesh(Wg, ndg, Z, cmap=SEQ, shading='gouraud')
    cs = ax.contour(Wg, ndg, Z, levels=[100, 200, 300, 400], colors=['white'],
                    linewidths=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(Wg[0], Wg[-1])
    ax.set_ylim(ndg[0], ndg[-1])
    # Contour labels are forced into the left third of the panel and the
    # growth-technology names into the right third, so the two label sets
    # cannot collide with each other or with the colour bar.
    x_lab = Wg[0] * (Wg[-1] / Wg[0]) ** 0.17
    ylo = ndg[0] * (ndg[-1] / ndg[0]) ** 0.06
    yhi = ndg[0] * (ndg[-1] / ndg[0]) ** 0.94
    man = []
    for segs in cs.allsegs:
        pts = [s for s in segs if len(s)]
        if not pts:
            man.append(None)
            continue
        P = np.vstack(pts)
        ok = (P[:, 1] > ylo) & (P[:, 1] < yhi)
        P = P[ok] if ok.any() else P
        j = int(np.argmin(np.abs(np.log(P[:, 0]) - np.log(x_lab))))
        man.append((float(P[j, 0]), float(P[j, 1])))
    man = [m for m in man if m is not None]
    if man:
        ax.clabel(cs, fontsize=5.4, fmt='%d', inline_spacing=1, manual=man)
    short = {'exfoliated': 'exfoliated', 'SS-CVD monolayer': 'SS-CVD',
             'ALD annealed': 'ALD ann.', 'ALD as-grown': 'ALD as-gr.',
             'sputtered': 'sputtered', 'MOCVD WSe2': 'MOCVD'}
    items = sorted(rb['tech'].items(), key=lambda kv: kv[1])
    lo, hi = np.log10(ndg[0]), np.log10(ndg[-1])
    pos = [np.log10(v) for _, v in items]
    gapmin = 0.085 * (hi - lo)                 # one label height
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gapmin)
    shift = max(0.0, pos[-1] - (hi - 0.035 * (hi - lo)))
    pos = [p - shift for p in pos]
    stroke = [pe.withStroke(linewidth=1.5, foreground='white')]
    for (lab, v), p in zip(items, pos):
        ax.axhline(v, color=INK, lw=0.45, alpha=0.45)
        yv = 10.0 ** p
        ax.plot([Wg[-1] * 0.965, Wg[-1] * 0.965], [v, yv], color=INK,
                lw=0.45, alpha=0.45)
        ax.text(Wg[-1] * 0.925, yv, short.get(lab, lab), fontsize=5.4,
                color=INK, ha='right', va='center', zorder=8,
                path_effects=stroke)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.050, pad=0.045)
    cb.ax.set_title(r'$I_{\rm on}/W$' '\n' r'($\mu$A/$\mu$m)', fontsize=6.0,
                    pad=2.5, color=INK, loc='left')
    cb.ax.tick_params(labelsize=6.0)
    cb.outline.set_linewidth(0.5)
    ax.set_xlabel('Nanoribbon width (nm)')
    ax.set_ylabel(r'$n_{\rm d}$ (cm$^{-2}$)')
    _panel(ax, '(b)')

    # ---- (c) family screening at 25 nm --------------------------------
    ax = fig.add_subplot(gs[2])
    x = np.arange(4)
    nds = [1e12, 1e13, 3e13]
    labs = [r'$10^{12}$', r'$10^{13}$', r'$3\times10^{13}$']
    st = transport.STACK['HfO2_EOT1p5']
    base = dict(halo_nm=D.HALO['GENTLE_nm'], sigma_line_cm=rb['sigma_line_cm'],
                Cox=transport.COX['HfO2_EOT1p5'], Vov=rb['Vov'], Vds=1.0,
                Lch_nm=300.0, n_it_cm2=st['nit'], eps_env=st['eps'])
    top = 0.0
    for j, ndv in enumerate(nds):
        vals = []
        for mname in MATERIALS:
            car = 'h' if mname == 'WSe2' else 'e'
            vals.append(float(transport.ribbon_current_density_uA_um(
                mats[mname], 25.0, ndv, carrier=car, **base)))
        top = max(top, max(vals))
        ax.bar(x + (j - 1) * 0.27, vals, 0.25, color=CAT[j], lw=0,
               label=labs[j])
    ax.set_xticks(x)
    ax.set_xticklabels([LAB[m] for m in MATERIALS], rotation=30, ha='right')
    ax.set_ylabel(r'$I_{\rm on}/W$ at 25 nm ($\mu$A/$\mu$m)',
                  labelpad=1.5)
    ax.set_ylim(0, top * 1.50)
    ax.legend(title=r'$n_{\rm d}$ (cm$^{-2}$)', title_fontsize=FS_LEG,
              loc='upper right', bbox_to_anchor=(0.99, 0.99),
              handlelength=0.9, labelspacing=0.16)
    _panel(ax, '(c)')

    # ---- (d) critical width of the four materials ---------------------
    ax = fig.add_subplot(gs[3])
    wc = [rb['Wc'][m] for m in MATERIALS]
    ax.bar(np.arange(4), wc, 0.52, color=[CMAT[m] for m in MATERIALS], lw=0)
    ax.axhline(25.0, color=INK2, ls='--', lw=0.9)
    _note(ax, 0.94, 0.90, 'narrowest ribbon\ndemonstrated', ha='right')
    for i, v in enumerate(wc):
        ax.text(i, v + 0.5, '%.0f' % v, ha='center', va='bottom',
                fontsize=6.0, color=INK2)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([LAB[m] for m in MATERIALS], rotation=30, ha='right')
    ax.set_ylabel(r'Critical width $W_{\rm c}$ (nm)')
    ax.set_ylim(0, max(max(wc), 25) * 1.75)
    _panel(ax, '(d)')

    fig.savefig(os.path.join(FIGS, 'fig3.pdf'))
    plt.close(fig)


def main():
    res = _load()
    fig_abstract(res)
    fig1(res)
    fig2(res)
    fig3(res)
    print('figures written to', FIGS)


if __name__ == '__main__':
    main()
