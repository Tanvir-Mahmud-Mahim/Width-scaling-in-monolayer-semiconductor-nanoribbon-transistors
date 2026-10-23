"""Supplementary figures and tables."""
from __future__ import annotations

import json
import os

import numpy as np

from .figures import (CAT, CMAT, DIV, FIGS, INK, INK2, LAB, MUTED, SEQ,
                      FULL_W, ROOT, _panel, plt)
from . import adjoint, spectra
from . import datasets as D, transport
from .materials import all_materials, MATERIALS

DFT = os.path.join(ROOT, 'dft')
COL = 3.35


def _res():
    with open(os.path.join(ROOT, 'results.json')) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
def figS1():
    """Convergence of the first-principles calculations."""
    path = os.path.join(DFT, 'conv.json')
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.98))
    fig.subplots_adjust(wspace=0.44, left=0.078, right=0.99, bottom=0.235,
                        top=0.865)
    if not os.path.exists(path):
        for ax in axes:
            ax.text(0.5, 0.5, 'convergence run pending', ha='center',
                    transform=ax.transAxes, color=INK2)
        fig.savefig(os.path.join(FIGS, 'figS1.pdf'))
        plt.close(fig)
        return
    c = json.load(open(path))
    ke = sorted([v for k, v in c.items() if k.startswith('ke')],
                key=lambda r: r['ke'])
    nk = sorted([v for k, v in c.items() if k.startswith('nk')],
                key=lambda r: r['nk'])
    vac = sorted([v for k, v in c.items() if k.startswith('vac')],
                 key=lambda r: r['vac'])
    if not (len(ke) >= 2 and len(nk) >= 2 and len(vac) >= 2):
        for ax in axes:
            ax.text(0.5, 0.5, 'convergence run incomplete', ha='center',
                    transform=ax.transAxes, color=INK2, fontsize=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.savefig(os.path.join(FIGS, 'figS1.pdf'))
        plt.close(fig)
        return
    ref = [r for r in ke if r['ke'] == 100.0]
    E0 = ref[0]['E'] if ref else ke[-1]['E']
    ax = axes[0]
    ax.plot([r['ke'] for r in ke], [1000 * (r['E'] - E0) / 3 for r in ke],
            'o-', ms=3.5, color=CAT[0], mfc='white')
    ax.set_xlabel('Density cutoff (Ha)')
    ax.set_ylabel('Energy per atom (meV)')
    ax.axvline(60, color=MUTED, ls='--', lw=0.7)
    _panel(ax, '(a)')
    ax = axes[1]
    e0 = [r for r in nk if r['nk'] == 6]
    e0 = e0[0]['E'] if e0 else nk[-1]['E']
    ax.plot([r['nk'] for r in nk], [1000 * (r['E'] - e0) / 3 for r in nk],
            'o-', ms=3.5, color=CAT[1], mfc='white')
    ax.set_xlabel(r'$k$-mesh, $N\times N\times1$')
    ax.set_ylabel('Energy per atom (meV)')
    ax.axvline(4, color=MUTED, ls='--', lw=0.7)
    _panel(ax, '(b)')
    ax = axes[2]
    v0 = vac[-1]['E']
    ax.plot([r['vac'] for r in vac], [1000 * (r['E'] - v0) / 3 for r in vac],
            'o-', ms=3.5, color=CAT[2], mfc='white')
    ax.set_xlabel(r'Cell height $c$ (Å)')
    ax.set_ylabel('Energy per atom (meV)')
    ax.axvline(15, color=MUTED, ls='--', lw=0.7)
    _panel(ax, '(c)')
    fig.savefig(os.path.join(FIGS, 'figS1.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
def figS2():
    """Strain sweep and frozen-phonon detail."""
    A = json.load(open(os.path.join(DFT, 'stageA.json')))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.02))
    fig.subplots_adjust(wspace=0.44, left=0.078, right=0.99, bottom=0.265,
                        top=0.870)
    S = [-0.02, -0.01, 0.0, 0.01, 0.02]
    ax = axes[0]
    for m in MATERIALS:
        E, s = [], []
        for x in S:
            k = '%s_e%+.3f' % (m, x)
            if k in A and A[k].get('conv'):
                s.append(x * 100)
                E.append(A[k]['E'])
        if len(E) >= 3:
            E = np.array(E) - np.interp(0.0, s, E)
            ax.plot(s, 1000 * E, 'o-', ms=3.0, color=CMAT[m], mfc='white',
                    label=LAB[m])
    ax.set_xlabel('Biaxial strain (%)')
    ax.set_ylabel('Energy per cell (meV)')
    ax.legend(handlelength=1.0, labelspacing=0.2, loc='upper center')
    _panel(ax, '(a)')

    ax = axes[1]
    mats = all_materials()
    for m in MATERIALS:
        gap, s = [], []
        for x in S:
            k = '%s_e%+.3f' % (m, x)
            if k in A and A[k].get('bands'):
                b = A[k]['bands']
                s.append(x * 100)
                gap.append(b['K']['cb'] - b['K']['vb'])
        if len(gap) >= 3:
            ax.plot(s, gap, 'o-', ms=3.0, color=CMAT[m], mfc='white')
    ax.set_xlabel('Biaxial strain (%)')
    ax.set_ylabel(r'$K$-point gap (eV)')
    _panel(ax, '(b)')

    ax = axes[2]
    w = np.arange(4)
    dfts = [mats[m].dft for m in MATERIALS]
    wE = [d.get('wE_dft', np.nan) for d in dfts]
    wA = [d.get('wA_dft', np.nan) for d in dfts]
    from .materials import PHONON
    ax.bar(w - 0.19, wE, 0.33, color=CAT[0], lw=0, label=r"$E'$ (DFT)")
    ax.bar(w + 0.19, wA, 0.33, color=CAT[1], lw=0, label=r"$A_1'$ (DFT)")
    ax.plot(w - 0.19, [PHONON[m]['wE'] for m in MATERIALS], 'o', ms=3.4,
            color=INK, mfc='white', mew=0.9, label='experiment')
    ax.plot(w + 0.19, [PHONON[m]['wA'] for m in MATERIALS], 'o', ms=3.4,
            color=INK, mfc='white', mew=0.9)
    ax.set_xticks(w)
    ax.set_xticklabels([LAB[m] for m in MATERIALS], rotation=30, ha='right')
    ax.set_ylabel(r'Frequency (cm$^{-1}$)')
    ax.set_ylim(0, 640)
    ax.legend(handlelength=1.0, labelspacing=0.2, loc='upper left',
              bbox_to_anchor=(0.02, 0.99), ncol=2, columnspacing=0.7,
              handletextpad=0.3)
    _panel(ax, '(c)')
    fig.savefig(os.path.join(FIGS, 'figS2.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_adjoint():
    """Adjoint inversion: conditioning and synthetic map recovery.

    Written as figS5.pdf because it is the fifth supplementary figure in
    reading order."""
    mats = all_materials()
    m = mats['MoS2']
    from .analysis import SIGMA
    res = _res()
    fig = plt.figure(figsize=(FULL_W, 3.60))
    gs = fig.add_gridspec(2, 4, hspace=0.72, wspace=0.60, left=0.072,
                          right=0.982, top=0.905, bottom=0.095)

    th = np.array([12.7, -0.3, 0.5])
    r = adjoint.single_point(spectra.forward(th, m)[0], SIGMA, m)
    C = r['cov']
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    lbl = [r'$\log_{10}n_{\rm d}$', r'$\varepsilon$', r'$n$']
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(R, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(lbl, rotation=25, ha='right')
    ax.set_yticks(range(3))
    ax.set_yticklabels(lbl)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, '%.2f' % R[i, j], ha='center', va='center',
                    fontsize=6.2,
                    color='white' if abs(R[i, j]) > 0.6 else INK)
    ax.set_title('posterior correlation', fontsize=6.8, pad=3)
    _panel(ax, '(a)', dy=1.19)

    ax = fig.add_subplot(gs[0, 1])
    c = res['conditioning']['MoS2']
    xs = np.arange(3)
    sets = [('all four channels', c['sigma_post'], CAT[0]),
            ('no exciton channel', c['sigma_post_no_exciton'], CAT[1]),
            ('no LA(M) channel', c['sigma_post_no_defect_channel'], CAT[4])]
    for i, (lab, v, col) in enumerate(sets):
        ax.bar(xs + (i - 1) * 0.27, v, 0.25, color=col, lw=0, label=lab)
    ax.set_yscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels(lbl)
    ax.set_ylabel('posterior std. dev.')
    ax.set_ylim(8e-3, 3e2)
    ax.legend(loc='upper center', handlelength=0.9, borderpad=0.1,
              labelspacing=0.22, bbox_to_anchor=(0.52, 0.99), fontsize=5.9)
    _panel(ax, '(b)')

    ax = fig.add_subplot(gs[0, 2])
    scales = np.geomspace(0.2, 10.0, 20)
    for k, (lab, col) in enumerate(zip(lbl, [CAT[0], CAT[1], CAT[2]])):
        vals = [adjoint.single_point(spectra.forward(th, m)[0], SIGMA * sc,
                                     m)['sigma_post'][k] for sc in scales]
        ax.loglog(scales, vals, color=col, lw=1.3, label=lab)
    ax.set_xlabel('measurement noise (relative)')
    ax.set_ylabel('posterior std. dev.')
    ax.legend(handlelength=1.0, labelspacing=0.2, loc='upper left',
              bbox_to_anchor=(0.01, 0.99))
    _panel(ax, '(c)')

    ax = fig.add_subplot(gs[0, 3])
    J = np.array(res['map_recovery']['J'])
    ax.semilogy(np.arange(len(J)), J / J[0], 'o-', ms=3.0, color=CAT[0],
                mfc='white', mew=0.9)
    ax.set_xlabel('Gauss-Newton iteration')
    ax.set_ylabel(r'$J/J_0$')
    ax.grid(True, alpha=0.6, lw=0.4)
    _panel(ax, '(d)')

    d = np.load(os.path.join(ROOT, 'map_recovery.npz'))
    t, rr = d['true'], d['rec']
    names = [r'$\log_{10}n_{\rm d}$ (cm$^{-2}$)', r'$\varepsilon$ (%)',
             r'$n$ ($10^{13}$ cm$^{-2}$)']
    cmaps = [SEQ, DIV, SEQ]
    for k in range(3):
        axt = fig.add_subplot(gs[1, k])
        vmin = min(t[..., k].min(), rr[..., k].min())
        vmax = max(t[..., k].max(), rr[..., k].max())
        if k == 1:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        pad = np.full((t.shape[0], 3), np.nan)
        comb = np.concatenate([t[..., k], pad, rr[..., k]], axis=1)
        im = axt.imshow(comb, cmap=cmaps[k], origin='lower', vmin=vmin,
                        vmax=vmax, aspect='auto')
        cb = fig.colorbar(im, ax=axt, fraction=0.040, pad=0.025)
        cb.ax.tick_params(labelsize=6.0)
        axt.set_xticks([])
        axt.set_yticks([])
        axt.set_title(names[k], fontsize=6.7, pad=2.5)
        axt.text(0.24, -0.085, 'true', transform=axt.transAxes, fontsize=6.3,
                 ha='center', color=INK2)
        axt.text(0.77, -0.085, 'recovered', transform=axt.transAxes,
                 fontsize=6.3, ha='center', color=INK2)
        _panel(axt, '(%s)' % 'efg'[k], dy=1.19)

    ax = fig.add_subplot(gs[1, 3])
    for k, col in enumerate([CAT[0], CAT[1], CAT[2]]):
        e = (rr[..., k] - t[..., k]).ravel()
        ax.hist(e, bins=45, histtype='step', color=col, lw=1.0, density=True,
                label=lbl[k])
    ax.set_xlabel('recovery error')
    ax.set_ylabel('probability density')
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0, 50)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), handlelength=1.0,
              labelspacing=0.18, fontsize=5.9)
    rm = res['map_recovery']['rmse']
    ax.text(0.94, 0.97, 'RMSE\n'
            r'$n_{\rm d}$: %.3f dex' '\n' r'$\varepsilon$: %.3f %%' '\n'
            r'$n$: %.3f$\times10^{13}$' % tuple(rm), transform=ax.transAxes, fontsize=5.9,
            va='top', ha='right', color=INK2)
    _panel(ax, '(h)')
    fig.savefig(os.path.join(FIGS, 'figS5.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_transport():
    """Transport channel decomposition and nanoribbon model sensitivity.

    Written as figS3.pdf because it is the third supplementary figure in
    reading order."""
    mats = all_materials()
    m = mats['MoS2']
    res = _res()
    rb = res['ribbons']
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.04))
    fig.subplots_adjust(wspace=0.50, left=0.070, right=0.99, bottom=0.235,
                        top=0.865)

    ax = axes[0]
    nd = np.geomspace(1e11, 1e14, 70)
    tot, ph, pd, ci = [], [], [], []
    for n in nd:
        # the same Boltzmann integral the reported currents use, so the
        # decomposition and the device numbers come from one calculation
        t, parts, _ = transport.sheet_mobility_energy_resolved(m, n)
        tot.append(t)
        ph.append(parts['ph'])
        pd.append(parts['pd'])
        ci.append(parts['ci'])
    ax.loglog(nd, ph, color=CAT[0], lw=1.2, ls='--', label='phonon')
    ax.loglog(nd, pd, color=CAT[1], lw=1.2, ls='-.', label='point defect')
    ax.loglog(nd, ci, color=CAT[2], lw=1.2, ls=':', label='interface charge')
    ax.loglog(nd, tot, color=INK, lw=1.5, label='total')
    ax.set_xlabel(r'$n_{\rm d}$ (cm$^{-2}$)')
    ax.set_ylabel(r'$\mu$ (cm$^2$V$^{-1}$s$^{-1}$)')
    ax.set_ylim(1, 3e3)
    ax.legend(loc='lower left', bbox_to_anchor=(0.01, -0.01),
              handlelength=2.2, labelspacing=0.18, fontsize=6.0)
    _panel(ax, '(a)')

    ax = axes[1]
    W = np.geomspace(5, 1000, 120)
    for j, h in enumerate([1.0, 5.0, 20.0, 150.0]):
        mu = transport.ribbon_mobility(m, W, 1.3e12, halo_nm=h)
        ax.semilogx(W, mu, color=CAT[j], lw=1.3, label='halo %g nm' % h)
    ax.set_xlabel('Nanoribbon width (nm)')
    ax.set_ylabel(r'$\mu$ (cm$^2$V$^{-1}$s$^{-1}$)')
    ax.set_ylim(10, 64)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=2,
              handlelength=1.1, labelspacing=0.18, columnspacing=0.7,
              fontsize=6.0)
    _panel(ax, '(b)')

    ax = axes[2]
    # below about 0.35 the widest halo crosses on the plateau where the
    # ribbon is damaged edge to edge, so W_c is not meaningful there
    fr = np.linspace(0.4, 0.8, 18)
    st = transport.STACK['HfO2_EOT1p5']
    for j, h in enumerate([5.0, 20.0, 150.0]):
        vals = [transport.critical_width(
            m, 1.3e12, frac=f, halo_nm=h,
            sigma_line_cm=rb['sigma_line_cm'],
            Cox=transport.COX['HfO2_EOT1p5'], Vov=1.5, Lch_nm=300.0,
            n_it_cm2=st['nit'], eps_env=st['eps']) for f in fr]
        ax.semilogy(fr, vals, color=CAT[j], lw=1.3, label='halo %g nm' % h)
    ax.set_xlabel(r'criterion $I/I_{\rm wide}$')
    ax.set_ylabel(r'$W_{\rm c}$ (nm)')
    ax.set_ylim(8, 2e4)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=1,
              handlelength=1.1, labelspacing=0.16, fontsize=5.9)
    _panel(ax, '(c)')

    # ---- (d) compact expression against the self-consistent solve ------
    ax = axes[3]
    sc = res['self_consistent']
    cmp_ = sc['compare']
    Wv = np.array(cmp_['W'])
    ax.semilogx(Wv, cmp_['I_compact'], color=MUTED, lw=1.3, ls='--',
                label='compact')
    ax.semilogx(Wv, cmp_['I_sc'], color=CAT[0], lw=1.5,
                label='self-consistent')
    ax.set_xlabel('Nanoribbon width (nm)')
    ax.set_ylabel(r'$I_{\rm on}/W$ ($\mu$A/$\mu$m)')
    ax.set_ylim(0, max(cmp_['I_compact']) * 1.80)
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.01), handlelength=2.0,
              labelspacing=0.18, fontsize=6.0)
    ax.text(0.97, 0.97, 'current ratio %.2f\nself-consistent: $W_{\\rm c}$ = %.1f nm\ncompact: $W_{\\rm c}$ = %.1f nm'
            % (cmp_['ratio_mean'], sc['Wc_self_consistent'],
               sc['Wc_compact']),
            transform=ax.transAxes, fontsize=5.9, color=INK2, ha='right',
            va='top')
    _panel(ax, '(d)')

    fig.savefig(os.path.join(FIGS, 'figS3.pdf'))
    plt.close(fig)


def tables():
    """LaTeX tables for the supplement."""
    mats = all_materials()
    res = _res()
    out = []

    rows = []
    for m in MATERIALS:
        mm = mats[m]
        rows.append(r'%s & %.3f & %.0f & %.0f & %.2f & %.2f & %.2f & %.0f \\'
                    % (LAB[m].replace('$_2$', r'$_2$'), mm.a0,
                       mm.dft.get('wE_dft', float('nan')),
                       mm.dft.get('wA_dft', float('nan')),
                       mm.gE, mm.gA,
                       mm.dft.get('gamma_LA', float('nan')), mm.C2D))
    out.append(r"""\begin{table*}[t]
\caption{First-principles constants computed in this work. $a_0$ is the
in-plane lattice constant used (\AA), $\omega_{E'}$ and $\omega_{A_1'}$ are the
frozen-phonon zone-centre frequencies (cm$^{-1}$), $\gamma_{E'}$ and
$\gamma_{A_1'}$ the corresponding mode Gr\"uneisen parameters,
$\gamma_{\rm LA}$ the acoustic Gr\"uneisen parameter obtained from the strain
dependence of the elastic constant, and $C_{2D}\equiv C_{11}$ the in-plane elastic constant
(N/m).  The lattice constant is an input; every other entry is computed in
this work and none is fitted.}
\label{tab:dft}
\begin{ruledtabular}
\begin{tabular}{lccccccc}
Material & $a_0$ & $\omega_{E'}$ & $\omega_{A_1'}$ & $\gamma_{E'}$ &
$\gamma_{A_1'}$ & $\gamma_{\rm LA}$ & $C_{2D}$ \\
\hline
""" + '\n'.join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}""")

    rows = []
    for row in res['nattoo']['rows']:
        mu = row['mu_meas']
        mus = ('%.2f' % mu) if mu == mu else 'n/a'
        rows.append(r'%s & %.3f & %.3f & $%.1f\times10^{%d}$ & %.1f & %s & %.0f \\'
                    % (row['label'], row['R_sh'], row['R_LA'],
                       row['nd'] / 10 ** int(np.floor(np.log10(row['nd']))),
                       int(np.floor(np.log10(row['nd']))), row['LD_nm'], mus,
                       row['mu_ceiling']))
    out.append(r"""\begin{table*}[t]
\caption{Inversion of the published atomic-layer-deposited and sputtered
MoS$_2$ films. $I_{\rm sh}/I_{E'}$ and $I_{\rm LA}/I_{A_1'}$ are the two
disorder-activated Raman intensity ratios reported in
Ref.~\onlinecite{Nattoo2025}, $n_{\rm d}$ (cm$^{-2}$) and $L_{\rm D}$ (nm) the inverted defect
density and mean inter-defect distance, and the last two columns the measured
and point-defect-limited mobility (cm$^2$V$^{-1}$s$^{-1}$).}
\label{tab:nattoo}
\begin{ruledtabular}
\begin{tabular}{lcccccc}
Film & $I_{\rm sh}/I_{E'}$ & $I_{\rm LA}/I_{A_1'}$ & $n_{\rm d}$ & $L_{\rm D}$ &
$\mu_{\rm meas}$ & $\mu_{\rm ceiling}$ \\
\hline
""" + '\n'.join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}""")

    from .materials import PROVENANCE
    kind_of = {'dft': 'this work', 'lit': 'literature',
               'cal': 'calibrated once', 'geom': 'geometry',
               'assumed': 'assumed', 'unused': 'not used'}
    rows = []
    extra = [
        ('EPS_MONO, T_MONO', 'literature',
         'monolayer out-of-plane permittivity and thickness, '
         'set the contact screening length'),
        ('barrier profile', 'assumed',
         'band edge relaxes exponentially from the contact over the '
         'screening length'),
    ]
    for k, v in PROVENANCE.items():
        tag, _, src = v.partition(': ')
        kind = kind_of.get(tag, tag)
        if src == 'this work':
            src = 'frozen phonons and biaxial strain sweep, this work'
        rows.append((k, kind, src))
    rows.extend(extra)
    body = []
    for k, kind, src in rows:
        key = k.replace('_', r'\_')
        # The two wide columns wrap inside a \parbox rather than through an
        # array-package p-column.  REVTeX 4-2 patches the array internals and
        # refuses to do so for array releases it does not recognise, which
        # makes >{...}p{...} specifiers fail outright on current TeX
        # distributions.  \parbox is plain LaTeX and cannot break that way.
        body.append(r'\provkey{%s} & %s & \provsrc{%s} \\' % (key, kind, src))
    out.append(r"""\begin{table*}[t]
\caption{Provenance of every group of constants used in this work.  Entries
marked ``this work'' are computed here from first principles; ``calibrated
once'' means the value was fixed a single time against the cited published
result and then held for every number reported; ``geometry'' means a nominal
device dimension; ``assumed'' marks a modelling choice stated in the text; and
``not used'' marks a constant that is present in the code but multiplies zero
in every result reported here.}
\label{tab:prov}
\providecommand{\provkey}[1]{\parbox[t]{0.26\textwidth}{\raggedright #1}}
\providecommand{\provsrc}[1]{\parbox[t]{0.44\textwidth}{\raggedright #1}}
\begin{ruledtabular}
\begin{tabular}{@{}lll@{}}
Constants & Origin & Source \\
\hline
""" + '\n'.join(body) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}""")

    with open(os.path.join(ROOT, 'paper', 'tables.tex'), 'w') as fh:
        fh.write('\n\n'.join(out) + '\n')

    # rows of the digitised threshold-voltage table (Pena et al. 2026)
    rows = []
    for w in (850, 75, 25):
        pts = [(L, V) for ww, L, V in D.PENA_VT if ww == w]
        Ls = sorted(set(L for L, _ in pts), reverse=True)
        cells = []
        for L in Ls:
            vs = ', '.join('$%+.2f$' % V for LL, V in pts if LL == L)
            cells.append('%.2f: %s' % (L, vs))
        rows.append(r'$W\approx%d$~nm & %s \\' % (w, '; '.join(cells)))
    with open(os.path.join(ROOT, 'paper', 'penavt.tex'), 'w') as fh:
        fh.write('% generated by rapid/supplement.py -- do not edit\n')
        fh.write('\n'.join(rows) + '\n')


def fig_quantum():
    """Quantum transport: mobility kernel, ballisticity, barrier, contact.

    Written as figS4.pdf, the fourth supplementary figure in reading order."""
    mats = all_materials()
    res = _res()
    er, q = res['energy_resolved'], res['quantum']
    st = transport.STACK['HfO2_EOT1p5']
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.04))
    fig.subplots_adjust(wspace=0.50, left=0.070, right=0.99, bottom=0.235,
                        top=0.865)

    # ---- (a) the energy-resolved relaxation time --------------------------
    ax = axes[0]
    ker = transport.transport_kernels(mats['MoS2'], 1.3e12, 0.0, 1.9e13,
                                      st['nit'], 'e', st['eps'], W_nm=25.0)
    kT = transport.KB * transport.T300 / transport.QE
    x = ker['E'] / transport.QE / kT
    # the phonon and point-defect times are nearly equal at this defect
    # density, so they are drawn with different dashes to stay legible
    for key, lab, c, ls in (('ph', 'phonon', CAT[0], '-'),
                            ('pd', 'point defect', CAT[1], (0, (4, 2))),
                            ('ci', 'interface charge', CAT[2], '-'),
                            ('ed', 'edge, 25 nm', CAT[3], '-')):
        ax.semilogy(x, np.asarray(ker['tau_channels'][key]) * 1e15,
                    color=c, lw=1.3, ls=ls, label=lab)
    ax.semilogy(x, ker['tau'] * 1e15, color=INK, lw=1.6, label='total')
    ax.set_xlabel(r'$(E - E_{\rm c}) / k_{\rm B}T$')
    ax.set_ylabel(r'$\tau$ (fs)')
    ax.set_xlim(0, 12)
    ax.set_ylim(1, 3e5)
    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=1,
              handlelength=1.6, labelspacing=0.12, fontsize=5.6)
    _panel(ax, '(a)')

    # ---- (b) ballistic to diffusive crossover -----------------------------
    ax = axes[1]
    for name in MATERIALS:
        b = q['ballistic'][name]
        ax.semilogx(b['L_nm'], b['T'], color=CMAT[name], lw=1.4,
                    label=LAB[name])
    ax.axvline(50.0, color=INK2, lw=0.7, ls=':')
    ax.axvline(300.0, color=INK2, lw=0.7, ls='--')
    ax.axhline(0.5, color=INK2, lw=0.7, ls=':')
    ax.set_xlabel(r'Channel length $L$ (nm)')
    ax.set_ylabel(r'channel transmission $\mathcal{T}$')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=1,
              handlelength=0.9, labelspacing=0.15, fontsize=5.8)
    _panel(ax, '(b)')

    # ---- (c) the resolved Schottky barrier and its transmission -----------
    ax = axes[2]
    pr = q['wse2']['profile']
    ax.plot(pr['x_nm'], pr['U_eV'], color=CAT[1], lw=1.5)
    ax.set_xlabel(r'Distance from contact (nm)')
    ax.set_ylabel(r'$E_{\rm v}$ (eV)', color=CAT[1])
    ax.tick_params(axis='y', colors=CAT[1])
    ax.set_xlim(0, 8)
    tc = q['wse2']['T_curve']
    axi = ax.inset_axes([0.44, 0.44, 0.53, 0.50])
    axi.semilogy(tc['E_eV'], np.maximum(tc['T'], 1e-6), color=CAT[0], lw=1.2)
    axi.set_ylim(1e-4, 2)
    axi.set_xlim(0, 0.6)
    axi.tick_params(labelsize=5.0, pad=1.2)
    axi.set_xlabel(r'$E_x$ (eV)', fontsize=5.4, labelpad=0.8)
    axi.set_ylabel(r'$T$', fontsize=5.4, labelpad=0.8)
    ax.annotate(r'$\phi_{\rm B}$', xy=(0.0, q['wse2']['phi_b_eV']),
                xytext=(1.3, q['wse2']['phi_b_eV'] * 0.80),
                fontsize=6.6, color=CAT[1], ha='left', va='center',
                arrowprops=dict(arrowstyle='-', lw=0.5, color=CAT[1],
                                shrinkA=1, shrinkB=2))
    _panel(ax, '(c)')

    # ---- (d) contact resistance against barrier height --------------------
    ax = axes[3]
    from . import quantum as qm
    phis = np.linspace(0.0, 0.45, 22)
    for name in ('MoS2', 'WS2', 'WSe2'):
        car = 'h' if name == 'WSe2' else 'e'
        Rc = [qm.contact_resistance(mats[name], p, q['screening_length_nm'],
                                    1.9e13, car) for p in phis]
        ax.semilogy(phis, Rc, color=CMAT[name], lw=1.4, label=LAB[name])
    ax.axhline(q['R_measured_ohm_um'], color=INK2, lw=0.8, ls='--')
    ax.annotate(r'MoS$_2$ bound', xy=(0.02, q['R_measured_ohm_um'] * 0.80),
                fontsize=5.8, color=INK2, ha='left', va='top')
    ax.set_xlabel(r'Barrier height $\phi_{\rm B}$ (eV)')
    ax.set_ylabel(r'$R_{\rm c}$ ($\Omega\,\mu$m)')
    ax.set_ylim(10, 1e5)
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.02), ncol=1,
              handlelength=0.9, labelspacing=0.15, fontsize=5.8)
    _panel(ax, '(d)')

    fig.savefig(os.path.join(FIGS, 'figS4.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_robust():
    """Robustness of the edge charge and of the critical width (stage A16).

    Written as figS6.pdf, the last supplementary figure in reading order."""
    res = _res()
    rr = res['robustness']
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.10))
    fig.subplots_adjust(wspace=0.55, left=0.070, right=0.99, bottom=0.235,
                        top=0.865)

    # (a) probability of an excess of electrons against shift noise
    ax = axes[0]
    nc = rr['noise_curve']
    ax.plot(nc['sigma_A'], 100.0 * np.array(nc['prob_pos']), color=INK,
            lw=1.3)
    for row in rr['noise']:
        ax.plot(row['sigma_A'], 100.0 * row['prob_pos'], 'o', ms=3.6,
                color=CAT[0], mfc='white', mew=1.0, zorder=5)
    ax.axhline(95.0, color=MUTED, lw=0.8, ls=':')
    ax.axvline(0.15, color=MUTED, lw=0.8, ls='--')
    ax.set_xlabel(r"$A_1'$ shift noise (cm$^{-1}$)")
    ax.set_ylabel(r'Samples with $n>0$ (%)')
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(50, 101)
    _panel(ax, '(a)')

    # (b) edge density against the overtone charge coefficient
    ax = axes[1]
    lab = {'computed': r'computed $\gamma$ pair',
           'low': r'low $\gamma_{\rm LA}$',
           'measured_pair': r'low $\gamma_{\rm LA}$, measured $\gamma_{A_1^\prime}$'}
    for i, (k, v) in enumerate(rr['overtone_charge'].items()):
        f = np.array(v['f'])
        n = np.array(v['n']) / 1e12
        ax.plot(f, n, color=CAT[i], lw=1.3, label=lab[k])
        ax.axvline(v['f_flip'], color=CAT[i], lw=0.8, ls=':')
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.axvspan(0.0, 0.5, color=MUTED, alpha=0.15, lw=0)
    ax.set_xlabel(r'Overtone charge coefficient $f$')
    ax.set_ylabel(r'Edge $n$ ($10^{12}$ cm$^{-2}$)')
    ax.set_xlim(0, 4)
    ax.set_ylim(-8, 18)
    lg = ax.legend(loc='upper left', fontsize=5.4, handlelength=1.0,
                   labelspacing=0.15, borderaxespad=0.3, frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
    lg.set_zorder(10)
    _panel(ax, '(b)')

    # (c) critical width against the size and sign of the edge charge
    ax = axes[2]
    names = {'HfO2_EOT1p5': r'HfO$_2$', 'SiO2_90nm': r'90 nm SiO$_2$',
             'SiO2_300nm': r'300 nm SiO$_2$'}
    for i, (st, row) in enumerate(rr['Wc_charge'].items()):
        ks = np.array([float(k) for k in row])
        vs = np.array([np.nan if v is None else v for v in row.values()])
        o = np.argsort(ks)
        ax.semilogy(ks[o], vs[o], 'o-', ms=3.2, color=CAT[i], lw=1.2,
                    label=names[st])
    ax.axvline(1.0, color=MUTED, lw=0.8, ls='--')
    ax.set_xlabel(r'Edge charge / extracted value')
    ax.set_ylabel(r'$W_{\rm c}$ (nm)')
    ax.set_xlim(-1.3, 3.3)
    lg = ax.legend(loc='upper left', fontsize=5.6, handlelength=1.0,
                   labelspacing=0.15, frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
    lg.set_zorder(10)
    _panel(ax, '(c)')

    # (d) critical width against halo width, both signs, HfO2 stack
    ax = axes[3]
    hs = np.array([float(h) for h in rr['Wc_halo']])
    for i, (k, lb) in enumerate((('1', 'extracted charge'),
                                 ('0', 'no edge charge'),
                                 ('-1', 'opposite sign'))):
        vs = np.array([np.nan if rr['Wc_halo'][h][k] is None
                       else rr['Wc_halo'][h][k] for h in rr['Wc_halo']])
        ax.loglog(hs, vs, 'o-', ms=3.2, color=CAT[i], lw=1.2, label=lb)
    ax.axvline(5.0, color=MUTED, lw=0.8, ls='--')
    ax.set_xlabel('Damage halo width (nm)')
    ax.set_ylabel(r'$W_{\rm c}$ on HfO$_2$ (nm)')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 20]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '5', '10', '20']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(FixedFormatter(['3', '5', '10', '20', '50']))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(2.5, 110)
    lg = ax.legend(loc='upper left', fontsize=5.6, handlelength=1.0,
                   labelspacing=0.15, frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
    lg.set_zorder(10)
    _panel(ax, '(d)')
    fig.savefig(os.path.join(FIGS, 'figS6.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_electrical():
    """The electrical cross-check (stage A17). Written as figS7.pdf."""
    res = _res()
    el = res['electrical']
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.30),
                             gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    fig.subplots_adjust(wspace=0.34, left=0.075, right=0.985, bottom=0.215,
                        top=0.885)

    ax = axes[0]
    a = np.array(D.PENA_VT, float)
    mk = {850: ('>', CAT[5], r'$W\approx850$ nm'),
          75: ('o', CAT[0], r'$W\approx75$ nm'),
          25: ('s', CAT[3], r'$W\approx25$ nm')}
    for w, (m_, c_, lb) in mk.items():
        sel = a[:, 0] == w
        ax.plot(a[sel, 1], a[sel, 2], m_, ms=4.0, color=c_, mfc='white',
                mew=1.0, ls='', label=lb)
    ax.set_xlabel(r'$L_{\rm ch}$ ($\mu$m)')
    ax.set_ylabel(r'$V_{\rm T,cc}$ (V)')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-1.3, 1.15)
    ax.legend(loc='upper right', fontsize=5.8, handlelength=0.8,
              labelspacing=0.15, frameon=False)
    _panel(ax, '(a)')

    ax = axes[1]
    ex = res.get('exact')
    th = ex['thick'] if ex else None
    fx = ex['fit'] if ex else None
    # (label, parallel-plate value, kind, exact value, exact lo, exact hi)
    rows = [
        ('Raman, one ribbon on gold (this work)', None, 'raman', el['raman_sigma'],
         res['uncertainty']['sigma_line']['p16'],
         res['uncertainty']['sigma_line']['p84']),
        ('$V_{\\rm T}$, XeF$_2$, HfO$_2$', el['pena']['1.5']['sigma'], 'est',
         fx and fx['sigma_cm'], fx and fx['boot_sigma_cm'][1],
         fx and fx['boot_sigma_cm'][3]),
        ('$V_{\\rm T}$, Cl$_2$/O$_2$, HfO$_2$', el['lan']['sigma'], 'upper',
         th and th['lan']['sigma_cm'], None, None),
        ('$V_{\\rm T}$, SF$_6$, 300 nm SiO$_2$', el['kotekar']['sigma_lower'],
         'lower', th and th['kotekar']['mid'], th and th['kotekar']['lo'],
         th and th['kotekar']['hi']),
        ('$V_{\\rm T}$, multilayer, 300 nm SiO$_2$',
         el['liu']['sigma_lower'], 'lower', th and th['liu']['sigma_cm'], None,
         None),
    ]
    for i, (lb, v, kind, xv, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        col = CAT[1] if i == 0 else INK
        if v is not None:
            ax.plot([v], [y], 'o', ms=4.6, color=col, mfc='white', mew=1.1)
            if kind == 'lower':
                ax.annotate('', xy=(v * 2.0, y), xytext=(v * 1.08, y),
                            arrowprops=dict(arrowstyle='->', lw=0.8, color=col))
        if xv:
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=col, lw=1.2)
            mfc = 'white' if kind == 'raman' else col
            ax.plot([xv], [y], 'o', ms=4.6, color=col, mfc=mfc, mew=1.1)
            if kind == 'upper':
                ax.annotate('', xy=(xv / 2.0, y), xytext=(xv / 1.08, y),
                            arrowprops=dict(arrowstyle='->', lw=0.8, color=col))
        ax.text(1.05e6, y + 0.28, lb, fontsize=5.8, color=col, va='bottom')
    ax.set_xscale('log')
    ax.set_xlim(1e6, 2e8)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_yticks([])
    ax.set_xlabel(r'Edge line charge $\sigma_{\rm e}$ (cm$^{-1}$ per edge)')
    _panel(ax, '(b)')
    fig.savefig(os.path.join(FIGS, 'figS7.pdf'))
    plt.close(fig)


def fig_exact():
    """Self-consistent electrostatic inversion of the threshold data (stage
    A18).  Written as fig_exact.pdf, Figure 4 of the main text."""
    from . import electrostatics as E, exact as X
    res = _res()
    el, ex = res['electrical'], res['exact']
    fx = ex['fit']
    eps = X.eps_for_eot(1.5)
    ell, n0 = fx['ell_nm'] * 1e-9, fx['N0_cm2'] * 1e4
    sig_fit = n0 * ell
    sig_conf = 3 * X.SIG_SITE
    tg = E.cc_target(X.I_CC, 0.3e-6, X.MU)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.30),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0]))
    fig.subplots_adjust(wspace=0.42, left=0.07, right=0.985, bottom=0.215,
                        top=0.885)
    # (a) conduction-band edge across a 75 nm ribbon at the threshold current
    ax = axes[0]
    s75 = E.Sheet(75e-9, X.T_HFO, eps, 1.0, X.R0_MOS2)
    s850 = E.Sheet(850e-9, X.T_HFO, eps, 1.0, X.R0_MOS2)
    for (lab, w, sg, prof, col, ls) in (
            ('fitted profile', ell, sig_fit, 'exp', CAT[0], '-'),
            ('outer 2 nm, 3 e per cell', 2e-9, sig_conf, 'strip', CAT[3], '--'),
            ('no edge charge', 2e-9, 0.0, 'strip', MUTED, ':')):
        v, phi = s75.gate_for(tg, -s75.fixed_potential(w, sg, prof))
        d = (75e-9 / 2 - s75.x) * 1e9
        ax.plot(d, -phi, ls, color=col, lw=1.1, label=lab)
    ax.set_xlim(0, 37.5)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel('Distance from edge (nm)')
    ax.set_ylabel(r'$E_{\rm c}-E_{\rm F}$ (eV)')
    ax.legend(loc='upper right', fontsize=5.6, handlelength=1.4,
              labelspacing=0.15, frameon=False)
    _panel(ax, '(a)')
    # (b) threshold offset against width, L = 0.3 um
    ax = axes[1]
    Wg = np.unique(np.concatenate([np.geomspace(15, 120, 22), [150, 200, 300, 500, 850.0]]))
    v850 = {}
    curves = {}
    for tag, w, sg, prof in (('fit', ell, sig_fit, 'exp'),
                             ('conf', 2e-9, sig_conf, 'strip')):
        vals = []
        for W in Wg:
            sh = E.Sheet(W * 1e-9, X.T_HFO, eps, 1.0, X.R0_MOS2,
                         h_max=max(1.5e-9, W * 1e-9 / 400))
            vals.append(sh.gate_for(tg, -sh.fixed_potential(w, sg, prof))[0])
        vals = np.array(vals)
        curves[tag] = vals - vals[-1]
    eq1 = 2 * E.QE * el['pena']['1.5']['sigma'] * 1e2 / (E.EPS0 * eps / X.T_HFO) * (
        1 / (Wg * 1e-9) - 1 / 850e-9)
    ax.plot(Wg, curves['fit'], '-', color=CAT[0], lw=1.1, label='fitted profile')
    ax.plot(Wg, curves['conf'], '--', color=CAT[3], lw=1.1, label='outer 2 nm only')
    ax.plot(Wg, eq1, ':', color=INK, lw=1.0, label='Eq. (1), fitted')
    ax.errorbar([75, 25], [el['d75'], el['d25']], yerr=[el['d75_se'], el['d25_se']],
                fmt='D', ms=4, color=INK, mfc='white', mew=1.0, capsize=2, lw=0.8,
                label='measured')
    ax.set_xscale('log')
    ax.set_xlim(14, 900)
    ax.set_ylim(-0.1, 2.2)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(FixedFormatter(['20', '50', '100', '200', '500']))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$W$ (nm)')
    ax.set_ylabel(r'$\Delta V_{\rm T,cc}$ vs. 850 nm (V)')
    ax.legend(loc='upper right', fontsize=5.6, handlelength=1.4,
              labelspacing=0.15, frameon=False)
    _panel(ax, '(b)')
    # (c) profile likelihood over (ell, N0)
    ax = axes[2]
    g = fx['grid']
    le, nn, dc = np.array(g['ell_nm']), np.array(g['N0_cm2']), np.array(g['dchi'])
    ax.contourf(le, nn, dc.T, levels=[0, 2.30, 6.18, 11.8], colors=['#3D8FC9', '#9CC5E6', '#E3EFF8'],
                alpha=0.8)
    ax.contour(le, nn, dc.T, levels=[2.30, 6.18], colors=[INK], linewidths=0.6)
    ax.plot([fx['ell_nm']], [fx['N0_cm2']], '+', color='white', ms=7, mew=1.4)
    ax.axhline(X.N_CELL * 1e-4, color=INK, lw=0.7, ls='--')
    ax.text(2.3, X.N_CELL * 1e-4 * 1.25, 'one per unit cell', fontsize=5.6, color=INK)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e12, 3e15)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 40]))
    ax.xaxis.set_major_formatter(FixedFormatter(['2', '5', '10', '20', '40']))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'Decay length $\ell$ (nm)')
    ax.set_ylabel(r'$N_0$ (cm$^{-2}$)')
    _panel(ax, '(c)')
    fig.savefig(os.path.join(FIGS, 'fig_exact.pdf'))
    plt.close(fig)


def main():
    figS1()
    figS2()
    fig_transport()
    fig_quantum()
    fig_adjoint()
    fig_robust()
    fig_electrical()
    fig_exact()
    tables()
    print('supplementary figures and tables written')


if __name__ == '__main__':
    main()
