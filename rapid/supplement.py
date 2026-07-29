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
    ax.set_xlabel('Vacuum thickness (Å)')
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
            color=INK, mfc='white', mew=0.9)
    ax.plot(w + 0.19, [PHONON[m]['wA'] for m in MATERIALS], 'o', ms=3.4,
            color=INK, mfc='white', mew=0.9)
    ax.set_xticks(w)
    ax.set_xticklabels([LAB[m] for m in MATERIALS], rotation=30, ha='right')
    ax.set_ylabel(r'Frequency (cm$^{-1}$)')
    ax.set_ylim(0, 560)
    ax.legend(handlelength=1.0, labelspacing=0.2, loc='upper left',
              bbox_to_anchor=(0.02, 0.99))
    ax.text(0.975, 0.955, 'circles: experiment', fontsize=5.9, color=INK2,
            transform=ax.transAxes, va='top', ha='right')
    _panel(ax, '(c)')
    fig.savefig(os.path.join(FIGS, 'figS2.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_adjoint():
    """Adjoint inversion: conditioning and synthetic map recovery.

    Written as figS4.pdf because it is the fourth supplementary figure in
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
    ax.set_ylim(0, 33)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), handlelength=1.0,
              labelspacing=0.18, fontsize=5.9)
    rm = res['map_recovery']['rmse']
    ax.text(0.985, 0.99, 'RMSE\n'
            r'$n_{\rm d}$: %.3f dex' '\n' r'$\varepsilon$: %.3f %%' '\n'
            r'$n$: %.3f' % tuple(rm), transform=ax.transAxes, fontsize=5.9,
            va='top', ha='right', color=INK2)
    _panel(ax, '(h)')
    fig.savefig(os.path.join(FIGS, 'figS4.pdf'))
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
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.04))
    fig.subplots_adjust(wspace=0.48, left=0.078, right=0.99, bottom=0.235,
                        top=0.865)

    ax = axes[0]
    nd = np.geomspace(1e11, 1e14, 70)
    tot, ph, pd, ci = [], [], [], []
    for n in nd:
        t, parts = transport.sheet_mobility(m, n)
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
              handlelength=1.4, labelspacing=0.18, fontsize=6.0)
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
    fr = np.linspace(0.3, 0.8, 18)
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
    ax.set_ylim(8, 6e3)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=2,
              handlelength=1.1, labelspacing=0.18, columnspacing=0.7,
              fontsize=6.0)
    _panel(ax, '(c)')
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
dependence of the elastic constant, and $C_{2D}$ the in-plane elastic constant
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
        mus = ('%.2f' % mu) if mu == mu else '---'
        rows.append(r'%s & %.3f & %.3f & $%.1f\times10^{%d}$ & %.1f & %s & %.0f \\'
                    % (row['label'], row['R_sh'], row['R_LA'],
                       row['nd'] / 10 ** int(np.floor(np.log10(row['nd']))),
                       int(np.floor(np.log10(row['nd']))), row['LD_nm'], mus,
                       row['mu_ceiling']))
    out.append(r"""\begin{table*}[t]
\caption{Inversion of the published atomic-layer-deposited and sputtered
MoS$_2$ films. $R_{\rm sh}$ and $R_{\rm LA}$ are the two disorder-activated
Raman ratios, $n_{\rm d}$ (cm$^{-2}$) and $L_{\rm D}$ (nm) the inverted defect
density and mean inter-defect distance, and the last two columns the measured
and point-defect-limited mobility (cm$^2$V$^{-1}$s$^{-1}$).}
\label{tab:nattoo}
\begin{ruledtabular}
\begin{tabular}{lcccccc}
Film & $R_{\rm sh}$ & $R_{\rm LA}$ & $n_{\rm d}$ & $L_{\rm D}$ &
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
    for k, v in PROVENANCE.items():
        tag, _, src = v.partition(': ')
        kind = kind_of.get(tag, tag)
        if src == 'this work':
            src = 'frozen phonons and biaxial strain sweep, this work'
        rows.append((k, kind, src))
    body = []
    for k, kind, src in rows:
        key = k.replace('_', r'\_')
        body.append(r'%s & %s & %s \\' % (key, kind, src))
    out.append(r"""\begin{table*}[t]
\caption{Provenance of every group of constants used in this work.  Entries
marked ``this work'' are computed here from first principles; ``calibrated
once'' means the value was fixed a single time against the cited published
result and then held for every number reported; ``geometry'' means a nominal
device dimension; ``assumed'' marks a modelling choice stated in the text; and
``not used'' marks a constant that is present in the code but multiplies zero
in every result reported here.}
\label{tab:prov}
\begin{ruledtabular}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.26\textwidth} l
>{\raggedright\arraybackslash}p{0.44\textwidth}@{}}
Constants & Origin & Source \\
\hline
""" + '\n'.join(body) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}""")

    with open(os.path.join(ROOT, 'paper', 'tables.tex'), 'w') as fh:
        fh.write('\n\n'.join(out) + '\n')


def main():
    figS1()
    figS2()
    fig_transport()
    fig_adjoint()
    tables()
    print('supplementary figures and tables written')


if __name__ == '__main__':
    main()
