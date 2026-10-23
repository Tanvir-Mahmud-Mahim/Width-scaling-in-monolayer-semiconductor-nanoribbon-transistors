"""A18: exact electrostatic inversion of the published threshold voltages.

Equation (1) of the article spreads the edge charge evenly over the ribbon.
The published thresholds are taken at a constant current of 100 nA/um, where
the channel is in weak to moderate inversion, and there the gate screens a
charge that sits near the edge within a few oxide thicknesses, so the
parallel-plate reading is not self-consistent.  This stage replaces it by the
self-consistent sheet electrostatics of `electrostatics.py` (layered Green's
function, Fermi-Dirac electrons in the K and Q valleys, Keldysh in-plane
screening) and inverts the digitised data of Pena et al. (2026) for both the
amount of fixed charge and the depth over which it extends.

Outputs (all written to results.json under 'exact'):
  checks     solver checks: uniform-sheet limit, a finite-difference solution
             of the same geometry, and agreement with the linear model
  ceiling    the largest 75 nm offset that charge confined to the outer 2 nm
             can produce, at one and at three electrons per edge unit cell
  fit        global fit of an exponential profile N0 exp(-d/l) to all devices
             with one offset per channel length; profile-likelihood and
             bootstrap intervals for l, N0 and sigma = N0 l
  variants   the same inversion (offsets at L = 0.3 um) for other stack
             parameters and for a uniform-strip profile
  thick      the same physics applied to the thick-oxide ribbons of
             Kotekar-Patil et al. and Liu et al., where the result hardly
             depends on the profile, and to Lan et al.
  wc         critical width on the reference high-kappa stack with the fitted
             profile in the on-state electrostatics
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import electrostatics as E
from . import datasets as D
from . import transport

T_HFO = 5.5e-9                      # HfO2 film of the threshold data
A_LATT = 0.316e-9                   # MoS2 lattice constant, m (edge unit cell)
SIG_SITE = 1.0 / A_LATT             # one electron per edge unit cell, per m
N_CELL = 2.0 / (np.sqrt(3.0) * A_LATT ** 2)   # unit cells per m^2
R0_MOS2 = 41.5e-10                  # 2 pi chi_2D, chi_2D = 6.60 A (DFT, Berkelbach 2013)
R0_MOS2_GW = 2 * np.pi * 5.0e-10    # chi_2D = 5.0 A
MU = 40e-4                          # m^2/Vs, middle of the 30 to 60 cm^2/Vs reported
I_CC = 0.1                          # 100 nA/um in A/m

L_GRID = np.array([0.03, 0.1, 0.3, 1.0]) * 1e-6
ELL_GRID = np.array([2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 32, 45.0]) * 1e-9
N0_GRID = np.geomspace(1e15, N_CELL, 16)          # m^-2, up to one per cell


def eps_for_eot(eot_nm, t=T_HFO):
    return 3.9 * t / (eot_nm * 1e-9)


def _sheets(widths_nm, t, eps, eps_top, r0, c_it=0.0):
    out = {}
    for W in widths_nm:
        hm = max(1.5e-9, W * 1e-9 / 400.0)
        out[W] = E.Sheet(W * 1e-9, t, eps, eps_top, r0, c_it=c_it, h_max=hm)
    return out


def vg_table(sheets, Ls, ells, n0s, profile='exp', I=I_CC, mu=MU):
    """Gate voltage at constant current, T[W, L, ell, N0] (V), and the same
    without fixed charge, T0[W, L]."""
    Ws = sorted(sheets)
    T = np.zeros((len(Ws), len(Ls), len(ells), len(n0s)))
    T0 = np.zeros((len(Ws), len(Ls)))
    for iw, W in enumerate(Ws):
        s = sheets[W]
        for il, L in enumerate(Ls):
            tg = E.cc_target(I, L, mu)
            T0[iw, il] = s.gate_for(tg, np.zeros_like(s.x))[0]
        for ie, ell in enumerate(ells):
            unit = -s.fixed_potential(ell, 1.0, profile)   # per unit sigma (1/m)
            for jn, n0 in enumerate(n0s):
                sig = n0 * (ell if profile == 'exp' else ell)
                pf = unit * sig
                for il, L in enumerate(Ls):
                    tg = E.cc_target(I, L, mu)
                    T[iw, il, ie, jn] = s.gate_for(tg, pf)[0]
    return Ws, T, T0


def _refine(T, ells, n0s, nf_ell=90, nf_n0=120):
    """Monotone (PCHIP) interpolation of a table on (log ell, log N0) to a
    fine grid; monotone so that no spurious minima appear between nodes."""
    le, ln = np.log(ells), np.log(n0s)
    fe = np.linspace(le[0], le[-1], nf_ell)
    fn = np.linspace(ln[0], ln[-1], nf_n0)
    a = PchipInterpolator(ln, T, axis=-1)(fn)
    a = PchipInterpolator(le, a, axis=-2)(fe)
    return np.exp(fe), np.exp(fn), a


def _device_model(Ws, Ls, Tf, Wdev, Ldev):
    """Model gate voltage for each device on the fine grid: interpolation in
    log L between the table lengths.  Returns (ndev, nell, nn0)."""
    lL = np.log(Ls)
    out = []
    for W, L in zip(Wdev, Ldev):
        iw = Ws.index(W)
        x = np.log(L)
        j = int(np.clip(np.searchsorted(lL, x) - 1, 0, len(Ls) - 2))
        f = (x - lL[j]) / (lL[j + 1] - lL[j])
        out.append((1 - f) * Tf[iw, j] + f * Tf[iw, j + 1])
    return np.array(out)


def _rss(V, Ldev, M):
    """Residual sum of squares with one free offset per channel length."""
    res = V[:, None, None] - M
    tot = np.zeros(M.shape[1:])
    for L in np.unique(Ldev):
        m = Ldev == L
        r = res[m]
        tot += ((r - r.mean(axis=0)) ** 2).sum(axis=0)
    return tot


def checks():
    """Solver checks against analytic limits and a finite-difference solve."""
    out = dict(E.verify())
    # linear Fourier potential against a finite-difference Poisson solution
    out['fd_max_rel_diff'] = float(_fd_check())
    # self-consistent solver against the linear subthreshold model
    t, eps = T_HFO, eps_for_eot(1.5)
    sh = _sheets((25, 75, 850), t, eps, 1.0, R0_MOS2)
    ell, sig = 13e-9, 4.0e9
    tg = E.cc_target(I_CC, 0.05e-6, MU)
    v = {W: s.gate_for(tg, -s.fixed_potential(ell, sig, 'exp'))[0] for W, s in sh.items()}
    lin = {W: E.threshold_shift(W * 1e-9, ell, sig, t, eps, 1.0, R0_MOS2, profile='exp')
           for W in (25, 75, 850)}
    sc = (v[75] - v[850], v[25] - v[850])
    li = (lin[75] - lin[850], lin[25] - lin[850])
    out['sc_vs_linear'] = [float(a / b - 1.0) for a, b in zip(sc, li)]
    # strong inversion with a weak charge (no local depletion): the
    # self-consistent shift approaches Eq. (1), reduced only by the quantum
    # capacitance and the fringing of the finite ribbon
    s75 = sh[75]
    sig2 = 1e8
    n_t = 5e16
    v0 = s75.gate_for(n_t, np.zeros_like(s75.x), measure='density')[0]
    vf = s75.gate_for(n_t, -s75.fixed_potential(5e-9, sig2, 'strip'), measure='density')[0]
    out['strong_inversion_ratio_to_eq1'] = float((vf - v0) /
                                                 E.parallel_plate(75e-9, sig2, t, eps))
    return out


def _fd_check():
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    W, ell, sig, t, eps = 75e-9, 13e-9, 1e9, T_HFO, 14.3
    h, Lpad, H = 0.25e-9, 150e-9, 300e-9
    Lx = W / 2 + Lpad
    nx = int(round(2 * Lx / h)) + 1
    x = np.linspace(-Lx, Lx, nx)
    h = x[1] - x[0]
    nzo = int(round(t / h))
    za, dz = [t], t / nzo
    while za[-1] < H:
        za.append(za[-1] + dz)
        dz *= 1.04
    z = np.concatenate([np.linspace(0, t, nzo + 1), np.array(za[1:])])
    nz = len(z)
    epsz = np.where(np.arange(nz - 1) < nzo, eps, 1.0)
    dens = E.QE * sig / ell * (np.exp(-(W / 2 - x) / ell) + np.exp(-(W / 2 + x) / ell)) * (np.abs(x) <= W / 2)
    N = nx * nz
    rows, cols, vals = [], [], []
    b = np.zeros(N)
    for j in range(nz):
        for i in range(nx):
            k = j * nx + i
            if j == 0 or j == nz - 1:
                rows.append(k); cols.append(k); vals.append(1.0)
                continue
            dzm, dzp = z[j] - z[j - 1], z[j + 1] - z[j]
            em, ep = epsz[j - 1], epsz[j]
            wz = (dzm + dzp) / 2
            ex = (em * dzm + ep * dzp) / (dzm + dzp)
            diag = 0.0
            for di in (-1, 1):
                ii = i + di
                if 0 <= ii < nx:
                    c = ex * wz / h
                    rows.append(k); cols.append(j * nx + ii); vals.append(-c)
                    diag += c
            cm, cp = em * h / dzm, ep * h / dzp
            rows += [k, k]; cols += [k - nx, k + nx]; vals += [-cm, -cp]
            diag += cm + cp
            rows.append(k); cols.append(k); vals.append(diag)
            if j == nzo:
                b[k] = dens[i] * h / E.EPS0
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    phi = spl.spsolve(A, b).reshape(nz, nx)[nzo]
    xs = np.array([0, 20e-9, 30e-9, 35e-9])
    pf = E.potential(xs, W, ell, sig, t, eps, 1.0, 0.0, profile='exp')
    return np.max(np.abs(np.interp(xs, x, phi) / pf - 1.0))


def ceiling(eps, t=T_HFO):
    """Largest 75 nm (and 25 nm) offset from charge in the outer 2 nm."""
    sh = _sheets((25, 75, 850), t, eps, 1.0, R0_MOS2)
    out = {}
    for tag, sig in (('one_per_site', SIG_SITE), ('three_per_site', 3 * SIG_SITE)):
        best = None
        for L in L_GRID:
            tg = E.cc_target(I_CC, L, MU)
            v = {W: s.gate_for(tg, -s.fixed_potential(2e-9, sig, 'strip'))[0]
                 for W, s in sh.items()}
            d = (v[75] - v[850], v[25] - v[850])
            if best is None or d[0] > best[0]:
                best = d
        out[tag] = dict(d75=float(best[0]), d25=float(best[1]))
    return out


def global_fit(eps, n_boot=2000, seed=11, eps_top=1.0, r0=R0_MOS2):
    a = np.array(D.PENA_VT, float)
    Wdev, Ldev, V = a[:, 0], a[:, 1] * 1e-6, a[:, 2]
    sh = _sheets((25, 75, 850), T_HFO, eps, eps_top, r0)
    Ws, T, T0 = vg_table(sh, L_GRID, ELL_GRID, N0_GRID)
    fe, fn, Tf = _refine(T, ELL_GRID, N0_GRID)
    M = _device_model(Ws, L_GRID, Tf, list(Wdev.astype(int)), Ldev)
    rss = _rss(V, Ldev, M)
    nL = len(np.unique(Ldev))
    dof = len(V) - nL - 2
    i, j = np.unravel_index(np.argmin(rss), rss.shape)
    s2 = rss[i, j] / dof
    dchi = (rss - rss[i, j]) / s2
    prof_ell = dchi.min(axis=1)
    sig = fe[:, None] * fn[None, :]
    def interval(prof, grid, lev=1.0):
        ok = grid[prof <= lev]
        return [float(ok.min()), float(ok.max())]
    # profile likelihood of sigma: min over the grid at each sigma bin
    lsig = np.log(sig).ravel()
    bins = np.linspace(lsig.min(), lsig.max(), 200)
    idx = np.digitize(lsig, bins)
    prof_sig = np.full(bins.size + 1, np.inf)
    np.minimum.at(prof_sig, idx, dchi.ravel())
    sig_ok = np.exp(bins[np.isfinite(prof_sig[1:]) & (prof_sig[1:] <= 1.0)])
    prof_n0 = dchi.min(axis=0)
    # null model: charge confined to 2 nm is the smallest-ell row
    rss_conf = float(rss[0].min())
    # bootstrap over devices with the reading uncertainty
    rng = np.random.default_rng(seed)
    err = D.PENA_VT_META['read_uncertainty_V']
    bl, bn, bs = [], [], []
    for _ in range(n_boot):
        k = rng.integers(0, len(V), len(V))
        if len(set(Wdev[k])) < 3:
            continue
        r = _rss(V[k] + rng.normal(0, err, len(k)), Ldev[k], M[k])
        p, q = np.unravel_index(np.argmin(r), r.shape)
        bl.append(fe[p]); bn.append(fn[q]); bs.append(fe[p] * fn[q])
    pct = lambda arr: [float(np.percentile(arr, z)) for z in (2.5, 16, 50, 84, 97.5)]
    # offsets the best fit predicts at L = 0.3 um, for comparison with the
    # fixed-effects offsets of A17
    il = list(L_GRID).index(0.3e-6)
    d = {W: Tf[Ws.index(W), il, i, j] for W in (25, 75, 850)}
    return dict(
        ell_nm=float(fe[i] * 1e9), N0_cm2=float(fn[j] * 1e-4),
        sigma_cm=float(fe[i] * fn[j] * 1e-2),
        N0_per_cell=float(fn[j] / N_CELL),
        sigma_per_site=float(fe[i] * fn[j] / SIG_SITE),
        rss=float(rss[i, j]), s=float(np.sqrt(s2)), dof=int(dof),
        rss_confined=rss_conf,
        dchi_confined=float((rss_conf - rss[i, j]) / s2),
        ell_68_nm=[v * 1e9 for v in interval(prof_ell, fe)],
        ell_95_nm=[v * 1e9 for v in interval(prof_ell, fe, 3.84)],
        N0_68_cm2=[v * 1e-4 for v in interval(prof_n0, fn)],
        sigma_68_cm=[float(sig_ok.min() * 1e-2), float(sig_ok.max() * 1e-2)],
        boot_ell_nm=[v * 1e9 for v in pct(bl)],
        boot_N0_cm2=[v * 1e-4 for v in pct(bn)],
        boot_sigma_cm=[v * 1e-2 for v in pct(bs)],
        n_boot=len(bl),
        d75_fit=float(d[75] - d[850]), d25_fit=float(d[25] - d[850]),
        grid=dict(ell_nm=(fe * 1e9).tolist(), N0_cm2=(fn * 1e-4).tolist(),
                  dchi=dchi.tolist()),
    )


def offsets_fit(d75, e75, d25, e25, eps, eps_top=1.0, r0=R0_MOS2, c_it=0.0,
                profile='exp', L=0.3e-6, ells=None, n0s=None):
    """Two-parameter inversion of the two width offsets at one length."""
    ells = ELL_GRID if ells is None else ells
    n0s = N0_GRID if n0s is None else n0s
    sh = _sheets((25, 75, 850), T_HFO, eps, eps_top, r0, c_it)
    Ws, T, _ = vg_table(sh, [L], ells, n0s, profile)
    fe, fn, Tf = _refine(T[:, 0], ells, n0s)
    a = Tf[Ws.index(75)] - Tf[Ws.index(850)]
    b = Tf[Ws.index(25)] - Tf[Ws.index(850)]
    chi = ((a - d75) / e75) ** 2 + ((b - d25) / e25) ** 2
    i, j = np.unravel_index(np.argmin(chi), chi.shape)
    if profile == 'exp':
        sig = fe[i] * fn[j]
    else:
        sig = fn[j] * fe[i]
    prof = chi.min(axis=1)
    ok = fe[prof <= chi[i, j] + 1.0]
    return dict(ell_nm=float(fe[i] * 1e9), N0_cm2=float(fn[j] * 1e-4),
                sigma_cm=float(sig * 1e-2), chi2=float(chi[i, j]),
                d75=float(a[i, j]), d25=float(b[i, j]),
                ell_68_nm=[float(ok.min() * 1e9), float(ok.max() * 1e9)])


def _solve_sigma(target_dV, sheet, wide, ell, n_target, profile='exp'):
    """Line charge (per m) whose ribbon-to-wide threshold difference equals
    target_dV on a thick oxide, threshold at a mean density n_target."""
    from scipy.optimize import brentq
    v_w = wide.gate_for(n_target, np.zeros_like(wide.x), measure='density',
                        Vlo=-50, Vhi=200)[0]
    unit = -sheet.fixed_potential(ell, 1.0, profile)

    def g(lsig):
        v = sheet.gate_for(n_target, unit * np.exp(lsig), measure='density',
                           Vlo=-50, Vhi=600)[0]
        return v - v_w - target_dV
    return float(np.exp(brentq(g, np.log(1e8), np.log(1e11), xtol=1e-4)))


def thick(ell):
    """Exact readings of the thick-oxide device studies and of Lan et al."""
    out = {}
    kp = D.KOTEKAR2019
    t = kp['t_SiO2_nm'] * 1e-9
    s = E.Sheet(kp['W_nm'] * 1e-9, t, 3.9, 1.0, R0_MOS2)
    w = E.Sheet(3000e-9, t, 3.9, 1.0, R0_MOS2, h_max=8e-9)
    dv_mid = np.mean(kp['Vth_ribbon_V']) - np.mean(kp['Vth_unetched_V'])
    dv_lo = min(kp['Vth_ribbon_V']) - max(kp['Vth_unetched_V'])
    dv_hi = max(kp['Vth_ribbon_V']) - min(kp['Vth_unetched_V'])
    r = {}
    for tag, dv in (('mid', dv_mid), ('lo', dv_lo), ('hi', dv_hi)):
        r[tag] = _solve_sigma(dv, s, w, ell, 1e15) * 1e-2
    r['short_ell'] = _solve_sigma(dv_mid, s, w, 2e-9, 1e15) * 1e-2
    r['n_lo'] = _solve_sigma(dv_mid, s, w, ell, 3e14) * 1e-2
    r['n_hi'] = _solve_sigma(dv_mid, s, w, ell, 3e15) * 1e-2
    r['dV'] = [float(dv_lo), float(dv_mid), float(dv_hi)]
    out['kotekar'] = r
    li = D.LIU2012
    s = E.Sheet(li['narrowest_nm'] * 1e-9, li['tox_nm'] * 1e-9, 3.9, 1.0, R0_MOS2)
    w = E.Sheet(2000e-9, li['tox_nm'] * 1e-9, 3.9, 1.0, R0_MOS2, h_max=6e-9)
    out['liu'] = dict(sigma_cm=_solve_sigma(li['dVT_total_V'], s, w, ell, 1e15) * 1e-2)
    ln = D.LAN2026
    t = ln['t_HfO2_nm'] * 1e-9
    sn = E.Sheet(ln['W_narrow_nm'] * 1e-9, t, ln['kappa'], 1.0, R0_MOS2)
    sw = E.Sheet(ln['W_wide_nm'] * 1e-9, t, ln['kappa'], 1.0, R0_MOS2)
    from scipy.optimize import brentq
    tg = E.cc_target(I_CC, 65e-9, MU)
    unit_n = -sn.fixed_potential(ell, 1.0, 'exp')
    unit_w = -sw.fixed_potential(ell, 1.0, 'exp')

    def g(lsig):
        sg = np.exp(lsig)
        return (sn.gate_for(tg, unit_n * sg)[0] - sw.gate_for(tg, unit_w * sg)[0]
                - ln['dVT_V'])
    out['lan'] = dict(sigma_cm=float(np.exp(brentq(g, np.log(1e7), np.log(1e11),
                                                   xtol=1e-4)) * 1e-2))
    return out


def wc_exact(mats, ell, n0, nd_ref=1.3e12, eot=1.5):
    """Critical width with the fitted profile in the on-state electrostatics.

    The overdrive at each width is the reference overdrive scaled by the
    width-averaged electron density that the self-consistent solve gives at
    the same gate voltage, relative to a ribbon without edge charge."""
    eps = eps_for_eot(eot)
    Vov = 1.5
    Wg = np.array([6, 8, 10, 12, 15, 18, 22, 27, 33, 40, 50, 60, 75, 90, 110,
                   140, 180, 250, 350, 500, 800, 1500, 4000.0])
    sheets = _sheets(Wg, T_HFO, eps, 1.0, R0_MOS2)
    ref = sheets[4000.0]
    tg = E.cc_target(I_CC, 0.3e-6, MU)
    v0 = ref.gate_for(tg, np.zeros_like(ref.x))[0]
    phi = ref.solve(v0 + Vov, np.zeros_like(ref.x))
    n_ref = ref.mean_density(phi)
    # Both ratios compare a charged ribbon with an uncharged ribbon of the
    # same width, so that the fringing capacitance of a narrow ribbon, which
    # the transport model does not carry, cancels and only the edge charge
    # remains.
    ratio, ratio_own, dvt = [], [], []
    for W in Wg:
        s = sheets[W]
        pf = -s.fixed_potential(ell, n0 * ell, 'exp')
        z = np.zeros_like(s.x)
        vt0 = s.gate_for(tg, z)[0]
        ratio.append(s.mean_density(s.solve(v0 + Vov, pf))
                     / s.mean_density(s.solve(v0 + Vov, z)))
        # the same ribbon driven at the same overdrive above its own
        # constant-current threshold, as a threshold-adjusted design would be
        vt = s.gate_for(tg, pf)[0]
        dvt.append(vt - vt0)
        ratio_own.append(s.mean_density(s.solve(vt + Vov, pf))
                         / s.mean_density(s.solve(vt0 + Vov, z)))
    ratio = np.array(ratio)
    ratio_own = np.array(ratio_own)
    lw = np.log(Wg)

    def vov_fn(W):
        return Vov * np.interp(np.log(np.asarray(W, float)), lw, ratio)

    def vov_own(W):
        return Vov * np.interp(np.log(np.asarray(W, float)), lw, ratio_own)

    m = mats['MoS2']
    st = transport.STACK['HfO2_EOT1p5']
    base = dict(Cox=transport.COX['HfO2_EOT1p5'], Vov=Vov, Vds=1.0,
                Lch_nm=300.0, n_it_cm2=st['nit'], eps_env=st['eps'])
    wc5 = transport.critical_width(m, nd_ref, halo_nm=D.HALO['GENTLE_nm'],
                                   vov_fn=vov_fn, **base)
    wc0 = transport.critical_width(m, nd_ref, halo_nm=0.0, vov_fn=vov_fn, **base)
    wco = transport.critical_width(m, nd_ref, halo_nm=D.HALO['GENTLE_nm'],
                                   vov_fn=vov_own, **base)
    i50 = list(Wg).index(50)
    return dict(Wc_halo5=float(wc5), Wc_halo0=float(wc0), Wc_own_halo5=float(wco),
                W=Wg.tolist(), density_ratio=ratio.tolist(),
                density_ratio_own=ratio_own.tolist(), dVT=dvt,
                ratio50=float(ratio[i50]), ratio50_own=float(ratio_own[i50]))


CACHE = os.path.join(os.path.dirname(__file__), '..', 'exact_cache.json')


def _cached(name, fn, cache):
    if name not in cache:
        cache[name] = fn()
        with open(CACHE, 'w') as fh:
            json.dump(cache, fh, default=float)
    return cache[name]


def A18_exact(mats, el, eot_main=1.5, n_boot=2000, use_cache=True):
    """Run every part of the stage; parts already computed for the same
    inputs are read from exact_cache.json (delete it to recompute all)."""
    t0 = time.time()
    key = '%.6f %.6f %.6f %.6f' % (el['d75'], el['d75_se'], el['d25'], el['d25_se'])
    cache = {}
    if use_cache and os.path.exists(CACHE):
        cache = json.load(open(CACHE))
        if cache.get('key') != key:
            cache = {}
    cache['key'] = key
    out = {}
    eps_main = eps_for_eot(eot_main)
    out['checks'] = _cached('checks', checks, cache)
    out['ceiling'] = _cached('ceiling', lambda: ceiling(eps_main), cache)
    out['fit'] = _cached('fit', lambda: global_fit(eps_main, n_boot=n_boot), cache)
    out['fit_eot1p1'] = _cached('fit_eot1p1', lambda: global_fit(eps_for_eot(1.1), n_boot=500), cache)
    d75, e75, d25, e25 = el['d75'], el['d75_se'], el['d25'], el['d25_se']
    cox = E.EPS0 * eps_main / T_HFO
    specs = {
        'base': dict(eps=eps_main),
        'eot1p1': dict(eps=eps_for_eot(1.1)),
        'eps_top4': dict(eps=eps_main, eps_top=4.0),
        'r0_gw': dict(eps=eps_main, r0=R0_MOS2_GW),
        'r0_zero': dict(eps=eps_main, r0=0.0),
        'traps_n1p5': dict(eps=eps_main, c_it=0.5 * cox),
        'L_0p05': dict(eps=eps_main, L=0.05e-6),
        'L_1': dict(eps=eps_main, L=1.0e-6),
        'strip': dict(eps=eps_main, profile='strip',
                      ells=np.array([2, 5, 10, 15, 20, 25, 30, 35, 40.0]) * 1e-9),
    }
    var = {}
    for name, kw in specs.items():
        var[name] = _cached('var_' + name,
                            lambda kw=kw: offsets_fit(d75, e75, d25, e25, **kw), cache)
    out['variants'] = var
    f = out['fit']
    out['thick'] = _cached('thick', lambda: thick(f['ell_nm'] * 1e-9), cache)
    out['wc'] = _cached('wc', lambda: wc_exact(mats, f['ell_nm'] * 1e-9,
                                               f['N0_cm2'] * 1e4, eot=eot_main), cache)
    out['wc_68'] = _cached('wc_68', lambda: [
        wc_exact(mats, f['ell_nm'] * 1e-9, sg * 1e2 / (f['ell_nm'] * 1e-9),
                 eot=eot_main)['Wc_halo5']
        for sg in (f['boot_sigma_cm'][1], f['boot_sigma_cm'][3])], cache)
    out['runtime_s'] = time.time() - t0
    return out


if __name__ == '__main__':
    from .materials import all_materials
    here = os.path.join(os.path.dirname(__file__), '..', 'results.json')
    res = json.load(open(here))
    mats = all_materials()
    res['exact'] = A18_exact(mats, res['electrical'])
    with open(here, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    ex = {k: v for k, v in res['exact'].items() if k != 'fit'}
    print(json.dumps(ex, indent=1, default=float)[:6000])
    print({k: v for k, v in res['exact']['fit'].items() if k != 'grid'})
