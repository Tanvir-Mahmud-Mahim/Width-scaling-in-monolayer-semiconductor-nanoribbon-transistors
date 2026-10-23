"""End-to-end uncertainty propagation and validity bounds.

This module turns the qualitative caveats of the Limitations section into
computed numbers.  Nothing here changes any reported central value; every
stage below re-runs the same estimators used in analysis.py with their inputs
drawn from explicitly stated distributions, or evaluates a closed-form
validity criterion from constants already in the parameter table.

Stages
------
A13_uncertainty
    Monte Carlo propagation of every stated input uncertainty of the edge
    charge inversion through to the recovered edge density, the line charge
    and the critical width:
      * the A1' doping coefficient dA, Gaussian with the 15 per cent
        uncertainty this work assigns to the Chakraborty measurement,
        which reports the softening without an error estimate,
      * the A1' Grueneisen parameter, log-uniform between the measured
        biaxial value (Michail 2024) and the value computed in this work,
        which is the full computed-to-measured bracket of limitation 10,
      * the acoustic Grueneisen parameter, log-uniform between the computed
        value scaled by the WSe2 measured-to-computed 2LA ratio (the one
        biaxial acoustic measurement that exists) and the computed value,
      * the two Raman shifts, Gaussian with the stated single-pixel noise,
      * the edge width w_e, uniform on 5 to 20 nm (the range scanned in A8),
    and, for the defect densities of A4, the calibration constant
    C_A = 0.59 +- 0.03 nm^2 in quadrature with the intensity-ratio error.

A14_validity
    Closed-form validity bounds: the Ioffe-Regel product k_F l at both
    Dossena comparison points, the mean free path against the inter-defect
    spacing, the Kane two-band bound on non-parabolic mass renormalisation
    over the occupied Fermi window, and the spread of the transferred Raman
    calibration constants when the exciton detuning is varied by the exciton
    linewidth already in the parameter table.

A15_mass_sensitivity
    The K-valley effective masses swapped from the production set (Jin 2014)
    to the independent spin-orbit-resolved k.p set (Kormanyos 2015), with the
    sheet mobility at the Dossena density and the critical width recomputed.
    The Kormanyos masses are passed in from analysis.py where they are
    recorded with their source; this module contains no numbers of its own.
"""
from __future__ import annotations

import numpy as np

from . import datasets as D, transport
from .materials import (M0, HBAR, QE, KB, T300, GAMMA_MEAS,
                        RAMAN_CAL)

SIGMA_A = 0.15          # single-pixel A1' shift noise, cm^-1 (as in A5)
SIGMA_2LA = 0.30        # single-pixel 2LA(M) shift noise, cm^-1


def _invert(dwA, dw2LA, dwA_deps, dwA_dn, dw2LA_deps,
            sA=SIGMA_A, s2=SIGMA_2LA):
    """The exact A5 estimator: weighted least squares with the same prior."""
    G = np.array([[dwA_deps, dwA_dn], [dw2LA_deps, 0.0]])
    Cd = np.diag([1 / sA ** 2, 1 / s2 ** 2])
    H = G.T @ Cd @ G + np.diag([1 / 1.0 ** 2, 1 / 2.0 ** 2])
    rhs = G.T @ Cd @ np.array([dwA, dw2LA])
    return np.linalg.solve(H, rhs)


def _pct(x, ps=(2.5, 16.0, 50.0, 84.0, 97.5)):
    q = np.percentile(np.asarray(x, float), ps)
    return {('p%g' % p).replace('.', '_'): float(v) for p, v in zip(ps, q)}


def A13_uncertainty(mats, n_edge_nominal, nd_ref=1.3e12,
                    N=20000, seed=11):
    m = mats['MoS2']
    rng = np.random.default_rng(seed)

    # ---- input distributions, every one tied to a stated source ---------
    gA_hi = m.gA                                   # computed, this work
    gA_lo = m.gA_meas                              # Michail 2024, biaxial
    gLA_dft = m.dft['gamma_LA']
    wse2 = mats['WSe2']
    # the one biaxial acoustic measurement: WSe2 2LA (GAMMA_MEAS)
    gLA_meas_wse2 = GAMMA_MEAS['WSe2']['gLA']
    gLA_lo = gLA_dft * (gLA_meas_wse2 / wse2.dft['gamma_LA'])
    gLA_hi = gLA_dft
    dA0, dA_sig = m.dA, 0.15 * abs(m.dA)           # Chakraborty, 15 %

    gA = np.exp(rng.uniform(np.log(gA_lo), np.log(gA_hi), N))
    gLA = np.exp(rng.uniform(np.log(gLA_lo), np.log(gLA_hi), N))
    dA = rng.normal(dA0, dA_sig, N)
    w2LA = 2.0 * m.wLA
    dwA_meas = rng.normal(D.KRAYEV['edge_dwA'], SIGMA_A, N)
    dw2_meas = rng.normal(D.KRAYEV['edge_dw2LA'], SIGMA_2LA, N)
    we = rng.uniform(5.0, 20.0, N)

    # ---- push through the A5 estimator ---------------------------------
    eps = np.empty(N)
    nn = np.empty(N)
    for i in range(N):
        x = _invert(dwA_meas[i], dw2_meas[i],
                    -2.0 * gA[i] * m.wA / 100.0, dA[i],
                    -2.0 * gLA[i] * w2LA / 100.0)
        eps[i], nn[i] = x[0], x[1]
    n_cm2 = nn * 1e13
    sigma_line = n_cm2 * we * 1e-7                  # cm^-1

    # ---- the same for the interior spot ---------------------------------
    dwA_sp = rng.normal(D.KRAYEV['spot_dwA'], SIGMA_A, N)
    dw2_sp = rng.normal(D.KRAYEV['spot_dw2LA'], SIGMA_2LA, N)
    eps_sp = np.empty(N)
    n_sp = np.empty(N)
    for i in range(N):
        x = _invert(dwA_sp[i], dw2_sp[i],
                    -2.0 * gA[i] * m.wA / 100.0, dA[i],
                    -2.0 * gLA[i] * w2LA / 100.0)
        eps_sp[i], n_sp[i] = x[0], x[1]

    # ---- critical width: exact 1D map over the line charge --------------
    st = transport.STACK['HfO2_EOT1p5']
    base = dict(halo_nm=D.HALO['GENTLE_nm'],
                Cox=transport.COX['HfO2_EOT1p5'], Vov=1.5, Vds=1.0,
                Lch_nm=300.0, n_it_cm2=st['nit'], eps_env=st['eps'])
    lo, hi = np.min(sigma_line), np.max(sigma_line)
    grid = np.linspace(lo, hi, 41)
    wc_grid = np.array([transport.critical_width(
        m, nd_ref, **dict(base, sigma_line_cm=float(s))) for s in grid])
    # A strongly negative line charge (the far tail of the accumulation
    # branch) enhances the narrow ribbons enough that the current never
    # falls to half its wide-channel value: there the 50 per cent criterion
    # is met at every width, so the critical width is below the scan floor
    # and is recorded at that floor.  This can only happen at negative
    # sigma; a NaN at positive sigma would be a real failure.
    bad = ~np.isfinite(wc_grid)
    if np.any(bad & (grid >= 0.0)):
        raise RuntimeError('critical width undefined at positive line charge')
    wc_grid[bad] = 3.0
    frac_floor = float(np.mean(np.interp(sigma_line, grid, wc_grid) <= 3.0))
    order = np.argsort(grid)
    Wc = np.interp(sigma_line, grid[order], wc_grid[order])
    Wc_nom = float(transport.critical_width(
        m, nd_ref, **dict(base, sigma_line_cm=transport.edge_line_charge(
            n_edge_nominal, 10.0))))

    # ---- defect-density uncertainty from the calibration constant -------
    CA_rel = RAMAN_CAL['C_A_err'] / RAMAN_CAL['C_A']
    nd_rel = {}
    for e in D.NATTOO:
        r = np.sqrt((e['R_LA_e'] / e['R_LA']) ** 2 + CA_rel ** 2)
        nd_rel[e['label']] = float(r)

    return dict(
        N=N,
        priors=dict(gA=[float(gA_lo), float(gA_hi)],
                    gLA=[float(gLA_lo), float(gLA_hi)],
                    dA=[float(dA0), float(dA_sig)],
                    we_nm=[5.0, 20.0],
                    shift_noise=[SIGMA_A, SIGMA_2LA]),
        edge_n_cm2=_pct(n_cm2),
        edge_eps=_pct(eps),
        spot_eps=_pct(eps_sp),
        spot_n_cm2=_pct(n_sp * 1e13),
        sigma_line=_pct(sigma_line),
        prob_n_positive=float(np.mean(n_cm2 > 0)),
        prob_spot_tensile=float(np.mean(eps_sp > 0)),
        Wc=_pct(Wc),
        Wc_nominal=Wc_nom,
        Wc_floor_fraction=frac_floor,
        wc_map=dict(sigma=grid.tolist(), Wc=wc_grid.tolist()),
        nd_rel_err=nd_rel)


def A14_validity(mats):
    ws2 = mats['WS2']
    out = {}

    # ---- Ioffe-Regel product at the two Dossena points ------------------
    # k_F from the carrier density of that work (1.6 to 1.8e13 cm^-2; the
    # model comparison in A7 uses 1.9e13), l = hbar k_F mu / q.
    n_cm2 = 1.9e13
    kF = np.sqrt(np.pi * n_cm2 * 1e4)   # m^-1, K-valley, g_s g_v = 4
    rows = []
    for nd, mu_ref in D.DOSSENA['points']:
        mu_pred, _ = transport.sheet_mobility(ws2, nd, n_cm2=n_cm2,
                                              n_it_cm2=0.0)
        d_defect = 1.0 / np.sqrt(nd * 1e4)          # m, inter-defect spacing
        row = dict(nd=nd, d_defect_nm=float(d_defect * 1e9))
        for tag, mu in (('model', float(mu_pred)), ('reference', mu_ref)):
            mob = mu * 1e-4                          # m^2/Vs
            l = HBAR * kF * mob / QE                 # m
            row[tag] = dict(mu=mu, kFl=float(kF * l),
                            mfp_nm=float(l * 1e9),
                            mfp_over_spacing=float(l / d_defect))
        rows.append(row)
    out['ioffe_regel'] = dict(n_cm2=n_cm2, kF_nm_inv=float(kF * 1e-9),
                              points=rows)

    # ---- Kane two-band bound on non-parabolicity ------------------------
    # m*(E) = m*(1 + 2E/Eg); the occupied window reaches E_F + 2kT.  Using
    # the PBE gap (smaller than the true quasiparticle gap) makes the bound
    # conservative.
    kane = {}
    n_cm2 = 1.9e13
    for name in ('MoS2', 'WS2'):
        mat = mats[name]
        g = 2.0                                  # K and K' valleys
        m_e = mat.mcK * M0
        EF = (np.pi * HBAR ** 2 * n_cm2 * 1e4) / (g * m_e) / QE  # eV
        Ewin = EF + 2.0 * KB * T300 / QE
        Eg = mat.dft['gap_K']
        kane[name] = dict(EF_eV=float(EF), E_window_eV=float(Ewin),
                          Eg_PBE_eV=float(Eg),
                          mass_shift_bound=float(2.0 * Ewin / Eg))
    out['kane'] = kane

    # ---- detuning spread of the transferred calibration constants -------
    # C_A scales as 1/Delta^2 with Delta the A-exciton detuning at 532 nm.
    # Varying Delta by the exciton linewidth Gam0 in the parameter table
    # spreads the transferred constant by ((D+G)/(D-G))^2.  For WS2 the
    # resonance that matters at 532 nm is the B exciton (Rodriguez 2026), so
    # this spread is a lower bound there.
    E_laser = 1239.84 / 532.0
    det = {}
    for name, mat in mats.items():
        Delta = abs(E_laser - mat.EA0)
        G = mat.GamX0
        det[name] = dict(Delta_eV=float(Delta), Gam0_eV=float(G),
                         spread=float(((Delta + G) / (Delta - G)) ** 2))
    out['detuning'] = det
    return out


def A15_mass_sensitivity(mats, kp_masses, sigma_line_cm):
    """Swap the K-valley masses for the independent k.p set and re-run.

    kp_masses: {material: dict(mcK=..., mvK=...)} with provenance recorded at
    the call site (analysis.py).  Only the K-valley masses are swapped; the
    Q-valley and Gamma parameters have no counterpart in the k.p set.
    """
    import copy
    out = {}
    st = transport.STACK['HfO2_EOT1p5']
    base = dict(halo_nm=D.HALO['GENTLE_nm'],
                sigma_line_cm=sigma_line_cm,
                Cox=transport.COX['HfO2_EOT1p5'], Vov=1.5, Vds=1.0,
                Lch_nm=300.0, n_it_cm2=st['nit'], eps_env=st['eps'])
    for name, mat in mats.items():
        if name not in kp_masses:
            continue
        alt = copy.copy(mat)
        alt.mcK = kp_masses[name]['mcK']
        alt.mvK = kp_masses[name]['mvK']
        row = dict(jin=dict(mcK=mat.mcK, mvK=mat.mvK),
                   kp=dict(mcK=alt.mcK, mvK=alt.mvK))
        for tag, mm in (('jin', mat), ('kp', alt)):
            car = 'h' if name == 'WSe2' else 'e'
            mu, _ = transport.sheet_mobility(mm, 2e13, n_cm2=1.9e13,
                                             n_it_cm2=0.0, carrier=car)
            row[tag]['mu_dossena'] = float(mu)
            if name == 'MoS2':
                row[tag]['Wc'] = float(transport.critical_width(
                    mm, 1.3e12, **base))
        row['mu_ratio'] = row['kp']['mu_dossena'] / row['jin']['mu_dossena']
        if 'Wc' in row['jin']:
            row['Wc_ratio'] = row['kp']['Wc'] / row['jin']['Wc']
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# A16_robustness
#     Added in response to review.  The edge charge rests on one published
#     tip-enhanced data set, and the critical width on a model, so this stage
#     asks how far each conclusion survives when the assumptions behind it
#     are pushed beyond the values used in A5 and A8:
#       * the shift noise, raised from the single-pixel value to more than
#         three times it, with the Monte Carlo of A13 repeated at each level;
#       * the charge coefficient of the 2LA(M) overtone, which has not been
#         measured for MoS2, scanned until the sign of the result reverses,
#         for both ends of the acoustic Grueneisen bracket;
#       * the internal closure at the interior spot, where the strain read
#         from 2LA(M) alone predicts the A1' shift;
#       * the critical width against the size and sign of the edge charge and
#         against the halo width, which the XeF2 etch does not report.
#     Every input is one already used elsewhere in the pipeline.
# ---------------------------------------------------------------------------
NOISE_LEVELS = (0.15, 0.25, 0.35, 0.50)     # cm^-1 on A1'; 2LA(M) twice that
CHARGE_SCALES = (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0)
HALO_LEVELS = (1.0, 5.0, 10.0, 20.0)


def _mc_edge(mats, sA, s2, N, seed, frac_2la=None):
    """A13 edge Monte Carlo at a chosen shift noise; optionally with a 2LA(M)
    charge coefficient drawn uniformly on [0, frac_2la] of the frequency-scaled
    A1' coefficient."""
    m = mats['MoS2']
    rng = np.random.default_rng(seed)
    gLA_dft = m.dft['gamma_LA']
    gLA_lo = gLA_dft * (GAMMA_MEAS['WSe2']['gLA']
                        / mats['WSe2'].dft['gamma_LA'])
    gA = np.exp(rng.uniform(np.log(m.gA_meas), np.log(m.gA), N))
    gLA = np.exp(rng.uniform(np.log(gLA_lo), np.log(gLA_dft), N))
    dA = rng.normal(m.dA, 0.15 * abs(m.dA), N)
    w2LA = 2.0 * m.wLA
    dwA = rng.normal(D.KRAYEV['edge_dwA'], sA, N)
    dw2 = rng.normal(D.KRAYEV['edge_dw2LA'], s2, N)
    f = (rng.uniform(0.0, frac_2la, N) if frac_2la else np.zeros(N))
    n = np.empty(N)
    for i in range(N):
        G = np.array([[-2.0 * gA[i] * m.wA / 100.0, dA[i]],
                      [-2.0 * gLA[i] * w2LA / 100.0,
                       f[i] * dA[i] * w2LA / m.wA]])
        Cd = np.diag([1 / sA ** 2, 1 / s2 ** 2])
        H = G.T @ Cd @ G + np.diag([1.0, 0.25])
        n[i] = np.linalg.solve(H, G.T @ Cd @ np.array([dwA[i], dw2[i]]))[1]
    return n * 1e13


def A16_robustness(mats, n_edge_nominal, nd_ref=1.3e12, w_edge=10.0,
                   N=20000, seed=11):
    m = mats['MoS2']
    out = {}

    # ---- 1. shift noise --------------------------------------------------
    rows = []
    for sA in NOISE_LEVELS:
        n = _mc_edge(mats, sA, 2.0 * sA, N, seed)
        rows.append(dict(sigma_A=sA, sigma_2LA=2.0 * sA,
                         prob_pos=float(np.mean(n > 0)), **_pct(n)))
    out['noise'] = rows
    fine = np.round(np.arange(0.10, 1.001, 0.05), 3)
    pp = [float(np.mean(_mc_edge(mats, s, 2 * s, 5000, seed) > 0))
          for s in fine]
    out['noise_curve'] = dict(sigma_A=fine.tolist(), prob_pos=pp)
    # noise at which the excess-electron reading falls to 95 % and 84 %
    pp_a = np.array(pp)
    for lvl, key in ((0.95, 'sigma_at_95'), (0.84, 'sigma_at_84')):
        idx = np.where(pp_a < lvl)[0]
        if len(idx) and idx[0] > 0:
            j = idx[0]
            out[key] = float(np.interp(lvl, [pp_a[j], pp_a[j - 1]],
                                       [fine[j], fine[j - 1]]))
        else:
            out[key] = float('nan')

    # ---- 2. charge coefficient of the overtone ---------------------------
    gLA_dft = m.dft['gamma_LA']
    gLA_lo = gLA_dft * (GAMMA_MEAS['WSe2']['gLA']
                        / mats['WSe2'].dft['gamma_LA'])
    w2LA = 2.0 * m.wLA
    fgrid = np.linspace(0.0, 4.0, 401)
    scan = {}
    for tag, gLA, gA in (('computed', gLA_dft, m.gA),
                         ('low', gLA_lo, m.gA),
                         ('measured_pair', gLA_lo, m.gA_meas)):
        ns = []
        for f in fgrid:
            G = np.array([[-2.0 * gA * m.wA / 100.0, m.dA],
                          [-2.0 * gLA * w2LA / 100.0, f * m.dA * w2LA / m.wA]])
            Cd = np.diag([1 / SIGMA_A ** 2, 1 / SIGMA_2LA ** 2])
            H = G.T @ Cd @ G + np.diag([1.0, 0.25])
            ns.append(float(np.linalg.solve(
                H, G.T @ Cd @ np.array([D.KRAYEV['edge_dwA'],
                                        D.KRAYEV['edge_dw2LA']]))[1] * 1e13))
        # without the prior the sign reverses exactly where the overtone's
        # charge-to-strain sensitivity equals that of A1': f* = gLA / gA
        scan[tag] = dict(gamma_LA=float(gLA), gamma_A=float(gA),
                         f_flip=float(gLA / gA), f=fgrid.tolist(), n=ns,
                         n_at_half=float(np.interp(0.5, fgrid, ns)))
    out['overtone_charge'] = scan
    n_f = _mc_edge(mats, SIGMA_A, SIGMA_2LA, N, seed, frac_2la=0.5)
    out['mc_with_overtone_charge'] = dict(frac_max=0.5,
                                          prob_pos=float(np.mean(n_f > 0)),
                                          **_pct(n_f))

    # ---- 3. closure at the interior spot ---------------------------------
    clos = {}
    for tag, gLA, gA in (('computed', gLA_dft, m.gA),
                         ('measured_pair', gLA_lo, m.gA_meas)):
        eps = D.KRAYEV['spot_dw2LA'] / (-2.0 * gLA * w2LA / 100.0)
        clos[tag] = dict(strain_pct=float(eps),
                         dwA_pred=float(-2.0 * gA * m.wA / 100.0 * eps),
                         dwA_meas=D.KRAYEV['spot_dwA'])
    out['spot_closure'] = clos

    # ---- 4. critical width against charge size, sign and halo ------------
    sig0 = transport.edge_line_charge(n_edge_nominal, w_edge)
    wc = {}
    for stack in ('HfO2_EOT1p5', 'SiO2_90nm', 'SiO2_300nm'):
        st = transport.STACK[stack]
        base = dict(halo_nm=D.HALO['GENTLE_nm'], Cox=transport.COX[stack],
                    Vov=1.5, Vds=1.0, Lch_nm=300.0, n_it_cm2=st['nit'],
                    eps_env=st['eps'])
        row = {}
        for k in CHARGE_SCALES:
            v = transport.critical_width(m, nd_ref,
                                         **dict(base, sigma_line_cm=k * sig0))
            row['%g' % k] = (float(v) if np.isfinite(v) else None)
        wc[stack] = row
    out['Wc_charge'] = wc
    st = transport.STACK['HfO2_EOT1p5']
    base = dict(Cox=transport.COX['HfO2_EOT1p5'], Vov=1.5, Vds=1.0,
                Lch_nm=300.0, n_it_cm2=st['nit'], eps_env=st['eps'])
    halo = {}
    for h in HALO_LEVELS:
        r = {}
        for k in (-1.0, 0.0, 1.0):
            v = transport.critical_width(
                m, nd_ref, **dict(base, halo_nm=h, sigma_line_cm=k * sig0))
            r['%g' % k] = (float(v) if np.isfinite(v) else None)
        halo['%g' % h] = r
    out['Wc_halo'] = halo

    # ---- 5. damage contrast inside the halo (assumed 20x in A8) ----------
    con = {}
    for c in (5.0, 20.0, 100.0):
        v = transport.critical_width(
            m, nd_ref, **dict(base, halo_nm=D.HALO['GENTLE_nm'],
                              sigma_line_cm=sig0, nd_edge=c * nd_ref))
        con['%g' % c] = (float(v) if np.isfinite(v) else None)
    out['Wc_contrast'] = con
    return out
