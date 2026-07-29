"""All numerical experiments reported in the manuscript.

Running this module writes results.json with every number quoted in the text,
tables and figures.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import adjoint, spectra, transport, datasets as D
from .materials import all_materials, MATERIALS, Material

OUT = os.path.join(os.path.dirname(__file__), '..', 'results.json')

# nominal single-pixel measurement uncertainties for a hyperspectral
# Raman + differential-reflectance acquisition
SIGMA = np.array([0.15, 0.15, 0.0020, 0.0040])     # wE, wA, R_A, EA


# ---------------------------------------------------------------------------
def A1_adjoint_verification(mats):
    out = {}
    for name, m in mats.items():
        ea, ej = adjoint.check_adjoint(m)
        out[name] = dict(dot_product_error=ea, jacobian_vs_fd=ej,
                         map_gradient_error=adjoint.check_map_gradient(m))
    return out


def A2_conditioning(mats):
    """Sensitivity, conditioning and resolution of the inverse problem."""
    out = {}
    theta = np.array([12.7, -0.3, 0.5])
    prior_sigma = np.array([2.0, 1.0, 1.0])
    for name, m in mats.items():
        G = spectra.jacobian(theta, m)[0]
        Gw = G / SIGMA[:, None]
        s = np.linalg.svd(Gw, compute_uv=False)
        R = adjoint.resolution_matrix(theta, SIGMA, m, prior_sigma)
        res = adjoint.single_point(spectra.forward(theta, m)[0], SIGMA, m)
        # what happens if the exciton channel is dropped
        y = spectra.forward(theta, m)[0]
        y2 = y.copy()
        y2[3] = np.nan
        res2 = adjoint.single_point(y2, SIGMA, m)
        y3 = y.copy()
        y3[2] = np.nan
        res3 = adjoint.single_point(y3, SIGMA, m)
        out[name] = dict(
            singular_values=s.tolist(),
            condition_number=float(s[0] / s[-1]),
            resolution_diag=np.diag(R).tolist(),
            sigma_post=res['sigma_post'].tolist(),
            sigma_post_no_exciton=res2['sigma_post'].tolist(),
            sigma_post_no_defect_channel=res3['sigma_post'].tolist(),
            nd_rel_err_percent=float(res['sigma_post'][0] * np.log(10) * 100),
        )
    return out


def A3_map_recovery(mat: Material, ny=48, nx=48, seed=7, noise=1.0,
                    model_error=0.03):
    """Synthetic hyperspectral map: a grain boundary carrying compressive
    strain, a doped patch, and a defective corner.

    To avoid an inverse crime the data are generated with coefficients that
    differ from those used in the inversion by `model_error` (3 per cent), and
    the noise is heavy tailed (Student t with five degrees of freedom) rather
    than Gaussian.  The inversion still assumes Gaussian noise and the
    unperturbed coefficients."""
    import copy
    rng = np.random.default_rng(seed)
    gen = copy.copy(mat)
    for attr, sgn in (('gE', +1), ('gA', -1), ('C_A', +1), ('gaugeA', -1),
                      ('dA', +1)):
        setattr(gen, attr, getattr(mat, attr) * (1.0 + sgn * model_error))
    yy, xx = np.mgrid[0:ny, 0:nx]
    # true fields
    u = 12.2 + 0.9 * np.exp(-(((xx - 36) / 7.0) ** 2 + ((yy - 34) / 7.0) ** 2))
    gb = np.exp(-((xx - yy) / 2.5) ** 2)
    s = -0.65 * gb
    n = 0.25 + 0.9 * (1.0 / (1.0 + np.exp(-(yy - 30) / 2.0))) * \
        (1.0 / (1.0 + np.exp(-(20 - xx) / 2.0)))
    TH = np.stack([u, s, n], axis=-1)
    Y = spectra.forward(TH.reshape(-1, 3), gen).reshape(ny, nx, 4)
    Y = Y + noise * SIGMA * rng.standard_t(5, Y.shape) / np.sqrt(5 / 3.0)

    r = adjoint.invert_map(Y, SIGMA, mat, alpha=(0.30, 0.30, 0.30),
                           n_outer=15, n_cg=60)
    TR = r['theta']
    err = TR - TH
    return dict(true=TH, obs=Y, rec=TR, J=r['J'],
                rmse=[float(np.sqrt(np.mean(err[..., k] ** 2))) for k in range(3)],
                bias=[float(np.mean(err[..., k])) for k in range(3)],
                grad_norm=r['grad_norm'])


def A4_nattoo(mats):
    """Two independent disorder-activated Raman channels must report the same
    defect density; the resulting point-defect ceiling is then compared with
    the measured mobility."""
    m = mats['MoS2']
    rows = []
    ratio = []
    for e in D.NATTOO:
        nd = e['R_LA'] / (m.C_A * 1e-14)
        nd_hi = (e['R_LA'] + e['R_LA_e']) / (m.C_A * 1e-14)
        nd_lo = (e['R_LA'] - e['R_LA_e']) / (m.C_A * 1e-14)
        LD = 1e7 / np.sqrt(nd)
        mu_ceiling, parts = transport.sheet_mobility(m, nd)
        ratio.append(e['R_sh'] / e['R_LA'])
        rows.append(dict(label=e['label'], R_sh=e['R_sh'], R_LA=e['R_LA'],
                         nd=nd, nd_lo=nd_lo, nd_hi=nd_hi, LD_nm=LD,
                         mu_meas=e['mu'], mu_ceiling=mu_ceiling,
                         deficit=(mu_ceiling / e['mu']) if e['mu'] == e['mu'] else None,
                         parts={k: float(v) for k, v in parts.items()}))
    ratio = np.array(ratio)
    return dict(rows=rows, channel_ratio_mean=float(ratio.mean()),
                channel_ratio_std=float(ratio.std(ddof=1)),
                channel_ratio_rel_scatter=float(ratio.std(ddof=1) / ratio.mean()),
                channel_ratio_ALD=float(ratio[:2].mean()),
                channel_ratio_sputtered=float(ratio[2:].mean()),
                channel_ratio_all=ratio.tolist())


def A5_krayev(mats):
    """Invert the published tip-enhanced Raman shifts of MoS2 nanoribbons.

    Two channels are available: the A1' shift, which responds to both strain
    and electron density, and the 2LA(M) shift, which is a zone-edge acoustic
    overtone and responds almost only to strain.  Together they separate the
    two effects without any further assumption.
    """
    m = mats['MoS2']
    gLA = m.dft.get('gamma_LA', 1.30)
    w2LA = 2.0 * (m.wLA or 227.6)
    dw2LA_deps = -2.0 * gLA * w2LA / 100.0          # cm^-1 per %
    dwA_deps = -2.0 * m.gA * m.wA / 100.0
    dwA_dn = m.dA                                   # cm^-1 per 1e13 cm^-2

    def invert(dwA, dw2LA, sA=0.15, s2=0.30):
        G = np.array([[dwA_deps, dwA_dn], [dw2LA_deps, 0.0]])
        Cd = np.diag([1 / sA ** 2, 1 / s2 ** 2])
        H = G.T @ Cd @ G + np.diag([1 / 1.0 ** 2, 1 / 2.0 ** 2])
        rhs = G.T @ Cd @ np.array([dwA, dw2LA])
        x = np.linalg.solve(H, rhs)
        cov = np.linalg.inv(H)
        return x, np.sqrt(np.diag(cov))

    edge, edge_e = invert(D.KRAYEV['edge_dwA'], D.KRAYEV['edge_dw2LA'])
    spot, spot_e = invert(D.KRAYEV['spot_dwA'], D.KRAYEV['spot_dw2LA'])
    # what a single-channel (A1' only) analysis would conclude
    strain_only = D.KRAYEV['edge_dwA'] / dwA_deps
    dope_only = D.KRAYEV['edge_dwA'] / dwA_dn
    return dict(gamma_LA=float(gLA), w2LA=float(w2LA),
                dw2LA_deps=float(dw2LA_deps), dwA_deps=float(dwA_deps),
                dwA_dn=float(dwA_dn),
                edge=dict(strain_pct=float(edge[0]), n_1e13=float(edge[1]),
                          strain_err=float(edge_e[0]), n_err=float(edge_e[1]),
                          n_cm2=float(edge[1] * 1e13)),
                spot=dict(strain_pct=float(spot[0]), n_1e13=float(spot[1]),
                          strain_err=float(spot_e[0]), n_err=float(spot_e[1]),
                          n_cm2=float(spot[1] * 1e13)),
                naive_strain_only_pct=float(strain_only),
                naive_doping_only_cm2=float(dope_only * 1e13))


def A5c_overtone_doping(mats, rel=None):
    """Bound on the assumed charge insensitivity of the 2LA(M) overtone.

    The separation assumes that the zone-edge acoustic overtone does not
    renormalise with carrier density, whereas the out-of-plane A1' mode does
    through symmetry-selective electron-phonon coupling.  There is no gated
    measurement of the 2LA(M) coefficient, so here the assumption is relaxed:
    lambda_2LA is set to a fraction `rel` of the A1' coefficient scaled by the
    frequency ratio, and the recovered edge state is recomputed.
    """
    m = mats['MoS2']
    gLA = m.dft.get('gamma_LA', 1.30)
    w2LA = 2.0 * (m.wLA or 227.6)
    dw2LA_deps = -2.0 * gLA * w2LA / 100.0
    dwA_deps = -2.0 * m.gA * m.wA / 100.0
    dwA_dn = m.dA
    if rel is None:
        rel = np.linspace(0.0, 0.5, 26)
    out = []
    for f in rel:
        dw2_dn = f * dwA_dn * (w2LA / m.wA)
        G = np.array([[dwA_deps, dwA_dn], [dw2LA_deps, dw2_dn]])
        Cd = np.diag([1 / 0.15 ** 2, 1 / 0.30 ** 2])
        H = G.T @ Cd @ G + np.diag([1.0, 0.25])
        e = np.linalg.solve(H, G.T @ Cd @ np.array([D.KRAYEV['edge_dwA'],
                                                    D.KRAYEV['edge_dw2LA']]))
        out.append(dict(rel=float(f), edge_n=float(e[1] * 1e13),
                        edge_eps=float(e[0])))
    return out


def A5b_krayev_robustness(mats, gvals=None):
    """How the edge assignment depends on the acoustic Grueneisen parameter.

    The separation of charge from strain rests on the 2LA(M) overtone being
    much more strain sensitive than A1'.  Here the assignment is repeated
    across the full plausible range of gamma_LA to show that the conclusion
    does not depend on its precise value.
    """
    m = mats['MoS2']
    if gvals is None:
        gvals = np.linspace(0.4, 2.6, 45)
    w2LA = 2.0 * (m.wLA or 227.6)
    dwA_deps = -2.0 * m.gA * m.wA / 100.0
    dwA_dn = m.dA
    out = []
    for g in gvals:
        dw2 = -2.0 * g * w2LA / 100.0
        G = np.array([[dwA_deps, dwA_dn], [dw2, 0.0]])
        Cd = np.diag([1 / 0.15 ** 2, 1 / 0.30 ** 2])
        H = G.T @ Cd @ G + np.diag([1.0, 0.25])
        e = np.linalg.solve(H, G.T @ Cd @ np.array([D.KRAYEV['edge_dwA'],
                                                    D.KRAYEV['edge_dw2LA']]))
        sp = np.linalg.solve(H, G.T @ Cd @ np.array([D.KRAYEV['spot_dwA'],
                                                     D.KRAYEV['spot_dw2LA']]))
        out.append(dict(gamma=float(g), edge_n=float(e[1] * 1e13),
                        edge_eps=float(e[0]), spot_n=float(sp[1] * 1e13),
                        spot_eps=float(sp[0])))
    return out


def A6_peng(mats):
    """Invert the hyperspectral differential-reflectance blue shift measured
    at a WS2 grain boundary, and the V-doping series."""
    m = mats['WS2']
    dE = D.PENG['dE_gb']                       # eV blue shift
    strain = dE * 1000.0 / m.gaugeA            # percent (gauge is meV per %)
    # doping series: attribute the redshift entirely to added holes
    ser = []
    for x, E in D.PENG['vdoped_peak']:
        dn = (E - D.PENG['vdoped_peak'][0][1]) * 1000.0 / spectra.CHI_DOP
        ser.append(dict(V_percent=x, EA=E, n_1e13=float(dn)))
    return dict(gb_strain_pct=float(strain),
                gb_strain_reported=D.PENG['strain_gb_reported'],
                gauge=m.gaugeA, vseries=ser)


def A7_transport_validation(mats):
    """Compare the transport layer with independent published results."""
    out = {}
    ws2 = mats['WS2']
    dos = []
    for nd, mu in D.DOSSENA['points']:
        pred, parts = transport.sheet_mobility(ws2, nd, n_cm2=1.9e13,
                                               n_it_cm2=0.0)
        dos.append(dict(nd=nd, mu_ref=mu, mu_pred=float(pred),
                        ratio=float(pred / mu)))
    out['dossena'] = dos
    yang = {}
    for k, (nd, err) in D.YANG_DEFECTS.items():
        name = 'WS2' if 'WS2' in k else 'WSe2'
        car = 'e' if name == 'WS2' else 'h'
        mu, parts = transport.sheet_mobility(mats[name], nd, carrier=car)
        yang[k] = dict(nd=nd, mu_pred=float(mu),
                       parts={a: float(b) for a, b in parts.items()})
    out['yang'] = yang
    return out


def A8_ribbons(mats, nd_ref=1.3e12, sigma_line=None, w_edge=10.0):
    """Nanoribbon scaling, critical width and family screening.

    The fixed edge charge is not assumed: it is the value returned by the
    adjoint inversion of the published tip-enhanced Raman data (A5).
    """
    if sigma_line is None:
        n_edge = A5_krayev(mats)['edge']['n_cm2']
        sigma_line = transport.edge_line_charge(n_edge, w_edge)
    st = transport.STACK['HfO2_EOT1p5']
    base = dict(halo_nm=D.HALO['RIE_nm'], sigma_line_cm=sigma_line,
                Cox=transport.COX['HfO2_EOT1p5'], Vov=1.5, Vds=1.0,
                Lch_nm=300.0, n_it_cm2=st['nit'], eps_env=st['eps'])
    W = np.geomspace(5.0, 1000.0, 140)
    curves, Wc_tab = {}, {}
    for name, m in mats.items():
        car = 'h' if name == 'WSe2' else 'e'
        kw = dict(base, carrier=car)
        I = transport.ribbon_current_density_uA_um(m, W, nd_ref, **kw)
        mu = transport.ribbon_mobility(m, W, nd_ref, halo_nm=base['halo_nm'],
                                       carrier=car)
        dVT = transport.threshold_shift(sigma_line, W, base['Cox'])
        curves[name] = dict(W=W.tolist(), I=np.asarray(I).tolist(),
                            mu=np.asarray(mu).tolist(), dVT=dVT.tolist())
        Wc_tab[name] = float(transport.critical_width(m, nd_ref, **kw))

    # critical width versus gate-stack capacitance and patterning halo
    halos = np.geomspace(1.0, 200.0, 40)
    Wc_halo = {tag: [float(transport.critical_width(
        mats['MoS2'], nd_ref, **dict(base, halo_nm=h, Cox=cox,
                                     n_it_cm2=transport.STACK[tag]['nit'],
                                     eps_env=transport.STACK[tag]['eps'])))
        for h in halos] for tag, cox in transport.COX.items()}

    # design map: on-current per width over (W, n_d)
    Wg = np.geomspace(8.0, 400.0, 70)
    ndg = np.geomspace(1e11, 1e14, 70)
    Z = np.zeros((len(ndg), len(Wg)))
    for i, nd in enumerate(ndg):
        Z[i] = transport.ribbon_current_density_uA_um(mats['MoS2'], Wg, nd,
                                                      **base)
    # growth-technology bands, from the inverted defect densities
    tech = {'exfoliated': 2e11, 'SS-CVD monolayer': 1.3e12,
            'ALD annealed': 5.4e12, 'ALD as-grown': 7.3e12,
            'sputtered': 2.2e13, 'MOCVD WSe2': 5.3e13}
    tech_mu = {k: float(transport.sheet_mobility(mats['MoS2'], v)[0])
               for k, v in tech.items()}

    # helium-ion versus reactive-ion patterning
    halo_cmp = {}
    for tag, h in (('RIE', D.HALO['RIE_nm']), ('HIM', D.HALO['HIM_nm'])):
        kw = dict(base, halo_nm=h)
        halo_cmp[tag] = dict(
            W=W.tolist(),
            I=np.asarray(transport.ribbon_current_density_uA_um(
                mats['MoS2'], W, nd_ref, **kw)).tolist(),
            Wc=float(transport.critical_width(mats['MoS2'], nd_ref, **kw)))

    # separate the two causes of the width limit
    split = {}
    for tag, cox in transport.COX.items():
        kw = dict(base, Cox=cox, n_it_cm2=transport.STACK[tag]['nit'],
                  eps_env=transport.STACK[tag]['eps'])
        split[tag] = dict(
            both=float(transport.critical_width(mats['MoS2'], nd_ref, **kw)),
            halo_only=float(transport.critical_width(
                mats['MoS2'], nd_ref, **dict(kw, sigma_line_cm=0.0))),
            charge_only=float(transport.critical_width(
                mats['MoS2'], nd_ref, **dict(kw, halo_nm=0.0))))

    # sensitivity of the critical width to the assumed edge width
    wedges = np.array([5.0, 10.0, 20.0])
    wedge_scan = {}
    for we in wedges:
        sl = transport.edge_line_charge(
            A5_krayev(mats)['edge']['n_cm2'], we)
        wedge_scan['%g' % we] = {
            tag: float(transport.critical_width(
                mats['MoS2'], nd_ref,
                **dict(base, Cox=cox, sigma_line_cm=sl,
                       n_it_cm2=transport.STACK[tag]['nit'],
                       eps_env=transport.STACK[tag]['eps'])))
            for tag, cox in transport.COX.items()}

    # gate-stack comparison at fixed halo
    cox_cmp = {}
    for tag, cox in transport.COX.items():
        kw = dict(base, Cox=cox, n_it_cm2=transport.STACK[tag]['nit'],
                  eps_env=transport.STACK[tag]['eps'])
        cox_cmp[tag] = dict(
            W=W.tolist(),
            I=np.asarray(transport.ribbon_current_density_uA_um(
                mats['MoS2'], W, nd_ref, **kw)).tolist(),
            Wc=float(transport.critical_width(mats['MoS2'], nd_ref, **kw)),
            I_ref=float(transport.ribbon_current_density_uA_um(
                mats['MoS2'], transport.W_REF_NM, nd_ref, **kw)),
            dVT_25nm=float(transport.threshold_shift(sigma_line, 25.0, cox)))
    return dict(nd_ref=nd_ref, sigma_line_cm=float(sigma_line),
                w_edge_nm=w_edge, Vov=base['Vov'], split=split,
                wedge_scan=wedge_scan, curves=curves, Wc=Wc_tab,
                halos=halos.tolist(), Wc_halo=Wc_halo,
                map=dict(W=Wg.tolist(), nd=ndg.tolist(), I=Z.tolist()),
                tech=tech, tech_mu=tech_mu, halo=halo_cmp, cox=cox_cmp)


def A9_predictions(mats):
    """Quantities the framework predicts that have not yet been measured."""
    out = {}
    for name, m in mats.items():
        out[name] = dict(C_A_nm2=float(m.C_A), C_E_nm2=float(m.C_E),
                         Udef_eVnm2=float(m.Udef) if m.Udef else None,
                         gamma_E=float(m.gE), gamma_A=float(m.gA),
                         gamma_LA=float(m.dft.get('gamma_LA', float('nan'))),
                         dwA_dn=float(m.dA),
                         dgap_deps=float(m.dft.get('dgap_deps',
                                                   float('nan'))))
    return out


def A10_self_consistent(mats, nd_ref=1.3e12, sigma_line=None, w_edge=10.0):
    """Self-consistent Poisson and drift-diffusion solve of the same devices.

    The compact expression used for the wide scans evaluates the current from
    the source-end sheet density, which omits the reduction of the channel
    charge towards the drain and the quantum capacitance of a two-dimensional
    channel, and cannot represent the feedback of a series contact resistance.
    This stage solves the same devices without those approximations, using the
    open-source device simulator DEVSIM, and reports

      * the four solver checks that run with the code,
      * the ratio between the two models across the whole width range,
      * the critical width computed both ways, which is what every conclusion
        about scaling actually rests on,
      * the on-currents of the measured devices with the contact resistance
        included, and the contact resistance the p-type WSe2 measurement
        implies.
    """
    from . import device
    if sigma_line is None:
        n_edge = A5_krayev(mats)['edge']['n_cm2']
        sigma_line = transport.edge_line_charge(n_edge, w_edge)
    st = transport.STACK['HfO2_EOT1p5']
    kw = dict(Cox=transport.COX['HfO2_EOT1p5'], n_it_cm2=st['nit'],
              eps_env=st['eps'], Vov=1.5, Vds=1.0, Lch_nm=300.0,
              halo_nm=D.HALO['RIE_nm'], sigma_line_cm=sigma_line)
    compact_kw = dict(halo_nm=D.HALO['RIE_nm'], sigma_line_cm=sigma_line,
                      Cox=transport.COX['HfO2_EOT1p5'], Vov=1.5, Vds=1.0,
                      Lch_nm=300.0, n_it_cm2=st['nit'], eps_env=st['eps'])

    out = dict(verify=device.verify(mats['MoS2']))

    # --- the two models compared across the width range ----------------
    W = np.geomspace(5.0, 4000.0, 26)
    I_sc = np.array([device.solve_iv(mats['MoS2'], w, nd_ref, Rc_ohm_um=0.0,
                                     **kw)['I_uA_um'] for w in W])
    I_cp = np.asarray(transport.ribbon_current_density_uA_um(
        mats['MoS2'], W, nd_ref, **compact_kw), float)
    ratio = I_sc / np.maximum(I_cp, 1e-30)
    keep = W >= 12.0
    out['compare'] = dict(W=W.tolist(), I_sc=I_sc.tolist(),
                          I_compact=I_cp.tolist(), ratio=ratio.tolist(),
                          ratio_mean=float(ratio[keep].mean()),
                          ratio_spread=float(ratio[keep].max()
                                             - ratio[keep].min()))

    # --- critical width both ways --------------------------------------
    f_sc = I_sc / I_sc[-1]
    f_cp = I_cp / I_cp[-1]

    def _cross(f):
        for i in range(1, len(f)):
            if f[i - 1] < 0.5 <= f[i]:
                x0, x1 = np.log(W[i - 1]), np.log(W[i])
                return float(np.exp(x0 + (0.5 - f[i - 1]) * (x1 - x0)
                                    / (f[i] - f[i - 1])))
        return float('nan')

    out['Wc_self_consistent'] = _cross(f_sc)
    out['Wc_compact'] = _cross(f_cp)

    # --- self-consistent I(W) curves for the measured materials --------
    Rc0 = D.PENA['Rc_ohm_um']
    Wc_curve = np.geomspace(9.0, 1000.0, 22)
    curves = {}
    for mname in ('WS2', 'MoS2', 'WSe2'):
        car = 'h' if mname == 'WSe2' else 'e'
        curves[mname] = dict(
            W=Wc_curve.tolist(),
            I=[device.solve_iv(mats[mname], w, nd_ref, carrier=car,
                               Rc_ohm_um=Rc0, **kw)['I_uA_um']
               for w in Wc_curve])
    out['curves'] = curves

    # --- the measured devices, with the contact resistance -------------
    Rc = Rc0
    dev = {}
    for name, w, car, meas in (('MoS2_25nm', 25.0, 'e', D.PENA['Ion']['MoS2_25nm']),
                               ('MoS2_75nm', 75.0, 'e', D.PENA['Ion']['MoS2_75nm']),
                               ('WS2_43nm', 43.0, 'e', D.PENA['Ion']['WS2']),
                               ('WSe2_43nm', 43.0, 'h', D.PENA['Ion']['WSe2'])):
        mname = name.split('_')[0]
        r0 = device.solve_iv(mats[mname], w, nd_ref, carrier=car,
                             Rc_ohm_um=0.0, **kw)
        rc = device.solve_iv(mats[mname], w, nd_ref, carrier=car,
                             Rc_ohm_um=Rc, **kw)
        dev[name] = dict(I_ideal=r0['I_uA_um'], I_with_Rc=rc['I_uA_um'],
                         measured=meas, mu=r0['mu'],
                         Vds_internal=rc['Vds_internal'])
    out['devices'] = dev

    # --- contact resistance implied by the p-type WSe2 measurement -----
    target = D.PENA['Ion']['WSe2']
    lo, hi = 0.0, 4.0e4
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        v = device.solve_iv(mats['WSe2'], 43.0, nd_ref, carrier='h',
                            Rc_ohm_um=mid, **kw)['I_uA_um']
        if v > target:
            lo = mid
        else:
            hi = mid
    out['Rc_WSe2_ohm_um'] = float(0.5 * (lo + hi))
    out['Rc_measured_ohm_um'] = float(Rc)
    return out


def main():
    mats = all_materials()
    res = {}
    res['adjoint_verification'] = A1_adjoint_verification(mats)
    res['conditioning'] = A2_conditioning(mats)
    rec = A3_map_recovery(mats['MoS2'])
    np.savez_compressed(os.path.join(os.path.dirname(__file__), '..',
                                     'map_recovery.npz'),
                        true=rec['true'], rec=rec['rec'], obs=rec['obs'])
    res['map_recovery'] = dict(rmse=rec['rmse'], bias=rec['bias'],
                               J=rec['J'], grad_norm=rec['grad_norm'])
    res['nattoo'] = A4_nattoo(mats)
    res['krayev'] = A5_krayev(mats)
    res['krayev_robust'] = A5b_krayev_robustness(mats)
    res['overtone_doping'] = A5c_overtone_doping(mats)
    res['peng'] = A6_peng(mats)
    res['transport_validation'] = A7_transport_validation(mats)
    res['ribbons'] = A8_ribbons(mats)
    res['self_consistent'] = A10_self_consistent(mats)
    res['predictions'] = A9_predictions(mats)
    res['materials'] = {n: dict(wE=m.wE, wA=m.wA, wLA=m.wLA, gE=m.gE, gA=m.gA,
                                C_A=m.C_A, Udef=m.Udef, mu_ph=m.mu_ph,
                                gauge=m.gaugeA, dA=m.dA, C2D=m.C2D)
                        for n, m in mats.items()}
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: v for k, v in r.items()
                      if k in ('adjoint_verification', 'map_recovery',
                               'krayev', 'peng')}, indent=1, default=float))
