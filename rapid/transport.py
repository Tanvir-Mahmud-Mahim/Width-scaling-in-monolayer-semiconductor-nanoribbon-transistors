"""Transport forward model: from the latent material state to device metrics.

Chain
-----
    theta = (log10 n_d, strain, carrier density)
        -> valley occupations and scattering rates
        -> sheet mobility of a wide channel
        -> nanoribbon mobility and on-current for a channel of width W
        -> critical width W_c

Scattering channels
-------------------
    phonons              anchored on published first-principles mobilities;
                         a strain dependence through the secondary-valley
                         occupation is available but multiplies zero in every
                         result reported, which is computed at zero strain
    neutral point defects short-range 2D Fermi-golden-rule scattering with a
                         defect potential U calibrated once against a published
                         first-principles disordered-monolayer transport result
    charged impurities   screened 2D Coulomb scattering from the gate dielectric
    ribbon edges         diffuse boundary scattering plus a process-induced
                         damage halo of elevated defect density
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .materials import (HBAR, QE, M0, KB, EPS0, T300, EPS_SIO2, EPS_HFO2,
                        EPS_TOP, U_DEFECT, Material)

# interface trapped-charge density of a thermal SiO2 / 2D semiconductor
# interface, cm^-2.  Single environmental constant of the transport layer,
# calibrated once so that a defect-free monolayer MoS2 channel on SiO2 has the
# room-temperature field-effect mobility reported for state-of-the-art devices.
# Re-fixed against the same anchors when the mobility moved from a
# single-energy relaxation time to the full Boltzmann integral: screened
# Coulomb scattering is the one channel whose rate depends on the wavevector,
# so it is the one whose calibration the refinement touches.  The neutral
# point-defect potential U_DEFECT is unchanged, because a short-range
# scatterer has an energy-independent rate in two dimensions and its anchor
# carries no interface charge.
N_IT_SIO2 = 1.867e13
N_IT_HFO2 = 5.848e13

# specularity of a top-down patterned nanoribbon edge (0 = fully diffuse)
P_SPEC = 0.0

# strain derivatives of the secondary-valley separations, eV per % biaxial
# strain, taken from the literature (see materials.PROVENANCE)
DEQK_DEPS = -0.06
DEGK_DEPS = 0.12


def _mean_speed(mstar):
    """Mean thermal speed of a two-dimensional Maxwellian gas, m/s."""
    return np.sqrt(np.pi * KB * T300 / (2.0 * mstar))


def valley_occupation(mat: Material, strain, carrier='e'):
    """Fraction of carriers in the secondary valley at 300 K.

    Biaxial tension lowers the Q (Lambda) conduction valley towards K and
    switches on intervalley scattering; compression does the opposite.  The
    strain derivative of the separation is a literature value: the 4x4 k-mesh
    used here does not resolve the shallow Q minimum well enough for the
    derivative computed in this work to be trusted, so that number is reported
    as a diagnostic only.
    """
    kT = KB * T300 / QE
    if carrier == 'e':
        dE0, m1, m2, g2 = mat.EQK, mat.mcK, mat.mcQ, 6.0
        slope = DEQK_DEPS                            # eV per % strain
    else:
        dE0, m1, m2, g2 = mat.EGK, mat.mvK, mat.mvG, 1.0
        slope = DEGK_DEPS
    dE = dE0 + slope * strain
    w1 = 2.0 * m1
    w2 = g2 * m2 * np.exp(-dE / kT)
    return w2 / (w1 + w2)


def mobility_phonon(mat: Material, strain=0.0, carrier='e'):
    """Phonon-limited mobility including strain-tuned intervalley scattering."""
    f2 = valley_occupation(mat, strain, carrier)
    f20 = valley_occupation(mat, 0.0, carrier)
    mu0 = mat.mu_ph
    # intervalley scattering scales with the available final-state weight
    r = (1.0 + 9.0 * f2) / (1.0 + 9.0 * f20)
    return mu0 / r


def mobility_point_defect(mat: Material, nd_cm2, carrier='e'):
    """Short-range neutral point-defect scattering, 2D Fermi golden rule.

        1/tau = m* n_d U^2 / hbar^3 ,   mu = e hbar^3 / (m*^2 n_d U^2)

    U (eV nm^2) is the areal integral of the defect potential.  It is
    calibrated once, against the first-principles disordered-monolayer
    transport result cited in materials.PROVENANCE, and then held fixed.
    """
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    U_eV_nm2 = mat.Udef if mat.Udef is not None else U_DEFECT
    U = U_eV_nm2 * QE * 1e-18                      # J m^2
    nd = np.asarray(nd_cm2, float) * 1e4           # m^-2
    nd = np.maximum(nd, 1.0)
    mu = QE * HBAR ** 3 / (m ** 2 * nd * U ** 2)
    return mu * 1e4                                # cm^2 / V s


def mobility_charged(mat: Material, n_it_cm2, n_cm2, carrier='e',
                     eps_env=None, d_setback_nm=0.3):
    """Screened 2D Coulomb scattering from interface charge.

    The impurity sheet sits a distance d below the channel, which introduces
    the standard exp(-2qd) form factor.  eps_env is the average of the
    dielectric constants above and below the monolayer.
    """
    if float(n_it_cm2) <= 0:
        return np.inf
    if eps_env is None:
        eps_env = (EPS_SIO2 + EPS_TOP) / 2.0
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    n = max(float(n_cm2), 1e11) * 1e4
    kF = np.sqrt(2.0 * np.pi * n)                  # single spin-valley pair
    qs = m * QE ** 2 / (2.0 * np.pi * EPS0 * eps_env * HBAR ** 2)   # 2D TF
    th = np.linspace(1e-4, np.pi, 400)
    q = 2.0 * kF * np.sin(th / 2.0)
    V = QE ** 2 / (2.0 * EPS0 * eps_env * (q + qs)) \
        * np.exp(-2.0 * q * d_setback_nm * 1e-9)
    integrand = V ** 2 * (1.0 - np.cos(th))
    avg = np.trapezoid(integrand, th) / np.pi
    nit = float(n_it_cm2) * 1e4
    inv_tau = m * nit * avg / (2 * np.pi * HBAR ** 3) * (2 * np.pi)
    mu = QE / (m * inv_tau)
    return mu * 1e4


def sheet_mobility(mat: Material, nd_cm2, strain=0.0, n_cm2=1e13,
                   n_it_cm2=N_IT_SIO2, carrier='e', eps_env=None):
    """Matthiessen combination of all wide-channel scattering mechanisms."""
    mu_ph = mobility_phonon(mat, strain, carrier)
    mu_pd = mobility_point_defect(mat, nd_cm2, carrier)
    mu_ci = mobility_charged(mat, n_it_cm2, n_cm2, carrier, eps_env=eps_env)
    inv = 1.0 / mu_ph + 1.0 / mu_pd + 1.0 / mu_ci
    return 1.0 / inv, dict(ph=mu_ph, pd=mu_pd, ci=mu_ci)


# ---------------------------------------------------------------------------
# energy-resolved transport kernel
# ---------------------------------------------------------------------------
# The mobilities above are relaxation-time results evaluated at a single
# carrier energy.  That is exact for the two channels whose rate is
# energy-independent in two dimensions (short-range point defects and, in the
# equipartition limit, acoustic phonons) but not for screened Coulomb
# scattering, whose matrix element depends on the wavevector, nor for diffuse
# edge scattering, whose rate is proportional to the carrier speed.  The
# functions below carry tau(E) for every channel and perform the Boltzmann
# transport integral over the Fermi window, which is what a transport
# calculation should do.


def dos_2d(mat: Material, carrier='e', gv=2.0):
    """Two-dimensional density of states, states per joule per m^2.

    g_v = 2 for the K and K' valleys of a 1H monolayer.  The spin degeneracy
    is 2 for the conduction band, whose spin splitting at K is a few tens of
    meV, but 1 for the valence band, where the spin-orbit splitting at K is
    0.15 eV in MoS2 and 0.46 eV in WSe2 and therefore far larger than k_B T.
    """
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    gs = 2.0 if carrier == 'e' else 1.0
    return gs * gv * m / (2.0 * np.pi * HBAR ** 2)


def fermi_level(mat: Material, n_cm2, carrier='e', T=T300):
    """Fermi level above the band edge, joules, for a 2D parabolic band.

    Inverts n_s = g_2D k_B T ln[1 + exp(eta)] analytically.
    """
    g2 = dos_2d(mat, carrier)
    kT = KB * T
    n = max(float(n_cm2), 1e9) * 1e4
    x = n / (g2 * kT)
    # eta = log(exp(x) - 1), written so that neither branch overflows
    if x > 1e-8:
        eta = x + np.log1p(-np.exp(-x))
    else:
        eta = np.log(x)
    return kT * eta


@lru_cache(maxsize=256)
def _tau_charged_cached(mstar, n_it_cm2, eps_env, d_setback_nm, nE, Emax_kT,
                        nth=400):
    """tau_ci on the standard energy grid, cached.

    The screened Coulomb rate depends on the material only through the
    effective mass, and on the device only through the interface charge
    density and the dielectric environment.  It does not depend on the defect
    density or on the ribbon width, which are the two quantities the sweeps
    vary, so computing it once per distinct environment turns the width and
    defect-density maps from minutes into seconds.
    """
    kT = KB * T300
    E = np.linspace(1e-4 * kT, Emax_kT * kT, nE)
    k = np.sqrt(2.0 * mstar * E) / HBAR
    if n_it_cm2 <= 0:
        return np.full(nE, np.inf)
    qs = mstar * QE ** 2 / (2.0 * np.pi * EPS0 * eps_env * HBAR ** 2)
    th = np.linspace(1e-4, np.pi, nth)
    q = 2.0 * k[:, None] * np.sin(th[None, :] / 2.0)
    V = QE ** 2 / (2.0 * EPS0 * eps_env * (q + qs)) \
        * np.exp(-2.0 * q * d_setback_nm * 1e-9)
    avg = np.trapezoid(V ** 2 * (1.0 - np.cos(th))[None, :], th, axis=1) / np.pi
    inv_tau = mstar * (n_it_cm2 * 1e4) * avg / HBAR ** 3
    return np.where(inv_tau > 0, 1.0 / np.maximum(inv_tau, 1e-300), np.inf)


def _tau_charged(mat, n_it_cm2, k, carrier='e', eps_env=None,
                 d_setback_nm=0.3, nth=400):
    """Momentum relaxation time for screened Coulomb scattering at wavevector k.

    Same matrix element as mobility_charged, but evaluated at the wavevector
    of the state rather than at the Fermi wavevector, so the energy dependence
    of the screened Coulomb cross-section is kept.
    """
    if float(n_it_cm2) <= 0:
        return np.full_like(np.asarray(k, float), np.inf)
    if eps_env is None:
        eps_env = (EPS_SIO2 + EPS_TOP) / 2.0
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    qs = m * QE ** 2 / (2.0 * np.pi * EPS0 * eps_env * HBAR ** 2)
    th = np.linspace(1e-4, np.pi, nth)
    k = np.atleast_1d(np.asarray(k, float))
    q = 2.0 * k[:, None] * np.sin(th[None, :] / 2.0)
    V = QE ** 2 / (2.0 * EPS0 * eps_env * (q + qs)) \
        * np.exp(-2.0 * q * d_setback_nm * 1e-9)
    avg = np.trapezoid(V ** 2 * (1.0 - np.cos(th))[None, :], th, axis=1) / np.pi
    nit = float(n_it_cm2) * 1e4
    inv_tau = m * nit * avg / HBAR ** 3
    return np.where(inv_tau > 0, 1.0 / np.maximum(inv_tau, 1e-300), np.inf)


def transport_kernels(mat: Material, nd_cm2, strain=0.0, n_cm2=1e13,
                      n_it_cm2=N_IT_SIO2, carrier='e', eps_env=None,
                      W_nm=None, p=P_SPEC, nE=600, Emax_kT=40.0):
    """Energy-resolved relaxation time, velocity and mean free path.

    Returns a dict with the energy grid measured from the band edge and, on
    that grid, the relaxation time of every channel, the total relaxation
    time, the group velocity and the mean free path for backscattering.

    The mean free path uses lambda(E) = (pi/2) v(E) tau(E), which is the
    factor that makes the diffusive limit of the Landauer expression identical
    to the Boltzmann conductivity for a two-dimensional parabolic band.  That
    identity is checked in quantum.verify.
    """
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    kT = KB * T300
    E = np.linspace(1e-4 * kT, Emax_kT * kT, nE)          # joules, from Ec
    k = np.sqrt(2.0 * m * E) / HBAR
    v = HBAR * k / m

    # phonons: anchored on a published first-principles mobility.  In two
    # dimensions the acoustic deformation-potential rate is proportional to
    # the density of states, which is energy independent, so a constant tau
    # is the correct reduction of that anchor and reproduces mu_ph exactly.
    tau_ph = m * mobility_phonon(mat, strain, carrier) * 1e-4 / QE
    # short-range neutral point defects: energy independent in 2D
    tau_pd = m * mobility_point_defect(mat, nd_cm2, carrier) * 1e-4 / QE
    # screened Coulomb: evaluated at the wavevector of each state
    if eps_env is None:
        eps_env = (EPS_SIO2 + EPS_TOP) / 2.0
    tau_ci = _tau_charged_cached(m, float(n_it_cm2), float(eps_env), 0.3,
                                 int(nE), float(Emax_kT))

    inv = 1.0 / tau_ph + 1.0 / tau_pd + 1.0 / tau_ci
    chans = dict(ph=np.full_like(E, tau_ph), pd=np.full_like(E, tau_pd),
                 ci=tau_ci)
    if W_nm is not None:
        # diffuse edge scattering: the rate is the speed divided by the width,
        # so unlike the single-energy form it is not set by a thermal average
        tau_ed = np.asarray(W_nm, float) * 1e-9 / (v * (1.0 - p))
        inv = inv + 1.0 / tau_ed
        chans['ed'] = tau_ed
    tau = 1.0 / inv
    return dict(E=E, k=k, v=v, tau=tau, tau_channels=chans, m=m,
                lam=0.5 * np.pi * v * tau)


def sheet_mobility_energy_resolved(mat: Material, nd_cm2, strain=0.0,
                                   n_cm2=1e13, n_it_cm2=N_IT_SIO2,
                                   carrier='e', eps_env=None, W_nm=None,
                                   p=P_SPEC, nE=600):
    """Boltzmann mobility from the full energy integral.

        sigma = e^2 g_2D / m* * int dE (-df/dE) E tau(E)
        n_s   = int dE g_2D f(E)
        mu    = sigma / (e n_s)

    A constant tau makes this identical to mu = e tau / m*, so the function
    reduces exactly to sheet_mobility when every channel is energy
    independent.  That degenerate case is checked in verify_energy_resolved.
    """
    ker = transport_kernels(mat, nd_cm2, strain, n_cm2, n_it_cm2, carrier,
                            eps_env, W_nm, p, nE)
    E, tau, m = ker['E'], ker['tau'], ker['m']
    g2 = dos_2d(mat, carrier)
    kT = KB * T300
    Ef = fermi_level(mat, n_cm2, carrier)
    x = np.clip((E - Ef) / kT, -300.0, 300.0)
    mdf = np.exp(x) / (kT * (1.0 + np.exp(x)) ** 2)        # -df/dE
    ns = g2 * kT * np.log1p(np.exp(np.clip(Ef / kT, -300.0, 300.0)))
    sigma = QE ** 2 * g2 / m * np.trapezoid(mdf * E * tau, E)
    mu = sigma / (QE * max(ns, 1e-30))
    # per-channel mobilities on the same footing, for the decomposition plot
    parts = {}
    for name, tc in ker['tau_channels'].items():
        s = QE ** 2 * g2 / m * np.trapezoid(mdf * E * tc, E)
        parts[name] = float(s / (QE * max(ns, 1e-30)) * 1e4)
    return float(mu * 1e4), parts, ker


def verify_energy_resolved(mat: Material, n_cm2=1e13, carrier='e'):
    """The energy integral must reproduce mu = e tau / m* for a constant tau.

    Every channel is made energy independent by switching off the Coulomb and
    edge terms, leaving only the phonon and point-defect times, so the exact
    Drude answer is known.
    """
    mu_int, _, _ = sheet_mobility_energy_resolved(
        mat, 1e12, n_cm2=n_cm2, n_it_cm2=0.0, carrier=carrier, W_nm=None)
    mu_ref, _ = sheet_mobility(mat, 1e12, n_cm2=n_cm2, n_it_cm2=0.0,
                               carrier=carrier)
    return dict(mu_integral=float(mu_int), mu_drude=float(mu_ref),
                rel_err=float(abs(mu_int - mu_ref) / mu_ref))


# ---------------------------------------------------------------------------
# nanoribbon layer
# ---------------------------------------------------------------------------
def edge_scattering_mobility(mat: Material, W_nm, carrier='e', p=P_SPEC):
    """Diffuse edge (boundary) scattering, Fuchs-Sondheimer limit in 2D."""
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    v = _mean_speed(m)
    W = np.asarray(W_nm, float) * 1e-9
    tau = W / (v * (1.0 - p))
    return QE * tau / m * 1e4


def ribbon_mobility(mat: Material, W_nm, nd_bulk, nd_edge=None, halo_nm=5.0,
                    strain=0.0, n_cm2=1e13, n_it_cm2=N_IT_SIO2, carrier='e',
                    eps_env=None, halo_contrast=20.0, energy_resolved=True):
    """Width-averaged nanoribbon mobility.

    The ribbon is treated as two parallel conducting strips: an interior of
    width W - 2*halo with the bulk defect density, and two damaged edge strips
    of width halo with an elevated defect density.  Both strips additionally
    suffer diffuse boundary scattering set by the full ribbon width.

    With energy_resolved the strip mobilities come from the Boltzmann
    integral, so the wavevector dependence of Coulomb scattering and the speed
    dependence of edge scattering are carried properly.  The single-energy
    route is kept because it is what the compact model needs and because the
    comparison between the two is reported in the article.
    """
    W = np.atleast_1d(np.asarray(W_nm, float))
    scalar = np.ndim(W_nm) == 0
    nd_edge = nd_bulk * halo_contrast if nd_edge is None else nd_edge

    if energy_resolved:
        mu_int = np.array([sheet_mobility_energy_resolved(
            mat, nd_bulk, strain, n_cm2, n_it_cm2, carrier, eps_env,
            W_nm=float(w))[0] for w in W])
        mu_dam = np.array([sheet_mobility_energy_resolved(
            mat, nd_edge, strain, n_cm2, n_it_cm2, carrier, eps_env,
            W_nm=float(w))[0] for w in W])
    else:
        mu_edge_geom = edge_scattering_mobility(mat, W, carrier)
        mi, _ = sheet_mobility(mat, nd_bulk, strain, n_cm2, n_it_cm2, carrier,
                               eps_env)
        md, _ = sheet_mobility(mat, nd_edge, strain, n_cm2, n_it_cm2, carrier,
                               eps_env)
        mu_int = 1.0 / (1.0 / mi + 1.0 / mu_edge_geom)
        mu_dam = 1.0 / (1.0 / md + 1.0 / mu_edge_geom)

    w_int = np.clip(W - 2.0 * halo_nm, 0.0, None)
    w_dam = np.minimum(W, 2.0 * halo_nm)
    out = (w_int * mu_int + w_dam * mu_dam) / W
    return float(out[0]) if scalar else out


# gate stacks: oxide capacitance (F/cm^2), interface trap density (cm^-2) and
# the dielectric environment seen by the monolayer.
COX = dict(SiO2_300nm=1.15e-8, SiO2_90nm=3.84e-8,
           SiO2_30nm=1.15e-7, HfO2_EOT1p5=2.30e-6)
STACK = {
    'SiO2_300nm': dict(nit=N_IT_SIO2, eps=(EPS_SIO2 + EPS_TOP) / 2.0),
    'SiO2_90nm': dict(nit=N_IT_SIO2, eps=(EPS_SIO2 + EPS_TOP) / 2.0),
    'SiO2_30nm': dict(nit=N_IT_SIO2, eps=(EPS_SIO2 + EPS_TOP) / 2.0),
    'HfO2_EOT1p5': dict(nit=N_IT_HFO2, eps=(EPS_SIO2 + EPS_HFO2) / 2.0),
}
# High-field saturation velocity, cm/s.  Measured for monolayer MoS2 by
# Smithe et al., Nano Lett. 18, 4516 (2018): (3.4 +/- 0.4)e6 cm/s at 300 K
# after correcting for self-heating.  No comparable measurement exists for
# monolayer WS2, MoSe2 or WSe2, so the MoS2 value is used for all four rather
# than a guessed material dependence.
V_SAT = 3.4e6
VSAT = {m: V_SAT for m in ('MoS2', 'WS2', 'MoSe2', 'WSe2')}


def edge_line_charge(n_edge_cm2, w_edge_nm):
    """Fixed edge charge per unit edge length, cm^-1, from the areal density
    and width returned by the tip-enhanced Raman inversion."""
    return float(n_edge_cm2) * float(w_edge_nm) * 1e-7


def threshold_shift(sigma_line_cm, W_nm, Cox):
    """Width-dependent threshold-voltage shift from fixed edge charge, V.

    Both edges contribute, and the charge is shared over the ribbon width:
        dV_T = q * 2 * sigma_line / (Cox * W)
    """
    W = np.asarray(W_nm, float) * 1e-7             # cm
    return QE * 2.0 * float(sigma_line_cm) / (Cox * W)


def ribbon_current_density_uA_um(mat: Material, W_nm, nd_bulk,
                                 sigma_line_cm=0.0, nd_edge=None,
                                 halo_nm=5.0, strain=0.0, Vov=2.0,
                                 Cox=COX['HfO2_EOT1p5'], n_it_cm2=N_IT_HFO2,
                                 Vds=1.0, Lch_nm=300.0, carrier='e',
                                 vsat=None, eps_env=None):
    """Drain current per unit width in uA/um (1 A/m is numerically 1 uA/um).

    The gate overdrive is reduced by the width-dependent threshold shift
    produced by the fixed edge charge, and the carrier velocity saturates.
    """
    W = np.asarray(W_nm, float)
    dVT = threshold_shift(sigma_line_cm, W, Cox)
    Vov_eff = np.clip(Vov - dVT, 0.0, None)
    n_cm2 = Cox * np.maximum(Vov_eff, 1e-3) / QE               # cm^-2
    # The screened-Coulomb mobility depends on the local carrier density, and
    # the carrier density depends on width through the threshold shift, so the
    # mobility is evaluated width by width.  Averaging over the width grid
    # first would make the answer depend on the grid.
    n_arr = np.atleast_1d(np.asarray(n_cm2, float))
    W_arr = np.broadcast_to(np.atleast_1d(W), n_arr.shape)
    mu = np.array([ribbon_mobility(mat, float(w), nd_bulk, nd_edge, halo_nm,
                                   strain, float(n), n_it_cm2, carrier,
                                   eps_env)
                   for w, n in zip(W_arr.ravel(), n_arr.ravel())])
    mu = mu.reshape(n_arr.shape)
    if np.ndim(n_cm2) == 0:
        mu = float(mu.ravel()[0])
    vs = (VSAT.get(mat.name, 3.0e6) if vsat is None else vsat)  # cm/s
    E = Vds / (Lch_nm * 1e-7)                                   # V/cm
    v = mu * E / (1.0 + mu * E / vs)                            # cm/s
    Qs = QE * n_cm2                                             # C/cm^2
    return Qs * v * 100.0                                       # A/m == uA/um


W_REF_NM = 4000.0     # reference wide-channel width for the 50 % criterion


def critical_width(mat: Material, nd_bulk, frac=0.5, W_ref=W_REF_NM,
                   Wmin=3.0, Wmax=2000.0, **kw):
    """Width at which the on-current per unit width falls to `frac` of its
    wide-channel value.  Returned in nm; NaN if the criterion is never met."""
    Iref = ribbon_current_density_uA_um(mat, W_ref, nd_bulk, **kw)
    W = np.geomspace(Wmin, Wmax, 600)
    I = ribbon_current_density_uA_um(mat, W, nd_bulk, **kw)
    target = frac * Iref
    below = I < target
    if not below.any() or below.all():
        return float('nan')
    i = np.argmax(~below)
    if i == 0:
        return float('nan')
    x0, x1 = np.log(W[i - 1]), np.log(W[i])
    y0, y1 = I[i - 1], I[i]
    return float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))
