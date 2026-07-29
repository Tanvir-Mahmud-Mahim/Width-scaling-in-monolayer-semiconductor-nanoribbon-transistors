"""Quantum transport: NEGF transmission, Schottky contacts, Landauer current.

Why this layer exists
---------------------
The drift-diffusion solve in ``device.py`` is semiclassical.  It cannot
describe carriers that cross the channel without scattering, it cannot
describe tunnelling through the contact barrier, and it hides the contact
inside a lumped series resistance.  This module removes all three
restrictions:

  * ballistic transport, because the current is written as a Landauer
    integral whose channel transmission tends to one when the channel is
    shorter than the mean free path;
  * tunnelling, because the transmission through the contact barrier comes
    from a non-equilibrium Green's function solution of the effective-mass
    Schroedinger equation, not from a thermionic-emission formula;
  * a resolved Schottky barrier, because the metal-to-monolayer contact is
    represented by its actual potential profile, whose height and screening
    length set the transmission.

Method
------
Transport is along ``x``.  The monolayer is a two-dimensional parabolic band,
so the transverse momentum is a good quantum number and the transmission
depends only on the longitudinal energy.  The current per unit width is then

    I/W = (q g_s g_v / (4 pi^2 hbar))
          Int dEx T(Ex) Int dky [f_S(Ex + Ey) - f_D(Ex + Ey)]

with Ey = hbar^2 ky^2 / 2m*.  T(Ex) is assembled from three transmissions in
incoherent series, source barrier, channel and drain barrier,

    1/T = 1/T_S + 1/T_ch + 1/T_D - 2 ,

which is the standard composition for series scatterers separated by
phase-breaking regions.  The channel transmission is T_ch = lam/(lam + L)
with lam(E) the mean free path for backscattering carried by
``transport.transport_kernels``.  That choice is not cosmetic: it is what
makes the long-channel limit of this expression reproduce the drift-diffusion
current exactly, which ``verify`` checks.

Everything here is numpy and scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from .materials import HBAR, QE, M0, KB, EPS0, T300, Material
from . import transport

# vacuum permittivity in the same units used by the electrostatics below
EPS_MONO = 4.2          # out-of-plane dielectric constant of a 1H monolayer
T_MONO_NM = 0.65        # monolayer thickness used for the screening length


# ---------------------------------------------------------------------------
# NEGF transmission for a one-dimensional effective-mass barrier
# ---------------------------------------------------------------------------
def negf_transmission(U_eV, dx_m, m_eff, E_eV, U_lead_eV=None):
    """Transmission through the potential U(x) by the NEGF method.

    The effective-mass Hamiltonian is discretised on a uniform grid, giving a
    tridiagonal matrix with hopping t = hbar^2 / (2 m dx^2) and on-site energy
    U_i + 2t.  Semi-infinite leads at each end contribute the exact surface
    self-energy of a one-dimensional chain,

        Sigma = -t exp(i k a),    2t[1 - cos(k a)] = E - U_lead ,

    which is retarded (decaying) for propagating states and real for
    evanescent ones.  The transmission is the Caroli expression
    T = Tr[Gamma_S G Gamma_D G^dagger], which for a chain with one open mode
    at each end reduces to T = Gamma_S |G_1N|^2 Gamma_D.  Only the corner
    element of the Green's function is needed, so each energy costs one
    banded solve rather than a full inversion.

    Parameters
    ----------
    U_eV : potential energy on the grid, eV, measured from the same zero as E
    dx_m : grid spacing, metres
    m_eff : effective mass, kg
    E_eV : energies at which to evaluate the transmission, eV
    U_lead_eV : (left, right) lead potentials; defaults to the end values of U

    Returns the transmission at each requested energy.
    """
    U = np.asarray(U_eV, float) * QE
    E = np.atleast_1d(np.asarray(E_eV, float)) * QE
    n = U.size
    t = HBAR ** 2 / (2.0 * m_eff * dx_m ** 2)
    if U_lead_eV is None:
        UL, UR = U[0], U[-1]
    else:
        UL, UR = (np.asarray(U_lead_eV, float) * QE)

    def sigma(Ei, Ulead):
        """Surface self-energy of a semi-infinite chain, complex."""
        c = 1.0 - (Ei - Ulead) / (2.0 * t)
        c = complex(c)
        if c.real <= -1.0:            # above the lead band: still propagating
            ka = np.pi - 1j * np.arccosh(-c.real)
        elif c.real >= 1.0:           # below the lead band bottom: evanescent
            ka = 1j * np.arccosh(c.real)
        else:
            ka = np.arccos(c.real)
        return -t * np.exp(1j * ka)

    T = np.zeros(E.size)
    # banded storage for a tridiagonal complex matrix: 3 rows, n columns
    ab = np.zeros((3, n), complex)
    rhs = np.zeros(n, complex)
    for j, Ei in enumerate(E):
        sL, sR = sigma(Ei, UL), sigma(Ei, UR)
        gL, gR = -2.0 * sL.imag, -2.0 * sR.imag
        if gL <= 0.0 or gR <= 0.0:
            continue                                   # closed at one end
        diag = Ei - (U + 2.0 * t) + 0j
        diag[0] -= sL
        diag[-1] -= sR
        ab[0, 1:] = t                                  # super-diagonal
        ab[1, :] = diag
        ab[2, :-1] = t                                 # sub-diagonal
        rhs[:] = 0.0
        rhs[-1] = 1.0
        x = solve_banded((1, 1), ab, rhs)
        T[j] = gL * abs(x[0]) ** 2 * gR
    return T if T.size > 1 else float(T[0])


def analytic_rectangular_T(E_eV, V0_eV, L_m, m_eff):
    """Exact transmission through a rectangular barrier, for validation."""
    E = np.atleast_1d(np.asarray(E_eV, float)) * QE
    V0 = float(V0_eV) * QE
    k1 = np.sqrt(2.0 * m_eff * np.maximum(E, 1e-30)) / HBAR
    out = np.zeros(E.size)
    below = E < V0
    kap = np.sqrt(2.0 * m_eff * np.maximum(V0 - E, 0.0)) / HBAR
    with np.errstate(over='ignore'):
        sh = np.sinh(np.clip(kap[below] * L_m, 0, 700))
    out[below] = 1.0 / (1.0 + (V0 ** 2 * sh ** 2)
                        / (4.0 * E[below] * (V0 - E[below])))
    above = ~below
    if above.any():
        k2 = np.sqrt(2.0 * m_eff * (E[above] - V0)) / HBAR
        sn = np.sin(k2 * L_m)
        out[above] = 1.0 / (1.0 + (V0 ** 2 * sn ** 2)
                            / (4.0 * E[above] * (E[above] - V0)))
    return out


# ---------------------------------------------------------------------------
# the metal-to-monolayer Schottky contact
# ---------------------------------------------------------------------------
def screening_length_nm(eps_ox, t_ox_nm, eps_ch=EPS_MONO, t_ch_nm=T_MONO_NM):
    """Natural length of a single-gated two-dimensional channel, nm.

        lambda = sqrt(eps_ch t_ch t_ox / eps_ox)

    This is the distance over which the contact potential relaxes into the
    channel, and therefore the width of the Schottky barrier that a carrier
    has to tunnel through.  A thin high-permittivity gate makes it short,
    which is why tunnelling dominates on the high-kappa stack.
    """
    return float(np.sqrt(eps_ch * t_ch_nm * t_ox_nm / eps_ox))


def schottky_profile(phi_b_eV, ec_channel_eV, lam_nm, n_pts=1601,
                     span_nm=None, image_force=False, eps_ch=EPS_MONO):
    """Band-edge profile from the metal edge into the channel.

    The band edge is pinned at the Schottky barrier height phi_b above the
    metal Fermi level at the contact and relaxes exponentially to the value
    the gate sets in the channel,

        U(x) = Ec + (phi_b - Ec) exp(-x / lambda) .

    With image_force the classical image potential of a carrier in front of a
    metal plane is subtracted, which lowers and rounds the peak.  The image
    term is cut off at the point where it would diverge, which is the usual
    treatment and is where the effective-mass description stops being
    meaningful anyway.
    """
    lam = float(lam_nm)
    span = 12.0 * lam if span_nm is None else float(span_nm)
    x = np.linspace(0.0, span, n_pts)
    U = ec_channel_eV + (phi_b_eV - ec_channel_eV) * np.exp(-x / lam)
    if image_force:
        # image potential energy of a carrier a distance x from a metal
        # plane, q^2 / (16 pi eps0 eps_ch x), expressed in eV.  It is cut off
        # at 0.2 nm, roughly the metal-to-monolayer van der Waals gap, below
        # which neither the image construction nor the effective-mass
        # description means anything.
        xm = np.maximum(x, 0.2)
        img = QE / (16.0 * np.pi * EPS0 * eps_ch * xm * 1e-9)
        # the lowering cannot exceed the barrier itself
        img = np.minimum(img, abs(phi_b_eV - ec_channel_eV))
        U = U - img
    # the profile must return to the channel band edge at the far end
    U[-1] = ec_channel_eV
    dx = (span * 1e-9) / (n_pts - 1)
    return x, U, dx


def contact_transmission(mat: Material, phi_b_eV, ec_channel_eV, lam_nm,
                         E_eV, carrier='e', image_force=False):
    """Transmission through one Schottky contact, from the NEGF solution.

    Carriers above the barrier top are transmitted with the quantum
    reflection the NEGF gives; carriers below it tunnel.  Nothing is
    thermionic by assumption: both regimes come out of the same calculation.
    """
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    _, U, dx = schottky_profile(phi_b_eV, ec_channel_eV, lam_nm,
                                image_force=image_force)
    return negf_transmission(U, dx, m, E_eV,
                             U_lead_eV=(ec_channel_eV, ec_channel_eV))


# ---------------------------------------------------------------------------
# Landauer current for a two-dimensional channel
# ---------------------------------------------------------------------------
def _grids(nEx=260, nEy=160, Emax_kT=30.0, T=T300):
    """Longitudinal and transverse energy grids, joules.

    The transverse grid is uniform in the wavevector rather than in the
    energy, because the transverse integral is taken over dky.  A grid that
    is uniform in Ey crowds its points near ky = 0 and starves the trapezoid
    where the supply function still has weight, which costs about a per cent
    in the diffusive check below.
    """
    kT = KB * T
    Ex = np.linspace(1e-6 * kT, Emax_kT * kT, nEx)
    Ey = np.linspace(0.0, 1.0, nEy) ** 2 * (Emax_kT * kT)
    return Ex, Ey


def landauer_current(mat: Material, Ex_J, Ey_J, T_long, lam_of_E, L_nm,
                     mu_S_J, mu_D_J, carrier='e', T=T300):
    """Current per unit width, A/m, as a two-dimensional Landauer integral.

    The two transmissions enter on different variables, which is the whole
    reason the integral is written in two dimensions rather than one.  A
    contact barrier is a one-dimensional potential, so it conserves transverse
    momentum and its transmission depends on the longitudinal energy Ex
    alone.  Channel scattering in an isotropic band depends on the total
    energy Ex + Ey.  Collapsing either onto the other variable is wrong, and
    the diffusive check in ``verify`` is sensitive enough to catch it.

    T_long is the contact transmission on the Ex grid, or None for an ohmic
    contact.  lam_of_E is the mean free path on the total-energy grid, given
    as (E_grid, lam_grid).
    """
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    gs = 2.0 if carrier == 'e' else 1.0
    gv = 2.0
    kT = KB * T

    Etot = Ex_J[:, None] + Ey_J[None, :]
    Eg, lg = lam_of_E
    lam = np.interp(Etot, Eg, lg, left=lg[0], right=lg[-1])
    L = float(L_nm) * 1e-9
    Tch = lam / (lam + L)
    if T_long is None:
        Ttot = Tch
    else:
        Tc = np.asarray(T_long, float)[:, None]
        Tc = np.maximum(Tc, 1e-14)
        Ttot = 1.0 / np.maximum(1.0 / Tch + 2.0 / Tc - 2.0, 1.0)

    fS = 1.0 / (1.0 + np.exp(np.clip((Etot - mu_S_J) / kT, -300, 300)))
    fD = 1.0 / (1.0 + np.exp(np.clip((Etot - mu_D_J) / kT, -300, 300)))
    ky = np.sqrt(2.0 * m * Ey_J) / HBAR
    inner = 2.0 * np.trapezoid(Ttot * (fS - fD), ky, axis=1)
    pref = QE * gs * gv / (4.0 * np.pi ** 2 * HBAR)
    return float(pref * np.trapezoid(inner, Ex_J))


def device_current(mat: Material, W_nm, L_nm, nd_cm2, n_cm2, n_it_cm2,
                   Vds=1.0, carrier='e', eps_env=None, strain=0.0,
                   phi_b_eV=None, lam_nm=None, image_force=False,
                   nEx=260, nEy=160, Emax_kT=30.0):
    """On-current per unit width, uA/um, with contacts and channel resolved.

    One ampere per metre is one microampere per micrometre, so no unit
    conversion is needed on the way out.
    """
    Ex, Ey = _grids(nEx, nEy, Emax_kT)
    ker = transport.transport_kernels(mat, nd_cm2, strain, n_cm2, n_it_cm2,
                                      carrier, eps_env, W_nm)
    Ef = transport.fermi_level(mat, n_cm2, carrier)
    if phi_b_eV is None:
        Tc = None
    else:
        Tc = contact_transmission(mat, phi_b_eV, 0.0, lam_nm, Ex / QE,
                                  carrier, image_force)
    I = landauer_current(mat, Ex, Ey, Tc, (ker['E'], ker['lam']), L_nm,
                         Ef, Ef - Vds * QE, carrier)
    return dict(I_uA_um=I, Ex_eV=Ex / QE, T_contact=Tc, Ef_eV=Ef / QE,
                lam=ker['lam'], E=ker['E'])


def ballistic_current(mat: Material, n_cm2, Vds=1.0, carrier='e',
                      nEx=260, nEy=160, Emax_kT=30.0):
    """The transmission-one ceiling, uA/um."""
    Ex, Ey = _grids(nEx, nEy, Emax_kT)
    Ef = transport.fermi_level(mat, n_cm2, carrier)
    big = np.array([0.0, 1e3 * QE])
    return landauer_current(mat, Ex, Ey, None, (big, np.array([1e30, 1e30])),
                            1.0, Ef, Ef - Vds * QE, carrier)


def contact_iv(mat: Material, phi_b_eV, lam_nm, n_cm2, V_eV, carrier='e',
               image_force=False, nEx=400, nEy=300, Emax_kT=40.0):
    """Current per unit width through one Schottky contact, A/m.

    The metal-covered region and the channel are both treated as reservoirs
    of the same two-dimensional band, separated by the barrier profile.  The
    transmission comes from the NEGF solution, so thermionic emission over
    the barrier and field emission through it are the same calculation
    evaluated at different energies rather than two formulas bolted together.
    """
    Ex, Ey = _grids(nEx, nEy, Emax_kT)
    Ef = transport.fermi_level(mat, n_cm2, carrier)
    Tc = contact_transmission(mat, phi_b_eV, 0.0, lam_nm, Ex / QE, carrier,
                              image_force)
    big = np.array([0.0, 1e3 * QE])
    huge = np.array([1e30, 1e30])
    out = []
    for V in np.atleast_1d(np.asarray(V_eV, float)):
        out.append(landauer_current(mat, Ex, Ey, Tc, (big, huge), 1.0,
                                    Ef, Ef - V * QE, carrier))
    return np.array(out)


def contact_resistance(mat: Material, phi_b_eV, lam_nm, n_cm2, carrier='e',
                       dV=2e-3, image_force=False):
    """Low-bias contact resistance of one contact, ohm micrometre.

    With phi_b = 0 this returns the quantum limit: even a barrier-free
    contact to a two-dimensional channel has a finite resistance, because
    only a finite number of modes carries the current.  Any measured contact
    resistance has to lie above that value, which makes it a useful yardstick.
    """
    I = contact_iv(mat, phi_b_eV, lam_nm, n_cm2, [dV], carrier,
                   image_force)[0]
    # A/m and volts -> ohm.um
    return float(dV / I * 1e6)


def ballisticity(mat: Material, L_nm, nd_cm2, n_cm2, n_it_cm2, carrier='e',
                 eps_env=None, W_nm=None, strain=0.0):
    """Thermally averaged channel transmission lam/(lam + L).

    One means the channel is ballistic, zero means every carrier is
    backscattered.  The average is taken over the same Fermi window that
    weights the conductance, so it is the number that decides whether a
    semiclassical description of the channel is adequate.
    """
    ker = transport.transport_kernels(mat, nd_cm2, strain, n_cm2, n_it_cm2,
                                      carrier, eps_env, W_nm)
    E, lam = ker['E'], ker['lam']
    kT = KB * T300
    Ef = transport.fermi_level(mat, n_cm2, carrier)
    x = np.clip((E - Ef) / kT, -300, 300)
    w = np.exp(x) / (kT * (1.0 + np.exp(x)) ** 2) * np.sqrt(E)
    L = float(L_nm) * 1e-9
    T = lam / (lam + L)
    return float(np.trapezoid(T * w, E) / np.trapezoid(w, E)), ker


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify(mat: Material = None, carrier='e'):
    """Four independent checks of this module.

    1. the NEGF transmission of a rectangular barrier matches the closed-form
       result, in the tunnelling regime and above the barrier where the
       resonances are;
    2. a flat potential transmits perfectly, so the discretisation introduces
       no spurious reflection;
    3. the non-degenerate ballistic limit of the Landauer integral equals
       q n_s sqrt(k_B T / 2 pi m*), which fixes the prefactor;
    4. the long-channel limit of the same integral reproduces the
       drift-diffusion sheet conductance n_s q mu computed from the Boltzmann
       mobility, which is what justifies lam = (pi/2) v tau.
    """
    from .materials import all_materials
    mat = all_materials()['MoS2'] if mat is None else mat
    m = (mat.mcK if carrier == 'e' else mat.mvK) * M0
    out = {}

    # --- 1. rectangular barrier against the analytic transmission ---------
    V0, Lb = 0.30, 3.0e-9
    npts = 4801
    x = np.linspace(-4e-9, Lb + 4e-9, npts)
    U = np.where((x >= 0) & (x <= Lb), V0, 0.0)
    dx = x[1] - x[0]
    Eg = np.linspace(0.02, 0.60, 60)
    Tn = negf_transmission(U, dx, m, Eg, U_lead_eV=(0.0, 0.0))
    Ta = analytic_rectangular_T(Eg, V0, Lb, m)
    out['rect_barrier_max_abs_err'] = float(np.max(np.abs(Tn - Ta)))
    sub = Eg < V0
    out['tunnelling_max_rel_err'] = float(
        np.max(np.abs(Tn[sub] - Ta[sub]) / np.maximum(Ta[sub], 1e-30)))

    # --- 2. flat potential must transmit perfectly ------------------------
    Uf = np.zeros(601)
    Tf = negf_transmission(Uf, 5e-11, m, np.linspace(0.01, 0.5, 40),
                           U_lead_eV=(0.0, 0.0))
    out['flat_max_dev_from_unity'] = float(np.max(np.abs(Tf - 1.0)))

    # --- 3. ballistic limit against the analytic injection current --------
    n_nd = 1e9                       # deep in the non-degenerate regime,
                                     # where the reference formula is exact
    I_ball = ballistic_current(mat, n_nd, Vds=1.0, carrier=carrier,
                               nEx=600, nEy=400, Emax_kT=40.0)
    kT = KB * T300
    ns = n_nd * 1e4
    I_ref = QE * ns * np.sqrt(kT / (2.0 * np.pi * m))
    out['ballistic_rel_err'] = float(abs(I_ball - I_ref) / I_ref)
    out['v_inj_m_s'] = float(I_ball / (QE * ns))

    # --- 4. diffusive limit against the Boltzmann conductivity ------------
    # A long channel at small bias: the Landauer conductance per square must
    # equal n_s q mu with mu from the energy-resolved Boltzmann integral.
    n = 1e13
    nd, nit = 1e12, transport.N_IT_SIO2
    L = 200000.0                      # nm, deep in the diffusive limit
    # the bias has to be well below k_B T / q for the conductance to be the
    # linear-response one that the Boltzmann result describes
    dV = 1e-5
    ker = transport.transport_kernels(mat, nd, 0.0, n, nit, carrier)
    Ef = transport.fermi_level(mat, n, carrier)
    Ex, Ey = _grids(900, 600, 50.0)
    I = landauer_current(mat, Ex, Ey, None, (ker['E'], ker['lam']), L,
                         Ef, Ef - dV * QE, carrier)
    G_sq = I / dV * (L * 1e-9)                     # S per square
    mu_b, _, _ = transport.sheet_mobility_energy_resolved(
        mat, nd, n_cm2=n, n_it_cm2=nit, carrier=carrier)
    G_ref = QE * (n * 1e4) * (mu_b * 1e-4)
    out['diffusive_rel_err'] = float(abs(G_sq - G_ref) / G_ref)
    out['G_landauer'] = float(G_sq)
    out['G_boltzmann'] = float(G_ref)
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=1, default=float))
