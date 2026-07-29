"""Differentiable spectroscopic forward model for monolayer TMDs.

Latent state
------------
    theta = (u, s, N)
        u : log10 of the areal point-defect density n_d in cm^-2
        s : biaxial strain in percent (negative = compressive)
        N : free-electron density in units of 1e13 cm^-2

Observables
-----------
    y = (wE, wA, R_A, EA)
        wE  : E' (E_2g) Raman frequency, cm^-1
        wA  : A1' (A_1g) Raman frequency, cm^-1
        R_A : disorder-activated intensity ratio I[LA(M)] / I[A1'], dimensionless
        EA  : A-exciton energy from differential reflectance, eV

Optional prediction channels (not used in the inversion) are the A-exciton
linewidth and the Raman full widths.

Every coefficient is either computed from first principles in this work
(Grueneisen parameters, deformation potentials, defect potentials) or taken
from a published measurement.  The model is analytic, so its Jacobian and
therefore the adjoint operator are available in closed form.
"""
from __future__ import annotations

import numpy as np

from .materials import Material

LN10 = np.log(10.0)

# Shift of the A-exciton resonance with electron doping, meV per 1e13 cm^-2.
# Chernikov et al., Phys. Rev. Lett. 115, 126802 (2015) show that in
# electrostatically gated monolayer WS2 the ground-state exciton moves by
# several tens of meV over this density range, the reduction of the exciton
# binding energy outweighing the bandgap renormalisation so that the net shift
# is to the blue.  A nominal +40 meV per 1e13 cm^-2 is used.  This coefficient
# enters only the three-field image inversion demonstrated in the supplement;
# no physical result reported in the article depends on it.
CHI_DOP = 40.0

OBS_NAMES = ['wE', 'wA', 'R_A', 'EA']
OBS_UNITS = ['cm$^{-1}$', 'cm$^{-1}$', '', 'eV']


def defect_length(u):
    """Mean inter-defect distance L_D in nm from u = log10(n_d [cm^-2])."""
    return 1.0e7 / np.sqrt(10.0 ** u)


def forward(theta, mat: Material):
    """Map the latent state to the observable vector.

    theta : array_like, shape (..., 3)
    returns array, shape (..., 4)
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    u, s, N = theta[..., 0], theta[..., 1], theta[..., 2]
    nd = 10.0 ** u                                    # cm^-2

    wE = mat.wE * (1.0 - 2.0 * mat.gE * s / 100.0) + mat.dE * N
    wA = mat.wA * (1.0 - 2.0 * mat.gA * s / 100.0) + mat.dA * N
    R_A = mat.C_A * nd * 1e-14                        # C_A in nm^2
    EA = mat.EA0 + mat.gaugeA * s / 1000.0 + CHI_DOP * N / 1000.0

    out = np.stack([wE, wA, R_A, EA], axis=-1)
    return out


def jacobian(theta, mat: Material):
    """Analytic Jacobian dg/dtheta, shape (..., 4, 3)."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    u = theta[..., 0]
    nd = 10.0 ** u
    shp = theta.shape[:-1]
    J = np.zeros(shp + (4, 3))

    # wE
    J[..., 0, 1] = -2.0 * mat.gE * mat.wE / 100.0
    J[..., 0, 2] = mat.dE
    # wA
    J[..., 1, 1] = -2.0 * mat.gA * mat.wA / 100.0
    J[..., 1, 2] = mat.dA
    # R_A
    J[..., 2, 0] = mat.C_A * 1e-14 * nd * LN10
    # EA
    J[..., 3, 1] = mat.gaugeA / 1000.0
    J[..., 3, 2] = CHI_DOP / 1000.0
    return J


def jacobian_fd(theta, mat: Material, h=1e-6):
    """Central-difference Jacobian, used to verify the analytic adjoint."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = theta.shape[-1]
    base = forward(theta, mat)
    J = np.zeros(base.shape + (n,))
    for i in range(n):
        dp = np.zeros(n)
        dp[i] = h
        J[..., i] = (forward(theta + dp, mat) - forward(theta - dp, mat)) / (2 * h)
    return J


# ---------------------------------------------------------------------------
# auxiliary channel, reported but never used in the inversion
# ---------------------------------------------------------------------------
def raman_fwhm(theta, mat: Material, G0E=3.0, G0A=4.0, gGam=6.0, bA=3.33):
    """Raman full widths (cm^-1).

    The doping term for A1' is anchored on Chakraborty et al., Phys. Rev. B
    85, 161403 (2012): +6 cm^-1 broadening at n = 1.8e13 cm^-2.  The disorder
    term uses a phonon lifetime that scales with n_d through the correlation
    length L_C = alpha L_D of Mignuzzi et al., Phys. Rev. B 91, 195411 (2015);
    gGam is not calibrated against any data set and this channel is therefore
    excluded from the inversion.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    u, N = theta[..., 0], theta[..., 2]
    LD = defect_length(u)
    GE = G0E + 100.0 * gGam / (mat.alpha_E * LD) ** 2
    GA = G0A + bA * N + 100.0 * gGam / (mat.alpha_A * LD) ** 2
    return np.stack([GE, GA], axis=-1)
