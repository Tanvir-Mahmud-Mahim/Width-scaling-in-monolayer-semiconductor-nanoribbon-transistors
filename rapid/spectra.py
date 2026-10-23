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
# electrostatically gated monolayer WS2 the ground-state exciton shifts to the
# blue with electron density, the reduction of the exciton binding energy
# outweighing the bandgap renormalisation.  Chernikov et al. report
# dE/dV = +0.172 meV/V with the density rising by 0.8e12 cm^-2 per 10 V, which
# is +21.5 meV per 1e13 cm^-2.  This coefficient enters only the three-field
# image inversion demonstrated in the supplement; no physical result reported
# in the article depends on it.
CHI_DOP = 21.5

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

