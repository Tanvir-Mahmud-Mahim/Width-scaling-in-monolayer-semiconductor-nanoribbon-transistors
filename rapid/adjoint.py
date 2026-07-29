"""Physics-guided adjoint inversion of spectroscopic observables.

The forward operator g is the analytic physics chain in spectra.py, so its
linearisation G = dg/dtheta and the adjoint G^T are available in closed form.
Two solvers are provided.

single_point
    Gauss-Newton inversion of one spectrum with a Gaussian prior.  Returns the
    maximum a posteriori state and the Laplace posterior covariance.

invert_map
    Joint inversion of a whole hyperspectral image.  The pixels are coupled by
    a gradient-squared regulariser, so the misfit gradient is assembled by an
    adjoint sweep and the Gauss-Newton system is solved matrix free by
    conjugate gradients.  Cost is one forward and one adjoint evaluation per
    CG iteration, independent of the number of pixels.

Sign conventions follow Plessix, Geophys. J. Int. 167, 495 (2006).
"""
from __future__ import annotations

import numpy as np

from . import spectra
from .materials import Material

NP = 3          # number of latent parameters per pixel
NOBS = 4        # number of observables


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def apply_forward_linear(V, Jac):
    """Action of the linearised forward operator, (G v)_o = sum_p J_op v_p.

    Implemented explicitly rather than through a matrix product so that the
    dot-product test below exercises real code rather than a numpy identity.
    """
    V = np.asarray(V, float)
    out = np.zeros(V.shape[:-1] + (NOBS,))
    for o in range(NOBS):
        acc = np.zeros(V.shape[:-1])
        for q in range(NP):
            acc = acc + Jac[..., o, q] * V[..., q]
        out[..., o] = acc
    return out


def apply_adjoint(R, Jac):
    """Action of the adjoint operator, (G^T r)_p = sum_o J_op r_o."""
    R = np.asarray(R, float)
    out = np.zeros(R.shape[:-1] + (NP,))
    for q in range(NP):
        acc = np.zeros(R.shape[:-1])
        for o in range(NOBS):
            acc = acc + Jac[..., o, q] * R[..., o]
        out[..., q] = acc
    return out


def _laplacian(field):
    """Five-point Laplacian with Neumann (zero-flux) boundaries."""
    f = field
    out = -4.0 * f
    out += np.roll(f, 1, axis=0)
    out += np.roll(f, -1, axis=0)
    out += np.roll(f, 1, axis=1)
    out += np.roll(f, -1, axis=1)
    # correct the wrapped rows/columns for Neumann conditions
    out[0, :] += f[0, :] - f[-1, :]
    out[-1, :] += f[-1, :] - f[0, :]
    out[:, 0] += f[:, 0] - f[:, -1]
    out[:, -1] += f[:, -1] - f[:, 0]
    return out


# ---------------------------------------------------------------------------
# single-spectrum inversion
# ---------------------------------------------------------------------------
def single_point(y, sigma, mat: Material, theta0=None, prior=None,
                 prior_sigma=None, n_iter=60, tol=1e-12, verbose=False):
    """Gauss-Newton maximum a posteriori inversion of one spectrum.

    Parameters
    ----------
    y : (4,) measured observables, NaN entries are treated as missing
    sigma : (4,) measurement standard deviations
    prior, prior_sigma : (3,) Gaussian prior mean and width on theta

    Returns
    -------
    dict with keys theta, cov, resid, chi2, n_iter, converged
    """
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    mask = np.isfinite(y)
    if prior is None:
        prior = np.array([12.5, 0.0, 0.5])
    if prior_sigma is None:
        prior_sigma = np.array([2.0, 1.0, 1.0])
    prior = np.asarray(prior, float)
    Cp_inv = np.diag(1.0 / np.asarray(prior_sigma, float) ** 2)

    th = np.array(prior if theta0 is None else theta0, float)
    Cd_inv = np.diag(np.where(mask, 1.0 / sigma ** 2, 0.0))

    converged = False
    lam = 1e-3                                    # Levenberg damping
    for it in range(n_iter):
        g = spectra.forward(th, mat)[0]
        r = np.where(mask, g - y, 0.0)
        G = spectra.jacobian(th, mat)[0]
        # adjoint action: gradient of the misfit
        grad = G.T @ (Cd_inv @ r) + Cp_inv @ (th - prior)
        H = G.T @ Cd_inv @ G + Cp_inv
        try:
            step = np.linalg.solve(H + lam * np.diag(np.diag(H)), -grad)
        except np.linalg.LinAlgError:                       # pragma: no cover
            break
        th_new = th + step
        # simple backtracking on the objective
        def obj(t):
            gg = spectra.forward(t, mat)[0]
            rr = np.where(mask, gg - y, 0.0)
            return 0.5 * rr @ Cd_inv @ rr + 0.5 * (t - prior) @ Cp_inv @ (t - prior)
        f0, f1 = obj(th), obj(th_new)
        k = 0
        while f1 > f0 and k < 30:
            step *= 0.5
            th_new = th + step
            f1 = obj(th_new)
            k += 1
        if f1 > f0:                       # no decrease found, stop here
            converged = True
            break
        if np.max(np.abs(step)) < tol * (1 + np.max(np.abs(th))):
            th = th_new
            converged = True
            break
        th = th_new
        lam = max(lam * 0.5, 1e-8)
        if verbose:
            print(it, th, f1)

    G = spectra.jacobian(th, mat)[0]
    H = G.T @ Cd_inv @ G + Cp_inv
    cov = np.linalg.inv(H)
    g = spectra.forward(th, mat)[0]
    r = np.where(mask, g - y, 0.0)
    chi2 = float(r @ Cd_inv @ r)
    return dict(theta=th, cov=cov, resid=r, chi2=chi2, n_iter=it + 1,
                converged=converged, sigma_post=np.sqrt(np.diag(cov)))


def resolution_matrix(theta, sigma, mat: Material, prior_sigma):
    """Model resolution matrix R = (G^T Cd^-1 G + Cp^-1)^-1 G^T Cd^-1 G.

    Diagonal entries near one mean the datum constrains that parameter;
    entries near zero mean the answer is set by the prior.
    """
    G = spectra.jacobian(theta, mat)[0]
    Cd_inv = np.diag(1.0 / np.asarray(sigma, float) ** 2)
    Cp_inv = np.diag(1.0 / np.asarray(prior_sigma, float) ** 2)
    A = G.T @ Cd_inv @ G
    return np.linalg.solve(A + Cp_inv, A)


# ---------------------------------------------------------------------------
# hyperspectral map inversion by the adjoint-state method
# ---------------------------------------------------------------------------
def _flat(TH):
    return TH.reshape(-1)


def map_objective(TH, Y, sigma, mat, prior, Cp_inv_diag, alpha):
    """Objective and adjoint gradient for a whole map.

    TH : (ny, nx, 3)   latent state field
    Y  : (ny, nx, 4)   observables
    """
    ny, nx = TH.shape[:2]
    G = spectra.forward(TH.reshape(-1, NP), mat).reshape(ny, nx, NOBS)
    mask = np.isfinite(Y).astype(float)
    R = (G - Y) / sigma                                   # whitened residual
    R = np.where(np.isfinite(R), R, 0.0) * mask
    J_data = 0.5 * float(np.sum(R ** 2))

    dTH = TH - prior
    J_prior = 0.5 * float(np.sum(Cp_inv_diag * dTH ** 2))

    J_reg = 0.0
    for k in range(NP):
        gx = np.diff(TH[..., k], axis=1)
        gy = np.diff(TH[..., k], axis=0)
        J_reg += 0.5 * alpha[k] * float(np.sum(gx ** 2) + np.sum(gy ** 2))

    # ---- adjoint sweep -------------------------------------------------
    Jac = spectra.jacobian(TH.reshape(-1, NP), mat).reshape(ny, nx, NOBS, NP)
    grad = apply_adjoint(R * mask / sigma, Jac)            # G^T Cd^-1 r
    grad += Cp_inv_diag * dTH
    for k in range(NP):
        grad[..., k] -= alpha[k] * _laplacian(TH[..., k])
    return J_data + J_prior + J_reg, grad


def _gn_hessian_vec(V, TH, sigma, mat, Cp_inv_diag, alpha, mask=None):
    """Matrix-free Gauss-Newton Hessian action, H v."""
    ny, nx = TH.shape[:2]
    Jac = spectra.jacobian(TH.reshape(-1, NP), mat).reshape(ny, nx, NOBS, NP)
    if mask is None:
        mask = np.ones(TH.shape[:2] + (NOBS,))
    JV = apply_forward_linear(V, Jac) * mask / sigma       # forward linearised
    HV = apply_adjoint(JV * mask / sigma, Jac)             # adjoint
    HV = HV + Cp_inv_diag * V
    for k in range(NP):
        HV[..., k] -= alpha[k] * _laplacian(V[..., k])
    return HV


def invert_map(Y, sigma, mat: Material, prior=None, prior_sigma=None,
               alpha=(0.05, 0.05, 0.05), n_outer=12, n_cg=40, verbose=False):
    """Joint adjoint-state inversion of a hyperspectral observable map.

    Y : (ny, nx, 4) observables.  NaN marks a missing channel.
    """
    Y = np.asarray(Y, float)
    ny, nx, _ = Y.shape
    sigma = np.asarray(sigma, float)
    if prior is None:
        prior = np.array([12.5, 0.0, 0.5])
    if prior_sigma is None:
        prior_sigma = np.array([2.0, 1.0, 1.0])
    prior_f = np.broadcast_to(prior, (ny, nx, NP)).copy()
    Cp_inv_diag = 1.0 / np.asarray(prior_sigma, float) ** 2
    alpha = np.asarray(alpha, float)

    TH = prior_f.copy()
    mask = np.isfinite(Y).astype(float)
    hist = []
    for outer in range(n_outer):
        J, grad = map_objective(TH, Y, sigma, mat, prior_f, Cp_inv_diag, alpha)
        hist.append(J)
        # conjugate gradients on H dTH = -grad, matrix free
        b = -grad
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rs = float(np.sum(r * r))
        if rs < 1e-16:
            break
        for _ in range(n_cg):
            Hp = _gn_hessian_vec(p, TH, sigma, mat, Cp_inv_diag, alpha, mask)
            denom = float(np.sum(p * Hp))
            if denom <= 0:
                break
            a = rs / denom
            x += a * p
            r -= a * Hp
            rs_new = float(np.sum(r * r))
            if np.sqrt(rs_new) < 1e-8 * np.sqrt(np.sum(b * b)):
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        # line search
        step = 1.0
        for _ in range(25):
            Jn, _ = map_objective(TH + step * x, Y, sigma, mat, prior_f,
                                  Cp_inv_diag, alpha)
            if Jn < J:
                break
            step *= 0.5
        TH = TH + step * x
        if verbose:
            print(f'outer {outer}  J={J:.4e} -> {Jn:.4e}  step={step:.3f}')
        if abs(J - Jn) < 1e-10 * max(1.0, abs(J)):
            break
    J, grad = map_objective(TH, Y, sigma, mat, prior_f, Cp_inv_diag, alpha)
    hist.append(J)
    return dict(theta=TH, J=hist, grad_norm=float(np.linalg.norm(grad)))


# ---------------------------------------------------------------------------
# verification utilities
# ---------------------------------------------------------------------------
def check_adjoint(mat: Material, theta=None, seed=0):
    """Dot-product test  <G v, w> == <v, G^T w>  and Jacobian versus finite
    differences.  Returns the two relative errors."""
    rng = np.random.default_rng(seed)
    if theta is None:
        theta = np.array([12.8, -0.4, 0.6])
    Jac = spectra.jacobian(theta, mat)
    v = rng.normal(size=(1, NP))
    w = rng.normal(size=(1, NOBS))
    lhs = float(np.sum(apply_forward_linear(v, Jac) * w))
    rhs = float(np.sum(v * apply_adjoint(w, Jac)))
    G = Jac[0]
    err_adj = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    Gfd = spectra.jacobian_fd(theta, mat)[0]
    scale = np.maximum(np.abs(G), 1e-12)
    err_jac = float(np.max(np.abs(G - Gfd) / scale))
    return err_adj, err_jac


def check_map_gradient(mat: Material, ny=4, nx=5, seed=1):
    """Directional finite-difference test of the adjoint map gradient."""
    rng = np.random.default_rng(seed)
    TH = np.stack([rng.normal(12.8, 0.2, (ny, nx)),
                   rng.normal(-0.3, 0.1, (ny, nx)),
                   rng.normal(0.6, 0.1, (ny, nx))], axis=-1)
    Y = spectra.forward(TH.reshape(-1, NP), mat).reshape(ny, nx, NOBS)
    Y = Y + rng.normal(0, 1e-3, Y.shape)
    sigma = np.array([0.15, 0.15, 0.004, 0.004])
    prior = np.broadcast_to(np.array([12.5, 0.0, 0.5]), TH.shape).copy()
    Cp = 1.0 / np.array([2.0, 1.0, 1.0]) ** 2
    alpha = np.array([0.05, 0.05, 0.05])
    J0, g = map_objective(TH, Y, sigma, mat, prior, Cp, alpha)
    d = rng.normal(size=TH.shape)
    h = 1e-6
    Jp, _ = map_objective(TH + h * d, Y, sigma, mat, prior, Cp, alpha)
    Jm, _ = map_objective(TH - h * d, Y, sigma, mat, prior, Cp, alpha)
    fd = (Jp - Jm) / (2 * h)
    ad = float(np.sum(g * d))
    return abs(fd - ad) / abs(fd)
