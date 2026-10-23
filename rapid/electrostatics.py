"""Two-dimensional electrostatics of edge charge in a gated nanoribbon.

Equation (1) of the article, dV_T = 2 q sigma / (C_ox W), spreads the edge
charge uniformly over the ribbon width. That is the parallel-plate limit and
holds only when the ribbon is much wider than the gate dielectric is thick
and the charge sits close to the edges. This module solves the problem
exactly within linear electrostatics, so that published threshold voltages
can be inverted without that assumption and both the line charge and the
width over which it sits can be read from them.

Geometry. The monolayer is a sheet at z = 0 with an in-plane two-dimensional
polarisability (Keldysh form, screening length r0), on a gate dielectric of
thickness t and permittivity eps_ox, on a grounded gate; above it is a medium
of permittivity eps_top. The edge charge is a strip of width w_e inside each
edge of a ribbon of width W, carrying sigma charges per unit length per edge.
Fourier transforming along the width, Gauss's law at the sheet gives

    phi(k) = rho(k) / ( eps0 |k| [eps_top + eps_ox coth(|k| t) + 2 r0 |k|] ),

which reduces to the parallel-plate value rho t / (eps0 eps_ox) for a uniform
sheet (k -> 0) and to the Keldysh potential for a free-standing layer.

In the subthreshold regime that defines V_T,cc there are almost no free
carriers in the channel, so the potential above is the whole response, the
channel potential follows the gate one to one, and the current per width is
proportional to the width average of exp(q phi(x) / kT). The threshold
shift at constant current is therefore

    dV_T = -(kT/q) ln[ (1/W) int_0^W exp(q phi(x)/kT) dx ],

which equals Eq. (1) for a charge spread uniformly over the ribbon.  For a
charge held near the edges it does not tend to Eq. (1) at any width: the
current flows through the unperturbed interior, which is the physical point
of this module.  Checks run in `verify()`: the uniform-sheet limit, and the
ratio to Eq. (1) of a 5 nm edge strip on a 2 um ribbon, which is far below
one for that reason.  Further checks (finite differences, strong inversion,
mesh) are in `exact.checks()`.
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
KT300 = 0.025852  # eV


def _rho_hat(k, W, w_e, sigma_per_m, profile='strip'):
    """Cosine transform of the two-edge charge profile (C/m^2), symmetric
    about the ribbon centre (origin at the centre).

    profile='strip': a uniform strip of width w_e inside each edge, carrying
    sigma charges per metre per edge.
    profile='exp': a density N0 exp(-d/w_e) at depth d from each edge, with
    N0 w_e = sigma (the line charge per edge of a semi-infinite sheet); in a
    ribbon the two tails overlap and are cut off at the opposite edge."""
    if profile == 'strip':
        we = min(w_e, W / 2.0)
        a, b = W / 2.0 - we, W / 2.0
        dens = QE * sigma_per_m / w_e      # same areal density as in a wide ribbon
        with np.errstate(invalid='ignore', divide='ignore'):
            f = np.where(k > 0, 2.0 * (np.sin(k * b) - np.sin(k * a)) / np.where(k > 0, k, 1.0),
                         2.0 * (b - a))
        return dens * f
    if profile == 'exp':
        n0 = QE * sigma_per_m / w_e
        z = 1.0 / w_e + 1j * k
        f = 2.0 * np.real(np.exp(1j * k * W / 2.0) * (1.0 - np.exp(-W * z)) / z)
        return n0 * f
    raise ValueError(profile)


def _trapz(y, x, axis=-1):
    f = getattr(np, 'trapezoid', None) or np.trapz
    return f(y, x, axis=axis)


def potential(x, W, w_e, sigma_per_m, t, eps_ox, eps_top=1.0, r0=0.0,
              c_it=0.0, nk=20000, kmax_factor=60.0, profile='strip'):
    """Electrostatic potential (V) at positions x (m, origin at the ribbon
    centre) produced by the edge strips, with the gate grounded.

    c_it (F/m^2) is an optional interface-trap capacitance to ground, the
    local term that sets the subthreshold ideality n = 1 + c_it / C_ox."""
    x = np.atleast_1d(np.asarray(x, float))
    kmax = kmax_factor / min(w_e, t)
    if profile == 'exp':
        kmax = max(kmax, kmax_factor / t)
    k = np.concatenate([[0.0], np.geomspace(1e-3 / W, kmax, nk)])
    kk = np.where(k > 0, k, 1e-30)
    denom = EPS0 * kk * (eps_top + eps_ox / np.tanh(kk * t) + 2.0 * r0 * kk) + c_it
    g = _rho_hat(k, W, w_e, sigma_per_m, profile) / denom
    # phi(x) = (1/pi) int_0^inf g(k) cos(kx) dk  (even profile)
    out = np.empty_like(x)
    for i0 in range(0, x.size, 64):
        xs = x[i0:i0 + 64]
        out[i0:i0 + 64] = _trapz(g[None, :] * np.cos(np.outer(xs, k)), k) / np.pi
    return out


def _xgrid(W, w_e, t, n_fine=160, n_coarse=200):
    """Half-ribbon grid (0 .. W/2), dense within a few t of the edge."""
    half = W / 2.0
    edge_zone = min(half, 6.0 * t + 2.0 * w_e)
    fine = half - edge_zone * (np.linspace(0.0, 1.0, n_fine) ** 2)
    coarse = np.linspace(0.0, half - edge_zone, n_coarse)
    return np.unique(np.concatenate([coarse, fine]))


def threshold_shift(W, w_e, sigma_per_m, t, eps_ox, eps_top=1.0, r0=0.0,
                    ideality=1.0, kT=KT300, nk=20000, profile='strip',
                    c_mobile=0.0):
    """Constant-current threshold shift (V) of an n-type ribbon whose edges
    carry sigma electrons per metre each (positive sigma = net negative).

    Subthreshold, nondegenerate channel: the current per width follows the
    width average of exp(-q phi/kT); an ideality n > 1 (interface traps)
    adds a local capacitance (n - 1) C_ox that screens phi and multiplies the
    gate-referred shift by n."""
    cox = EPS0 * eps_ox / t
    c_it = (ideality - 1.0) * cox
    x = _xgrid(W, w_e, t)
    phi = -potential(x, W, w_e, sigma_per_m, t, eps_ox, eps_top, r0,
                     c_it + c_mobile, nk=nk, profile=profile)
    m = np.max(phi)
    avg = _trapz(np.exp((phi - m) / kT), x) / (W / 2.0)
    return -ideality * (m + kT * np.log(avg))


def parallel_plate(W, sigma_per_m, t, eps_ox):
    """Eq. (1) of the article: 2 q sigma / (C_ox W)."""
    cox = EPS0 * eps_ox / t
    return 2.0 * QE * sigma_per_m / (cox * W)


def verify():
    """Analytic limits of the solver."""
    t, eps = 5.5e-9, 19.5
    out = {}
    # 1. a ribbon so wide and strips so wide that the charge is a uniform sheet
    W = 4e-6
    sig = 1e9                           # per metre
    num = threshold_shift(W, W / 2.0, sig, t, eps)
    out['uniform_sheet_rel_err'] = float(abs(num / parallel_plate(W, sig, t, eps) - 1))
    # 2. wide ribbon with edge charge: in subthreshold the shift stays far
    #    below the parallel-plate value (current through the interior)
    W = 2e-6
    num = threshold_shift(W, 5e-9, sig, t, eps)
    out['wide_ribbon_ratio_to_eq1'] = float(num / parallel_plate(W, sig, t, eps))
    return out


# ---------------------------------------------------------------------------
# Self-consistent sheet: fixed edge charge, mobile electrons and the gate
# ---------------------------------------------------------------------------
# The linear response above ignores the electrons themselves.  At the current
# that defines V_T,cc the channel is in weak to moderate inversion, so the
# electrons screen the edge charge partly; on the operating point they screen
# it strongly.  The solver below finds the sheet potential phi(x) for a given
# gate voltage with the electron density in the two lowest conduction valleys
# (K and Q, Fermi-Dirac statistics) and the fixed edge profile, by Newton's
# method on a non-uniform grid over half the ribbon, with the exact layered
# Green's function above.  The electron density is represented by
# piecewise-linear (hat) functions whose transforms are analytic, so the
# coarse centre grid does not alias.

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31


def dos_valleys(mcK=0.51, mcQ=0.76, EQK=0.081):
    """(degeneracy x DOS per valley set, offset in eV) for spin-degenerate
    K (2 valleys) and Q (6 valleys) pockets; DOS in electrons/(m^2 eV)."""
    def d(m, g):
        return 2.0 * g * m * M0 / (2.0 * np.pi * HBAR ** 2) * QE
    return ((d(mcK, 2), 0.0), (d(mcQ, 6), EQK))


def electron_density(phi, valleys, kT=KT300):
    """Sheet electron density (m^-2) and its derivative with respect to phi
    (V); phi is the electron potential energy measured downward, i.e. a
    higher phi fills the band, with the band edge at phi = 0."""
    n = np.zeros_like(phi)
    dn = np.zeros_like(phi)
    for D, off in valleys:
        eta = (phi - off) / kT
        n += D * kT * np.logaddexp(0.0, eta)
        dn += D * 0.5 * (1.0 + np.tanh(eta / 2.0))
    return n, dn


def _hat_transforms(x, k):
    """Cosine transforms of the hat functions on the half grid x (0 .. W/2),
    each mirrored to the other half, so that a symmetric profile
    rho(x) = sum_j rho_j h_j(|x|) has transform sum_j rho_j H_j(k)."""
    n = x.size
    H = np.zeros((n, k.size))
    kk = np.where(k > 0, k, 1e-30)

    def seg_rise(x0, x1):
        # int_{x0}^{x1} (x - x0)/(x1 - x0) cos(kx) dx
        dx = x1 - x0
        out = np.sin(kk * x1) / kk + (np.cos(kk * x1) - np.cos(kk * x0)) / (kk ** 2 * dx)
        return np.where(k > 0, out, dx / 2.0)

    def seg_fall(x0, x1):
        # int_{x0}^{x1} (x1 - x)/(x1 - x0) cos(kx) dx
        dx = x1 - x0
        out = -np.sin(kk * x0) / kk - (np.cos(kk * x1) - np.cos(kk * x0)) / (kk ** 2 * dx)
        return np.where(k > 0, out, dx / 2.0)

    for j in range(n):
        h = np.zeros(k.size)
        if j > 0:
            h += seg_rise(x[j - 1], x[j])
        if j < n - 1:
            h += seg_fall(x[j], x[j + 1])
        H[j] = 2.0 * h      # mirror image on the other half
    return H


class Sheet:
    """Gated ribbon of width W on the stack (t, eps_ox, eps_top, r0)."""

    def __init__(self, W, t, eps_ox, eps_top=1.0, r0=0.0, c_it=0.0,
                 valleys=None, nk=12000, kmax_factor=60.0, h_edge=0.1e-9,
                 h_max=1.5e-9, growth=1.08):
        self.W, self.t, self.eps_ox, self.eps_top, self.r0 = W, t, eps_ox, eps_top, r0
        self.c_it = c_it
        self.cox = EPS0 * eps_ox / t
        self.valleys = valleys or dos_valleys()
        half = W / 2.0
        # grid: h_edge at the edge growing geometrically to h_max inward
        d = [0.0]
        step = h_edge
        while d[-1] < half:
            d.append(d[-1] + step)
            step = min(step * growth, h_max)
        d = np.array(d)
        d = d[d < half]
        x = np.unique(np.concatenate([half - d, [0.0]]))
        self.x = x
        # the transform must resolve both the oxide thickness and the finest
        # element at the edge, or the nodal Green's matrix becomes singular
        kmax = max(kmax_factor / t, 3.0 / h_edge)
        k = np.concatenate([[0.0], np.geomspace(1e-3 / max(W, 50 * t), kmax, nk)])
        self.k = k
        kk = np.where(k > 0, k, 1e-30)
        ghat = 1.0 / (EPS0 * kk * (eps_top + eps_ox / np.tanh(kk * t) + 2.0 * r0 * kk))
        ghat[0] = t / (EPS0 * eps_ox) * 1.0 / (1.0 + 0.0)   # k -> 0 limit
        # trapezoid weights in k
        wk = np.zeros_like(k)
        dk = np.diff(k)
        wk[:-1] += dk / 2.0
        wk[1:] += dk / 2.0
        H = _hat_transforms(x, k)                      # (n, nk)
        C = np.cos(np.outer(x, k))                     # (n, nk)
        self.G = (C * (wk * ghat)) @ H.T / np.pi       # V per (C/m^2) nodal
        # width-average weights of a piecewise-linear profile on the half grid
        wx = np.zeros_like(x)
        dx = np.diff(x)
        wx[:-1] += dx / 2.0
        wx[1:] += dx / 2.0
        self.wx = wx / half

    def fixed_potential(self, w_e, sigma_per_m, profile='exp'):
        if sigma_per_m == 0.0:
            return np.zeros_like(self.x)
        return potential(self.x, self.W, w_e, sigma_per_m, self.t, self.eps_ox,
                         self.eps_top, self.r0, 0.0, nk=20000, profile=profile)

    def solve(self, Vg, phi_fixed, phi0=None, tol=1e-9, maxit=400):
        """Sheet potential (V) for gate voltage Vg (band edge at zero gate
        voltage and zero charge set at phi = 0)."""
        phi = (np.minimum(Vg + phi_fixed, 0.0) if phi0 is None else phi0.copy())
        I = np.eye(self.x.size)
        for _ in range(maxit):
            n, dn = electron_density(phi, self.valleys)
            rho = -QE * n - self.c_it * phi
            F = phi - Vg - phi_fixed - self.G @ rho
            J = I + self.G * (QE * dn + self.c_it)[None, :]
            step = np.linalg.solve(J, -F)
            m = np.max(np.abs(step))
            if m > 0.5:
                step *= 0.5 / m
            phi += step
            if m < tol:
                break
        return phi

    def mean_density(self, phi):
        n, _ = electron_density(phi, self.valleys)
        return float(self.wx @ n)

    def current_measure(self, phi):
        """Width average of the charge-sheet current integrand
        kT q n + (q n)^2 / (2 C_ox), in C V / m^2."""
        n, _ = electron_density(phi, self.valleys)
        q_n = QE * n
        return float(self.wx @ (KT300 * q_n + q_n ** 2 / (2.0 * self.cox)))

    def gate_for(self, target, phi_fixed, measure='current', Vlo=-3.0, Vhi=6.0,
                 xtol=1e-5):
        """Gate voltage at which the chosen measure reaches `target`
        (Brent's method on log(measure), warm-started Newton inside)."""
        from scipy.optimize import brentq
        f = self.current_measure if measure == 'current' else self.mean_density
        state = {'phi': None, 'v': None}

        def g(v):
            warm = state['phi'] if (state['v'] is not None and abs(v - state['v']) < 0.5) else None
            state['phi'] = self.solve(v, phi_fixed, warm)
            state['v'] = v
            return np.log(max(f(state['phi']), 1e-300)) - np.log(target)

        while g(Vhi) < 0:
            Vhi += 2.0 * (Vhi - Vlo)
        while g(Vlo) > 0:
            Vlo -= 2.0 * (Vhi - Vlo)
        state['phi'], state['v'] = None, None
        v = brentq(g, Vlo, Vhi, xtol=xtol)
        phi = self.solve(v, phi_fixed, state['phi'] if state['v'] is not None and abs(v - state['v']) < 0.5 else None)
        return v, phi


def cc_target(I_per_W, L, mu):
    """Charge-sheet target for a constant-current threshold:
    I/W = (mu/L) [kT q n + (q n)^2/(2 C_ox)]  ->  returns (I/W) L / mu."""
    return I_per_W * L / mu


def line_capacitance(W, t, eps_ox, eps_top=1.0, h_max=None):
    """Geometric capacitance per unit length (F/m) of a perfectly conducting
    strip of width W at the top of the gate dielectric: the sheet potential
    is set to 1 V on every node and the nodal charge solved from the exact
    Green's function.  Its ratio to C_ox W is the fringing factor."""
    hm = h_max if h_max is not None else max(1.5e-9, W / 300.0)
    s = Sheet(W, t, eps_ox, eps_top, 0.0, h_max=hm)
    rho = np.linalg.solve(s.G, np.ones_like(s.x))
    return float(W * (s.wx @ rho))


def fringing_factor(W, t, eps_ox, eps_top=1.0):
    """C_ox W / C_line: the factor by which the parallel-plate Eq. (1)
    overstates the strong-inversion threshold shift of a ribbon."""
    return EPS0 * eps_ox / t * W / line_capacitance(W, t, eps_ox, eps_top)
