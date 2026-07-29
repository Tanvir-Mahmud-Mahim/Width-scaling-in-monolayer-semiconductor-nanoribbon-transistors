"""Self-consistent device solver for the monolayer nanoribbon transistor.

The compact expression used for the wide parameter scans evaluates the drain
current from the source-end sheet density and a velocity-saturation factor.
That is fast enough to sweep a design map, but it fixes the channel charge at
its source value, ignores the quantum capacitance of a two-dimensional
channel, and cannot represent the feedback of a series contact resistance on
the internal bias.  This module removes those three approximations by solving
the same devices self-consistently.

Formulation
-----------
Two unknowns are carried at every point along the channel:

    psi   the surface band bending, that is the shift of the conduction band
          edge with respect to the source Fermi level, and
    V     the electron quasi-Fermi potential.

They satisfy two coupled equations.  The first is the vertical electrostatic
balance between the gate and the channel sheet charge,

    C_ox (V_ov - psi) = q n_s(psi, V) ,                                   (1)

with the sheet density of a two-dimensional parabolic band at 300 K,

    n_s = g_2D k_B T ln[1 + exp(q(psi - V)/k_B T)] ,                      (2)
    g_2D = g_s g_v m* / (2 pi hbar^2) .

Writing the charge this way rather than as C_ox(V_GS - V_T)/q is what puts the
quantum capacitance C_q = q^2 g_2D f(eta) in series with the oxide, and what
makes the solution roll smoothly into the subthreshold exponential instead of
clipping at zero.

The second equation is current continuity for drift and diffusion.  Because V
is the quasi-Fermi potential, drift and diffusion are both carried by its
gradient, so

    d/dx [ q n_s mu (dV/dx) / (1 + (mu/v_sat) |dV/dx|) ] = 0 .            (3)

The two equations are assembled on a one-dimensional finite-volume mesh and
solved together by Newton's method in DEVSIM, an open-source device
simulator.  The Jacobian is formed by DEVSIM from the symbolic derivatives of
the models, so no derivative is coded by hand here.

Contacts
--------
The source and drain contacts each carry a resistance R_c per unit width, so
the channel sees 2 R_c in series.  The drop is closed by bisection on the
internal drain bias: the channel current is monotone in that bias, so

    g(v) = v + 2 I(v) R_c - V_DS

has exactly one root, and bisection converges for any contact resistance.  A
fixed-point iteration on the current would diverge as soon as the contact
carries most of the applied bias, which is precisely the regime of interest
for a p-type device.

Everything the ribbon layer contributes is carried through unchanged: the
width-dependent threshold shift produced by the fixed edge charge enters as a
reduction of the overdrive, and the width-averaged mobility of the damaged
halo enters as mu.
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np

from .materials import QE, KB, T300, HBAR, M0, Material
from . import transport

# DEVSIM prints a report on every Newton step; keep it quiet unless asked.
_QUIET = os.environ.get('RAPID_DEVSIM_VERBOSE', '') == ''

_ds = None

DEVICE = 'ribbon'
REGION = 'channel'


def _quiet():
    if not _QUIET:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _devsim():
    """Import DEVSIM on first use, so the rest of the package does not need it."""
    global _ds
    if _ds is None:
        with _quiet():
            import devsim
        _ds = devsim
    return _ds


# ---------------------------------------------------------------------------
# two-dimensional channel electrostatics
# ---------------------------------------------------------------------------
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


def quantum_capacitance(mat: Material, carrier='e'):
    """Degenerate-limit quantum capacitance, F/cm^2."""
    return QE ** 2 * dos_2d(mat, carrier) * 1e-4


def sheet_density(mat, psi_minus_V, carrier='e'):
    """Equation (2), in m^-2.  Used to check the solver independently."""
    kT = KB * T300
    eta = QE * np.asarray(psi_minus_V, float) / kT
    return dos_2d(mat, carrier) * kT * np.log1p(np.exp(np.clip(eta, -700, 700)))


def surface_potential(mat, Vov_gate, V_channel, Cox_cm2, carrier='e'):
    """Solve Eq. (1) at one point by bisection; returns psi in volts.

    Cox_cm2 is in F/cm^2, as everywhere else in this package.  Used to seed the
    Newton solve and, independently, to verify it.
    """
    Cox = float(Cox_cm2) * 1e4                       # F/m^2
    lo, hi = -2.0, float(Vov_gate) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = Cox * (Vov_gate - mid) - QE * sheet_density(mat, mid - V_channel,
                                                        carrier)
        if f > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# the DEVSIM problem
# ---------------------------------------------------------------------------
def _build(mat, L_nm, nx, Cox_cm2, Vov, mu_cm2, vsat_cm_s, carrier):
    """Create the mesh, the models and the two coupled equations."""
    ds = _devsim()
    for d in ds.get_device_list():
        ds.delete_device(device=d)
    for mname in ds.get_mesh_list():
        ds.delete_mesh(mesh=mname)

    L = L_nm * 1e-7                                  # cm, DEVSIM length unit
    ds.create_1d_mesh(mesh='m')
    ds.add_1d_mesh_line(mesh='m', pos=0.0, ps=L / nx, tag='source')
    ds.add_1d_mesh_line(mesh='m', pos=L, ps=L / nx, tag='drain')
    ds.add_1d_region(mesh='m', material='TMD', region=REGION, tag1='source',
                     tag2='drain')
    ds.add_1d_contact(mesh='m', name='source', tag='source', material='metal')
    ds.add_1d_contact(mesh='m', name='drain', tag='drain', material='metal')
    ds.finalize_mesh(mesh='m')
    ds.create_device(mesh='m', device=DEVICE)

    kT = KB * T300 / QE                              # thermal voltage, V
    # DEVSIM works in cm, so the density of states is converted to states per
    # eV per cm^2 and the charge to C/cm^2.
    g2d = dos_2d(mat, carrier) * QE * 1e-4
    for k, v in dict(kT=kT, q=QE, Cox=float(Cox_cm2), Vov=float(Vov),
                     g2d=g2d, mu=float(mu_cm2),
                     vsat=float(vsat_cm_s)).items():
        ds.set_parameter(device=DEVICE, region=REGION, name=k, value=v)

    ds.node_solution(device=DEVICE, region=REGION, name='Psi')
    ds.node_solution(device=DEVICE, region=REGION, name='Vqf')
    ds.edge_from_node_model(device=DEVICE, region=REGION, node_model='Psi')
    ds.edge_from_node_model(device=DEVICE, region=REGION, node_model='Vqf')

    # --- sheet density, Eq. (2) ----------------------------------------
    # ln(1+e^x) and its derivative are written in the overflow-safe form
    #   softplus(x) = max(x,0) + ln(1 + e^-|x|),  sigma(x) = e^min(x,0)/(1+e^-|x|)
    # so that a wide Newton excursion cannot evaluate exp() out of range.
    ds.node_model(device=DEVICE, region=REGION, name='eta',
                  equation='(Psi - Vqf)/kT')
    ds.node_model(device=DEVICE, region=REGION, name='aeta',
                  equation='pow(eta*eta + 1e-30, 0.5)')
    ds.node_model(device=DEVICE, region=REGION, name='eneg',
                  equation='exp(-aeta)')
    ds.node_model(device=DEVICE, region=REGION, name='emin',
                  equation='exp(0.5*(eta - aeta))')
    ds.node_model(device=DEVICE, region=REGION, name='softplus',
                  equation='0.5*(eta + aeta) + log(1 + eneg)')
    ds.node_model(device=DEVICE, region=REGION, name='sigm',
                  equation='emin/(1 + eneg)')
    ds.node_model(device=DEVICE, region=REGION, name='ns',
                  equation='g2d*kT*softplus')
    ds.node_model(device=DEVICE, region=REGION, name='ns:Psi',
                  equation='g2d*sigm')
    ds.node_model(device=DEVICE, region=REGION, name='ns:Vqf',
                  equation='-g2d*sigm')

    # --- Eq. (1): gate charge balance, a pure node equation -------------
    ds.node_model(device=DEVICE, region=REGION, name='Gate',
                  equation='Cox*(Vov - Psi) - q*ns')
    ds.node_model(device=DEVICE, region=REGION, name='Gate:Psi',
                  equation='-Cox - q*ns:Psi')
    ds.node_model(device=DEVICE, region=REGION, name='Gate:Vqf',
                  equation='-q*ns:Vqf')
    ds.equation(device=DEVICE, region=REGION, name='PoissonEq',
                variable_name='Psi', node_model='Gate')

    # --- Eq. (3): drift-diffusion current on each edge ------------------
    ds.edge_average_model(device=DEVICE, region=REGION, node_model='ns',
                          edge_model='ns_edge')
    for u in ('Psi', 'Vqf'):
        ds.edge_average_model(device=DEVICE, region=REGION, node_model='ns',
                              edge_model='ns_edge', derivative=u)
    ds.edge_model(device=DEVICE, region=REGION, name='dV',
                  equation='(Vqf@n0 - Vqf@n1)*EdgeInverseLength')
    ds.edge_model(device=DEVICE, region=REGION, name='dV:Vqf@n0',
                  equation='EdgeInverseLength')
    ds.edge_model(device=DEVICE, region=REGION, name='dV:Vqf@n1',
                  equation='-EdgeInverseLength')
    # velocity saturation, with a smoothed absolute value
    ds.edge_model(device=DEVICE, region=REGION, name='sat',
                  equation='1 + (mu/vsat)*pow(dV*dV + 1e-12, 0.5)')
    ds.edge_model(device=DEVICE, region=REGION, name='sat:Vqf@n0',
                  equation='(mu/vsat)*dV*dV:Vqf@n0*pow(dV*dV + 1e-12, -0.5)')
    ds.edge_model(device=DEVICE, region=REGION, name='sat:Vqf@n1',
                  equation='(mu/vsat)*dV*dV:Vqf@n1*pow(dV*dV + 1e-12, -0.5)')
    ds.edge_model(device=DEVICE, region=REGION, name='J',
                  equation='q*ns_edge*mu*dV/sat')
    for u in ('Psi@n0', 'Psi@n1', 'Vqf@n0', 'Vqf@n1'):
        ds.edge_model(device=DEVICE, region=REGION, name='J:%s' % u,
                      equation='diff(q*ns_edge*mu*dV/sat, %s)' % u)
    ds.equation(device=DEVICE, region=REGION, name='ContinuityEq',
                variable_name='Vqf', edge_model='J')

    # --- contacts --------------------------------------------------------
    ds.set_parameter(device=DEVICE, name='Vsrc', value=0.0)
    ds.set_parameter(device=DEVICE, name='Vdrn', value=0.0)
    ds.contact_node_model(device=DEVICE, contact='source', name='src_V',
                          equation='Vqf - Vsrc')
    ds.contact_node_model(device=DEVICE, contact='source', name='src_V:Vqf',
                          equation='1')
    ds.contact_node_model(device=DEVICE, contact='drain', name='drn_V',
                          equation='Vqf - Vdrn')
    ds.contact_node_model(device=DEVICE, contact='drain', name='drn_V:Vqf',
                          equation='1')
    # edge_current_model lets DEVSIM integrate J over the contact, so that
    # get_contact_current returns the terminal current directly.
    ds.contact_equation(device=DEVICE, contact='source', name='ContinuityEq',
                        node_model='src_V', edge_current_model='J')
    ds.contact_equation(device=DEVICE, contact='drain', name='ContinuityEq',
                        node_model='drn_V', edge_current_model='J')
    # Psi stays free at the contacts: it still satisfies the gate balance.
    ds.contact_equation(device=DEVICE, contact='source', name='PoissonEq',
                        node_model='')
    ds.contact_equation(device=DEVICE, contact='drain', name='PoissonEq',
                        node_model='')


def _seed(mat, Vov, Vd, Cox_cm2, carrier):
    """Initial guess: a linear quasi-Fermi drop and the exact vertical solve."""
    ds = _devsim()
    x = np.array(ds.get_node_model_values(device=DEVICE, region=REGION,
                                          name='x'))
    span = max(x.max() - x.min(), 1e-30)
    V = Vd * (x - x.min()) / span
    psi = np.array([surface_potential(mat, Vov, v, Cox_cm2, carrier)
                    for v in V])
    ds.set_node_values(device=DEVICE, region=REGION, name='Vqf',
                       values=V.tolist())
    ds.set_node_values(device=DEVICE, region=REGION, name='Psi',
                       values=psi.tolist())


# ---------------------------------------------------------------------------
def solve_iv(mat: Material, W_nm, nd_bulk, Vov=1.5, Vds=1.0, Lch_nm=300.0,
             Cox=None, n_it_cm2=None, eps_env=None, sigma_line_cm=0.0,
             halo_nm=5.0, carrier='e', Rc_ohm_um=0.0, nx=48, strain=0.0,
             nd_edge=None, max_bisect=40, tol=1e-6):
    """Self-consistent drain current of one ribbon, in uA/um.

    Returns the terminal current, the internal bias left after the contact
    drop, and the converged profiles along the channel.
    """
    ds = _devsim()
    Cox = transport.COX['HfO2_EOT1p5'] if Cox is None else Cox
    if n_it_cm2 is None:
        n_it_cm2 = transport.N_IT_HFO2
    W = float(W_nm)

    # the ribbon layer supplies the two width-dependent inputs
    dVT = float(transport.threshold_shift(sigma_line_cm, W, Cox))
    Vov_eff = Vov - dVT
    n_ref = Cox * max(Vov_eff, 1e-3) / QE
    mu = float(transport.ribbon_mobility(mat, W, nd_bulk, nd_edge, halo_nm,
                                         strain, n_ref, n_it_cm2, carrier,
                                         eps_env))
    vsat = transport.VSAT.get(mat.name, transport.V_SAT)

    if Vov_eff <= 0.0:
        return dict(I_uA_um=0.0, mu=mu, dVT=dVT, Vov_eff=Vov_eff,
                    Vds_internal=0.0, iterations=0, converged=True,
                    x_nm=[], Psi=[], Vqf=[], ns=[])

    _build(mat, Lch_nm, nx, Cox, Vov_eff, mu, vsat, carrier)
    # R_c is quoted per contact, as is conventional; the channel sees 2 R_c.
    Rc = 2.0 * Rc_ohm_um * 1e-4                      # ohm cm of width

    def channel(vd):
        """Terminal current in A/cm for an internal drain bias vd."""
        ds.set_parameter(device=DEVICE, name='Vsrc', value=0.0)
        ds.set_parameter(device=DEVICE, name='Vdrn', value=float(vd))
        _seed(mat, Vov_eff, vd, Cox, carrier)
        with _quiet():
            ds.solve(type='dc', absolute_error=1e-12, relative_error=1e-9,
                     maximum_iterations=60)
        return abs(float(ds.get_contact_current(device=DEVICE,
                                                contact='drain',
                                                equation='ContinuityEq')))

    ok, it, vd, I = True, 0, float(Vds), 0.0
    try:
        if Rc <= 0.0:
            I = channel(Vds)
        else:
            lo, hi = 0.0, float(Vds)
            for it in range(max_bisect):
                vd = 0.5 * (lo + hi)
                I = channel(vd)
                g = vd + I * Rc - Vds
                if abs(g) <= tol * Vds:
                    break
                if g > 0:
                    hi = vd
                else:
                    lo = vd
    except Exception:
        I, ok = 0.0, False

    x = np.array(ds.get_node_model_values(device=DEVICE, region=REGION,
                                          name='x'))
    return dict(I_uA_um=float(I * 1e2),              # A/cm -> uA/um
                mu=mu, dVT=dVT, Vov_eff=Vov_eff,
                Vds_internal=float(vd), iterations=it + 1, converged=ok,
                x_nm=(x * 1e7).tolist(),
                Psi=list(ds.get_node_model_values(device=DEVICE,
                                                  region=REGION, name='Psi')),
                Vqf=list(ds.get_node_model_values(device=DEVICE,
                                                  region=REGION, name='Vqf')),
                ns=list(ds.get_node_model_values(device=DEVICE,
                                                 region=REGION, name='ns')))


# ---------------------------------------------------------------------------
def verify(mat: Material, Cox=None, Vov=1.5):
    """Four checks that run with the code.

    1. the surface potential returned by the coupled Newton solve matches an
       independent bisection solution of Eq. (1) at every node;
    2. the residual of Eq. (1) itself is at round-off relative to the gate
       charge C_ox V_ov;
    3. current continuity holds, that is the current entering the source
       equals the current leaving the drain;
    4. in the long-channel, low-field limit the solver reproduces the textbook
       square law once the quantum capacitance of a two-dimensional channel is
       placed in series with the oxide.  The uncorrected square law is high by
       C_ox/(C_ox + C_q), which is the size of the effect this solver was
       written to capture.
    """
    Cox = transport.COX['HfO2_EOT1p5'] if Cox is None else Cox
    out = {}

    r = solve_iv(mat, 1000.0, 1.3e12, Vov=Vov, Vds=0.6, Lch_nm=2000.0,
                 Cox=Cox, nx=64)
    psi = np.array(r['Psi'])
    V = np.array(r['Vqf'])
    ns = np.array(r['ns'])
    # independent solution of Eq. (1) by bisection, using only the converged
    # quasi-Fermi potential as input
    psi_ref = np.array([surface_potential(mat, r['Vov_eff'], v, Cox)
                        for v in V])
    out['surface_potential_rel_err'] = float(
        np.max(np.abs(psi - psi_ref)) / max(np.max(np.abs(psi_ref)), 1e-30))
    # residual of the gate balance itself, relative to the gate charge
    resid = Cox * (r['Vov_eff'] - psi) - QE * ns
    out['electrostatics_rel_err'] = float(
        np.max(np.abs(resid)) / (Cox * max(r['Vov_eff'], 1e-30)))

    ds = _devsim()
    Is = abs(float(ds.get_contact_current(device=DEVICE, contact='source',
                                          equation='ContinuityEq')))
    Id = abs(float(ds.get_contact_current(device=DEVICE, contact='drain',
                                          equation='ContinuityEq')))
    out['continuity_rel_err'] = float(abs(Is - Id) / max(Id, 1e-30))

    Vds, L = 0.2, 4000.0
    rr = solve_iv(mat, 1000.0, 1.3e12, Vov=Vov, Vds=Vds, Lch_nm=L, Cox=Cox,
                  nx=64)
    mu = rr['mu']
    Cq = quantum_capacitance(mat)
    Cser = Cox * Cq / (Cox + Cq)
    I_sq = (mu * Cox * (Vov * Vds - 0.5 * Vds ** 2) / (L * 1e-7)) * 1e2
    I_qc = (mu * Cser * (Vov * Vds - 0.5 * Vds ** 2) / (L * 1e-7)) * 1e2
    out['Cq_F_cm2'] = float(Cq)
    out['quantum_capacitance_correction'] = float(Cox / (Cox + Cq))
    out['square_law_uncorrected_uA_um'] = float(I_sq)
    out['square_law_with_Cq_uA_um'] = float(I_qc)
    out['square_law_solver_uA_um'] = float(rr['I_uA_um'])
    out['square_law_rel_err'] = float(abs(rr['I_uA_um'] - I_qc)
                                      / max(I_qc, 1e-30))
    return out
