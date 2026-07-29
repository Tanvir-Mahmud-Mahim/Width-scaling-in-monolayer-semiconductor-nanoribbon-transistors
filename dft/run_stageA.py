"""Stage A+B first-principles runs for monolayer MX2 (M = Mo, W; X = S, Se).

PBE exchange-correlation, GTH pseudopotentials, DZVP-MOLOPT-SR Gaussian basis,
PySCF periodic Gamma-point-extended k-mesh DFT.

Outputs (stageA.json):
  * total energies on a chalcogen-height scan  -> relaxed internal coordinate
  * total energies and band edges on a biaxial strain sweep
        -> deformation potentials, valley separations, elastic constant C11+C12
  * frozen-phonon energies for the A1' and E' Raman-active modes at three strains
        -> Gamma-point frequencies and mode Grueneisen parameters
"""
import json
import os
import time

import numpy as np

from tmd_common import build_cell, run_scf, A0, Z0, MET, CHA, MASS

HA = 27.211386245988  # eV per Hartree
NK = 4
KE = 60.0
RES = 'stageA.json'
res = json.load(open(RES)) if os.path.exists(RES) else {}


def save():
    json.dump(res, open(RES, 'w'), indent=1)


def kpath_energies(mf, cell):
    """Non-self-consistent band energies at high-symmetry points."""
    pts = {'G': [0, 0, 0],
           'K': [1 / 3., 1 / 3., 0],
           'M': [0.5, 0., 0],
           'L': [1 / 6., 1 / 6., 0],          # Lambda, midpoint Gamma-K
           'Q': [5 / 12., 1 / 6., 0]}          # midpoint K-M
    frac = np.array([pts[k] for k in pts])
    kabs = cell.get_abs_kpts(frac)
    e = mf.get_bands(kabs)[0]
    nocc = cell.nelectron // 2
    out = {}
    for i, k in enumerate(pts):
        ei = np.sort(np.asarray(e[i]))
        out[k] = dict(vb=float(ei[nocc - 1] * HA), cb=float(ei[nocc] * HA))
    return out


def scf_point(tag, **kw):
    if tag in res and res[tag].get('conv'):
        return res[tag]
    t0 = time.time()
    cell = build_cell(ke_cutoff=KE, verbose=2, **kw)
    mf, e = run_scf(cell, nk=NK)
    rec = dict(E=float(e * HA), conv=bool(mf.converged), t=time.time() - t0,
               natom=int(cell.natm), mesh=list(map(int, cell.mesh)))
    try:
        rec['bands'] = kpath_energies(mf, cell)
    except Exception as ex:                                    # pragma: no cover
        rec['bands_err'] = str(ex)
    res[tag] = rec
    save()
    print(tag, json.dumps({k: v for k, v in rec.items() if k != 'bands'}), flush=True)
    return rec


MATS = ['MoS2', 'WS2', 'MoSe2', 'WSe2']

# ---------------- Stage A1: relax the chalcogen height at zero strain --------
zrel = {}
for m in MATS:
    dzs = [-0.04, 0.0, 0.04]
    E = [scf_point('%s_z%+.3f' % (m, dz), mat=m, dz=dz)['E'] for dz in dzs]
    c = np.polyfit(dzs, E, 2)
    dz_opt = float(np.clip(-c[1] / (2 * c[0]), -0.08, 0.08))
    zrel[m] = Z0[m] + dz_opt
    res['%s_zopt' % m] = zrel[m]
    save()
    print('ZOPT', m, zrel[m], flush=True)

# ---------------- Stage A2: biaxial strain sweep ----------------------------
STRAINS = [-0.02, -0.01, 0.0, 0.01, 0.02]
for m in MATS:
    for s in STRAINS:
        z = zrel[m] * (1 - 0.25 * s)       # out-of-plane Poisson response
        scf_point('%s_e%+.3f' % (m, s), mat=m, strain=s, z=z)

# ---------------- Stage B: frozen Gamma phonons vs strain -------------------
# A1': both chalcogens move oppositely along z, the metal stays fixed.
# E' : the metal and the chalcogen pair move oppositely in plane, zero net
#      momentum, so m_M u_M = -2 m_X u_X.
U = 0.05  # displacement amplitude, Angstrom
for m in MATS:
    mM, mX = MASS[MET[m]], MASS[CHA[m]]
    for s in [-0.01, 0.0, 0.01]:
        z = zrel[m] * (1 - 0.25 * s)
        base = scf_point('%s_e%+.3f' % (m, s), mat=m, strain=s, z=z)
        rA = scf_point('%s_A1_e%+.3f' % (m, s), mat=m, strain=s, z=z,
                       disp={1: [0, 0, U], 2: [0, 0, -U]})
        uX = U
        uM = -2 * mX * uX / mM
        rE = scf_point('%s_E1_e%+.3f' % (m, s), mat=m, strain=s, z=z,
                       disp={0: [uM, 0, 0], 1: [uX, 0, 0], 2: [uX, 0, 0]})
        res['%s_ph_e%+.3f' % (m, s)] = dict(E0=base['E'], EA=rA['E'], EE=rE['E'],
                                            U=U, uM=uM)
        save()

print('STAGE A+B DONE', flush=True)
