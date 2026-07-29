"""Convergence study for the supplementary information.

Monolayer MoS2 total energy, K-point gap and Gamma-phonon frequencies against
the plane-wave cutoff of the density grid, the k-point sampling and the
vacuum thickness.
"""
import json
import os
import time

import numpy as np

import tmd_common
from tmd_common import build_cell, run_scf, MASS

HA = 27.211386245988
RES = 'conv.json'
res = json.load(open(RES)) if os.path.exists(RES) else {}


def save():
    json.dump(res, open(RES, 'w'), indent=1)


def one(tag, ke, nk, vac, phonon=False):
    if tag in res:
        return res[tag]
    tmd_common.VAC = vac
    t0 = time.time()
    cell = build_cell('MoS2', ke_cutoff=ke, verbose=2)
    mf, e = run_scf(cell, nk=nk)
    mo = np.hstack([np.asarray(x) for x in mf.mo_energy])
    occ = np.hstack([np.asarray(x) for x in mf.mo_occ])
    rec = dict(ke=ke, nk=nk, vac=vac, E=float(e * HA),
               gap=float((mo[occ == 0].min() - mo[occ > 0].max()) * HA),
               mesh=list(map(int, cell.mesh)), t=time.time() - t0,
               conv=bool(mf.converged))
    if phonon:
        U = 0.05
        c2 = build_cell('MoS2', ke_cutoff=ke, verbose=2,
                        disp={1: [0, 0, U], 2: [0, 0, -U]})
        _, e2 = run_scf(c2, nk=nk)
        rec['dE_A1'] = float((e2 - e) * HA)
    res[tag] = rec
    save()
    print(tag, json.dumps(rec), flush=True)
    return rec


for ke in [40.0, 60.0, 100.0]:
    one('ke%g' % ke, ke, 4, 15.0, phonon=True)
for nk in [2, 3, 4, 6]:
    one('nk%d' % nk, 60.0, nk, 15.0)
for vac in [12.0, 15.0, 20.0]:
    one('vac%g' % vac, 60.0, 4, vac)
print('CONV DONE', flush=True)
