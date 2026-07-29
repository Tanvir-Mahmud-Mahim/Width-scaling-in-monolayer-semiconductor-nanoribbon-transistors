"""Check that the quantities used in the paper are insensitive to k-sampling.

The absolute total energy of monolayer MoS2 is not converged at
4 x 4 x 1 (see conv.json), but every quantity the paper uses is a difference
taken at fixed sampling.  This script repeats the MoS2 strain sweep and the
A1' frozen-phonon calculation on a denser 6 x 6 x 1 mesh so that
gamma_LA and gamma_A1' can be compared directly with the production values.

Output: kcheck.json
"""
import json
import os
import time

import numpy as np

from tmd_common import build_cell, run_scf, A0, Z0, MASS

HA = 27.211386245988
NK = 6
KE = 60.0
RES = 'kcheck.json'
res = json.load(open(RES)) if os.path.exists(RES) else {}

M = 'MoS2'
U = 0.05                     # frozen-phonon displacement, Angstrom
NU = 0.25                    # imposed out-of-plane Poisson scaling
# relaxed chalcogen half-thickness from the production run, so the geometries
# are identical to the 4 x 4 x 1 ones and only the sampling changes
ZREL = json.load(open('stageA.json'))['%s_zopt' % M]


def save():
    json.dump(res, open(RES, 'w'), indent=1)


def point(tag, **kw):
    if tag in res and res[tag].get('conv'):
        return res[tag]
    t0 = time.time()
    cell = build_cell(M, ke_cutoff=KE, verbose=2, **kw)
    mf, e = run_scf(cell, nk=NK)
    rec = dict(E=float(e * HA), conv=bool(mf.converged), t=time.time() - t0)
    res[tag] = rec
    save()
    print(tag, json.dumps(rec), flush=True)
    return rec


# ---- biaxial strain sweep (gives C11+C12 and gamma_LA) --------------------
for s in [-0.02, -0.01, 0.0, 0.01, 0.02]:
    point('e%+.3f' % s, strain=s, z=ZREL * (1 - NU * s))

# ---- A1' frozen phonons at three strains (gives gamma_A1') ---------------
for s in [-0.01, 0.0, 0.01]:
    z = ZREL * (1 - NU * s)
    for sgn, name in ((+1, 'A1p'), (-1, 'A1m')):
        point('%s_e%+.3f' % (name, s), strain=s, z=z,
              disp={1: [0, 0, sgn * U], 2: [0, 0, -sgn * U]})

print('KCHECK DONE', flush=True)
