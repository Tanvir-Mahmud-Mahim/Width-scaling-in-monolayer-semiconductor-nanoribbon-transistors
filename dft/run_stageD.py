"""Stage D: chalcogen height relaxed at finite strain.

The A1' normal coordinate is the chalcogen half-thickness z itself, so the
curvature of E(z) gives the mode force constant directly,

    omega_A1'^2 = (d^2E/dz^2) / (2 m_X) ,

and the minimum of the same scan gives the relaxed z at that strain.  Running a
three-point z scan at each of eps = -1, 0, +1 per cent therefore yields the
A1' frequency evaluated at the relaxed geometry, removing the clamped-ion
constraint that biases the frozen-phonon result computed at an imposed z.
"""
import json
import os
import time

import numpy as np

from tmd_common import build_cell, run_scf, Z0

HA = 27.211386245988
NK = 4
KE = 60.0
RES = 'stageA.json'
res = json.load(open(RES))
MATS = ['MoS2', 'WS2', 'MoSe2', 'WSe2']
zrel = {m: res['%s_zopt' % m] for m in MATS if '%s_zopt' % m in res}


def save():
    json.dump(res, open(RES, 'w'), indent=1)


def scf_point(tag, **kw):
    if tag in res and res[tag].get('conv'):
        return res[tag]
    t0 = time.time()
    cell = build_cell(ke_cutoff=KE, verbose=2, **kw)
    mf, e = run_scf(cell, nk=NK)
    rec = dict(E=float(e * HA), conv=bool(mf.converged), t=time.time() - t0)
    res[tag] = rec
    save()
    print(tag, json.dumps(rec), flush=True)
    return rec


DZ = 0.04
for m in MATS:
    z0 = zrel.get(m, Z0[m])
    for s in [-0.01, 0.0, 0.01]:
        zc = z0 * (1 - 0.25 * s)
        for j, dz in enumerate([-DZ, 0.0, DZ]):
            if abs(dz) < 1e-9:
                # already computed as the strain-sweep point
                continue
            scf_point('%s_zscan_e%+.3f_d%+.3f' % (m, s, dz), mat=m, strain=s,
                      z=zc + dz)
        res['%s_zscan_meta_e%+.3f' % (m, s)] = dict(z_centre=zc, dz=DZ)
        save()

print('STAGE D DONE', flush=True)
