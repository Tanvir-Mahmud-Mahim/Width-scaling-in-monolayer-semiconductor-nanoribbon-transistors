"""Stage C: chalcogen-vacancy supercells.

A 2x2 supercell of 1H-MX2 with one chalcogen removed gives the defect-induced
shift of the conduction and valence band edges relative to the pristine 2x2
cell.  In the linear (virtual-crystal) regime that shift is

    Delta E = U * n_d ,      n_d = 1 / A_supercell

so the areal integral of the defect potential is  U = Delta E * A_supercell,
which is the short-range scattering strength entering the 2D Fermi golden
rule.  For MoS2 a 3x3 cell is also run to verify linearity in n_d.
"""
import json
import os
import time

import numpy as np

from tmd_common import build_cell, run_scf, Z0

HA = 27.211386245988
KE = 60.0
RES = 'stageC.json'
res = json.load(open(RES)) if os.path.exists(RES) else {}
zopt = {}
if os.path.exists('stageA.json'):
    sa = json.load(open('stageA.json'))
    zopt = {k.replace('_zopt', ''): v for k, v in sa.items() if k.endswith('_zopt')}


def save():
    json.dump(res, open(RES, 'w'), indent=1)


def run(tag, nk, **kw):
    if tag in res and res[tag].get('conv'):
        return res[tag]
    t0 = time.time()
    cell = build_cell(ke_cutoff=KE, verbose=2, **kw)
    mf, e = run_scf(cell, nk=nk, max_cycle=100)
    mo = np.hstack([np.asarray(x) for x in mf.mo_energy])
    occ = np.hstack([np.asarray(x) for x in mf.mo_occ])
    lat = np.asarray(cell.a, dtype=float)
    area = float(abs(lat[0][0] * lat[1][1] - lat[0][1] * lat[1][0])) / 100.0
    rec = dict(E=float(e * HA),
               homo=float(mo[occ > 0].max() * HA),
               lumo=float(mo[occ == 0].min() * HA),
               conv=bool(mf.converged), natom=int(cell.natm),
               nelec=int(cell.nelectron), area_nm2=area,
               t=time.time() - t0)
    res[tag] = rec
    save()
    print(tag, json.dumps(rec), flush=True)
    return rec


MATS = ['MoS2', 'WS2', 'MoSe2', 'WSe2']
for m in MATS:
    z = zopt.get(m, Z0[m])
    run('%s_sc22_pristine' % m, 3, mat=m, z=z, sc=(2, 2))
    run('%s_sc22_vac' % m, 3, mat=m, z=z, sc=(2, 2), remove=[1])

# linearity check for MoS2 on a 3x3 cell
run('MoS2_sc33_pristine', 2, mat='MoS2', z=zopt.get('MoS2', Z0['MoS2']), sc=(3, 3))
run('MoS2_sc33_vac', 2, mat='MoS2', z=zopt.get('MoS2', Z0['MoS2']), sc=(3, 3),
    remove=[1])

print('STAGE C DONE', flush=True)
