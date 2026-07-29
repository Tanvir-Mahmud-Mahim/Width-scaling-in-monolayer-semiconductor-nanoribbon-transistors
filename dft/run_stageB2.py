"""Stage B2: negative frozen-phonon displacements.

Stage B computed E(+u) only.  Because the chalcogen height is not re-relaxed
at finite strain, E(u) can carry a residual linear term, which would alias
directly into the extracted mode Grueneisen parameter.  Adding E(-u) lets the
force constant be taken from the symmetric second difference

    Phi = [E(+u) + E(-u) - 2 E(0)] / u^2 ,

which cancels the linear term exactly and is accurate to O(u^2) in the
anharmonicity.  A finer amplitude is also run for MoS2 to check harmonicity.
"""
import json
import os
import time

from tmd_common import build_cell, run_scf, MASS, MET, CHA, Z0

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


U = 0.05
for m in MATS:
    mM, mX = MASS[MET[m]], MASS[CHA[m]]
    z0 = zrel.get(m, Z0[m])
    for s in [-0.01, 0.0, 0.01]:
        z = z0 * (1 - 0.25 * s)
        scf_point('%s_A1m_e%+.3f' % (m, s), mat=m, strain=s, z=z,
                  disp={1: [0, 0, -U], 2: [0, 0, U]})
        uX = U
        uM = -2 * mX * uX / mM
        scf_point('%s_E1m_e%+.3f' % (m, s), mat=m, strain=s, z=z,
                  disp={0: [-uM, 0, 0], 1: [-uX, 0, 0], 2: [-uX, 0, 0]})

# harmonicity check on MoS2 at zero strain
for uu in [0.025, 0.075]:
    mX = MASS['S']
    mM = MASS['Mo']
    z = zrel.get('MoS2', Z0['MoS2'])
    scf_point('MoS2_A1_u%.3f' % uu, mat='MoS2', strain=0.0, z=z,
              disp={1: [0, 0, uu], 2: [0, 0, -uu]})
    scf_point('MoS2_A1m_u%.3f' % uu, mat='MoS2', strain=0.0, z=z,
              disp={1: [0, 0, -uu], 2: [0, 0, uu]})

print('STAGE B2 DONE', flush=True)
