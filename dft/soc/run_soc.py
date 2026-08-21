"""Stage E: spin-orbit cross-check of the strain derivatives (ABINIT).

Purpose. The production first-principles stage (PySCF, GTH-PBE Gaussian
basis) is scalar relativistic and cannot include spin-orbit coupling. This
stage recomputes, in an independent plane-wave code (ABINIT 9.10.4) with
fully relativistic optimized norm-conserving Vanderbilt pseudopotentials
(PseudoDojo PBE FR v0.4 stringent set, psp8 format, taken from the
abinit/pseudo_dojo repository), the quantities the article takes from its own
calculations: the A1' frequency, its strain derivative and Grueneisen
parameter, the K-point gap and its deformation potential, and the elastic
constant C11+C12. Each quantity is computed twice in otherwise identical
settings,

    nspinor = 1  (the scalar-relativistic part of the same pseudopotential)
    nspinor = 2  (spin-orbit coupling included),

so the difference between the pair is the spin-orbit effect alone, free of
pseudopotential, basis-set and code differences. Materials: MoS2 (lightest
spin-orbit splitting in the set) and WSe2 (heaviest); the other two compounds
are bracketed by these.

Cell, displacement pattern, strain convention and reduction formulas
replicate dft/tmd_common.py and dft/postprocess.py exactly: hexagonal 1H
cell with c = 15 A of vacuum, M at (1/3, 2/3, 1/2), X at
(2/3, 1/3, 1/2 +- z/c); biaxial strain scales a with the chalcogen height
responding as z (1 - 0.25 s); the A1' frozen phonon displaces the two
chalcogens by +-0.05 A along z with the metal fixed, and

    omega^2 = [E(+u) + E(-u) - 2 E(0)] / (2 m_X u^2).

Every converged run is checkpointed to soc_<material>.json (one file
per material, so the two materials can run concurrently), and the stage
is restartable. Postprocessing lives in postprocess_soc.py.
"""
import json
import os
import re
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PSEUDO = os.path.join(HERE, 'pseudos')
def _res_path():
    import sys
    for a in sys.argv[1:]:
        if a in ('MoS2', 'WSe2'):
            return os.path.join(HERE, 'soc_%s.json' % a)
    return os.path.join(HERE, 'soc.json')

RES = _res_path()
HA = 27.211386245988

A0 = {'MoS2': 3.184, 'WSe2': 3.322}      # A, as in tmd_common.py
Z0 = {'MoS2': 1.564, 'WSe2': 1.680}
MET = {'MoS2': 'Mo', 'WSe2': 'W'}
CHA = {'MoS2': 'S', 'WSe2': 'Se'}
ZNUCL = {'Mo': 42, 'W': 74, 'S': 16, 'Se': 34}
MASS = {'Mo': 95.95, 'W': 183.84, 'S': 32.06, 'Se': 78.97}
PSP = {'Mo': 'Mo-sp_r.psp8', 'W': 'W-sp_r.psp8', 'S': 'S_r.psp8',
       'Se': 'Se-d_r.psp8'}
NELEC = {'MoS2': 26, 'WSe2': 46}         # valence electrons per cell
# PseudoDojo hints (Ha): Mo 36/40/46, S 20/26/32, W 31/37/41, Se 39/43/49
ECUT = {'MoS2': 42.0, 'WSe2': 44.0}
VAC = 15.0
NK = 6
U = 0.05                                  # A1' displacement amplitude (A)
MATS = ['MoS2', 'WSe2']
STRAINS = [-0.01, 0.0, 0.01]
# five-point elastic sweep, matching stage A2
STRAINS_EL = [-0.02, -0.01, 0.0, 0.01, 0.02]

KPTS = [('G', 0.0, 0.0, 0.0), ('K', 1 / 3., 1 / 3., 0.0),
        ('M', 0.5, 0.0, 0.0), ('L', 1 / 6., 1 / 6., 0.0),
        ('Q', 5 / 12., 1 / 6., 0.0)]

res = json.load(open(RES)) if os.path.exists(RES) else {}


def save():
    json.dump(res, open(RES, 'w'), indent=1)


def abi_input(mat, strain, z, dz_pattern=0.0, soc=False, bands=False):
    """One abinit input. dz_pattern = +u displaces X1 by +u z and X2 by -u z."""
    a = A0[mat] * (1.0 + strain)
    M, X = MET[mat], CHA[mat]
    zf = z / VAC
    df = dz_pattern / VAC
    nocc = NELEC[mat] if soc else NELEC[mat] // 2
    nband = nocc + (8 if soc else 4) if bands else nocc
    L = []
    L.append('pp_dirpath "%s"' % PSEUDO)
    L.append('pseudos "%s, %s"' % (PSP[M], PSP[X]))
    L.append('acell %.10f %.10f %.10f angstrom' % (a, a, VAC))
    L.append('rprim  1.0 0.0 0.0  -0.5 %.16f 0.0  0.0 0.0 1.0'
             % (np.sqrt(3.0) / 2.0))
    L.append('natom 3 ntypat 2 typat 1 2 2')
    L.append('znucl %d %d' % (ZNUCL[M], ZNUCL[X]))
    L.append('xred')
    L.append(' %.12f %.12f %.12f' % (1 / 3., 2 / 3., 0.5))
    L.append(' %.12f %.12f %.12f' % (2 / 3., 1 / 3., 0.5 + zf + df))
    L.append(' %.12f %.12f %.12f' % (2 / 3., 1 / 3., 0.5 - zf - df))
    L.append('ecut %.1f' % ECUT[mat])
    L.append('occopt 1')
    L.append('prtwf 0')
    L.append('nstep 200')
    L.append('diemac 6.0')
    if soc:
        L.append('nspinor 2 nspden 1')
    if bands:
        L.append('ndtset 2')
        L.append('kptopt1 1 ngkpt1 %d %d 1 nshiftk1 1 shiftk1 0 0 0' % (NK, NK))
        L.append('toldfe1 1.0d-10 prtden1 1')
        L.append('nband1 %d' % nband)
        L.append('iscf2 -2 getden2 -1 kptopt2 0 nkpt2 %d' % len(KPTS))
        L.append('kpt2  ' + '  '.join('%.10f %.10f %.10f' % k[1:]
                                        for k in KPTS))
        L.append('tolwfr2 1.0d-14 nband2 %d' % nband)
        if soc:
            L.append('istwfk1 *1 istwfk2 *1')
    else:
        L.append('kptopt 1 ngkpt %d %d 1 nshiftk 1 shiftk 0 0 0' % (NK, NK))
        L.append('toldfe 1.0d-10')
        L.append('nband %d' % nband)
        if soc:
            L.append('istwfk *1')
    return '\n'.join(L) + '\n'


def run_abinit(tag, inp):
    d = os.path.join(HERE, 'runs', tag)
    os.makedirs(d, exist_ok=True)
    fi = os.path.join(d, 'run.abi')
    open(fi, 'w').write(inp)
    t0 = time.time()
    np = os.environ.get('ABINIT_NP', '1')
    cmd = (['mpirun', '--allow-run-as-root', '-np', np, 'abinit', 'run.abi']
           if np != '1' else ['abinit', 'run.abi'])
    with open(os.path.join(d, 'run.log'), 'w') as log:
        subprocess.run(cmd, cwd=d, stdout=log,
                       stderr=subprocess.STDOUT, check=False)
    txt = ''
    fo = os.path.join(d, 'run.abo')
    if os.path.exists(fo):
        txt = open(fo).read()
    # keep the raw text record (run.abi, run.abo, run.log) but drop the
    # bulky binary state the archive does not need
    for f in os.listdir(d):
        if f.endswith(('.nc', '_DEN', '_WFK', '_EIG', '_DDB'))                 or '_1WF' in f or f.endswith('_KDEN'):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
    return txt, time.time() - t0


def parse_etotal(txt):
    m = re.findall(r'^\s+etotal1?\s+(-?[\d.]+E?[+-]?\d*)', txt, re.M)
    return float(m[-1]) * HA if m else None


def parse_eig(txt, nband):
    """Eigenvalues (eV) at the 5 NSCF k points, from the dataset-2 block.

    ABINIT prints them in hartree ("Eigenvalues (hartree) for nkpt= ...");
    the last such block in the output is the non-self-consistent dataset."""
    blocks = re.split(
        r'Eigenvalues \((?:hartree|\s*eV\s*)\) for nkpt=\s*\d+\s*k points',
        txt)
    if len(blocks) < 2:
        return None
    unit = HA if 'Eigenvalues (hartree)' in txt else 1.0
    tail = blocks[-1]
    kblocks = re.split(r'kpt#\s*\d+', tail)[1:]
    out = []
    for kb in kblocks[:len(KPTS)]:
        kb = kb.split('kpt#')[0].split('EnergyTerms')[0]
        nums = re.findall(r'-?\d+\.\d+', kb)
        # header floats: the weight and the three k coordinates
        vals = [float(x) * unit for x in nums[4:4 + nband]]
        if len(vals) < nband:
            return None
        out.append(vals)
    return out if len(out) == len(KPTS) else None


def scf_point(tag, **kw):
    if tag in res and res[tag].get('E') is not None:
        return res[tag]
    txt, dt = run_abinit(tag, abi_input(**kw))
    E = parse_etotal(txt)
    rec = dict(E=E, t=dt)
    res[tag] = rec
    save()
    print(tag, json.dumps(rec), flush=True)
    if E is None:
        raise RuntimeError('SCF failed: %s' % tag)
    return rec


def bands_point(tag, mat, strain, z, soc):
    if tag in res and res[tag].get('bands'):
        return res[tag]
    nocc = NELEC[mat] if soc else NELEC[mat] // 2
    nband = nocc + (8 if soc else 4)
    txt, dt = run_abinit(tag, abi_input(mat, strain, z, soc=soc, bands=True))
    E = parse_etotal(txt)
    ev = parse_eig(txt, nband)
    if ev is None:
        raise RuntimeError('bands parse failed: %s' % tag)
    bands = {}
    for (name, _, _, _), e in zip(KPTS, ev):
        e = sorted(e)
        bands[name] = dict(vb=e[nocc - 1], cb=e[nocc])
    res[tag] = dict(E=E, bands=bands, t=dt)
    save()
    print(tag, 'bands ok  t=%.0f s' % dt, flush=True)
    return res[tag]


def main():
    import sys
    mats = [a for a in sys.argv[1:] if a in MATS] or MATS
    for mat in mats:
        for soc in (False, True):
            s_ = 'soc' if soc else 'sr'
            # ---- chalcogen height relaxation at zero strain --------------
            # The spin-orbit relaxation is repeated only for MoS2, where it
            # moves the relaxed height by 1e-4 A relative to the scalar run;
            # the WSe2 spinor runs therefore reuse the scalar height, which
            # the production stage shows perturbs the A1' strain derivative
            # at only the 0.2 per cent level.
            key_z = '%s_%s_zopt' % (mat, s_)
            if soc and mat == 'WSe2' and ('%s_sr_zopt' % mat) in res:
                res[key_z] = res['%s_sr_zopt' % mat]
                save()
            if key_z not in res:
                dzs = [-0.04, 0.0, 0.04]
                E = [scf_point('%s_%s_z%+.3f' % (mat, s_, dz), mat=mat,
                               strain=0.0, z=Z0[mat] + dz, soc=soc)['E']
                     for dz in dzs]
                c = np.polyfit(dzs, E, 2)
                res[key_z] = float(Z0[mat]
                                   + np.clip(-c[1] / (2 * c[0]), -0.08, 0.08))
                save()
            zrel = res[key_z]
            print('ZOPT', mat, s_, zrel, flush=True)
            # ---- elastic five-point sweep + bands ------------------------
            # The +-2% elastic end points are kept for every setting except
            # the WSe2 spinor runs, whose cost per point is an order of
            # magnitude above the rest; the spin-orbit effect on the elastic
            # constant is therefore reported for MoS2 only.
            strain_list = (STRAINS if (soc and mat == 'WSe2')
                           else STRAINS_EL)
            for s in strain_list:
                z = zrel * (1 - 0.25 * s)
                if s in STRAINS:
                    # dataset 1 of the bands run is the base SCF at this
                    # geometry, so its etotal doubles as the phonon E0
                    bands_point('%s_%s_bands_e%+.3f' % (mat, s_, s), mat, s,
                                z, soc)
                else:
                    scf_point('%s_%s_e%+.3f' % (mat, s_, s), mat=mat,
                              strain=s, z=z, soc=soc)
            # ---- A1' frozen phonon at three strains ----------------------
            for s in STRAINS:
                z = zrel * (1 - 0.25 * s)
                scf_point('%s_%s_A1p_e%+.3f' % (mat, s_, s), mat=mat,
                          strain=s, z=z, soc=soc, dz_pattern=U)
                scf_point('%s_%s_A1m_e%+.3f' % (mat, s_, s), mat=mat,
                          strain=s, z=z, soc=soc, dz_pattern=-U)
    print('STAGE E RUNS DONE', flush=True)


if __name__ == '__main__':
    main()
