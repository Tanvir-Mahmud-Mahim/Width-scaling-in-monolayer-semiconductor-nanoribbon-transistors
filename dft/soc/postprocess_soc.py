"""Reduce the stage E (ABINIT, spin-orbit) raw output to physical constants.

Reads soc.json and writes soc_summary.json holding, for each material and
each setting (sr = scalar relativistic, soc = spin-orbit coupled):

  * the relaxed chalcogen height,
  * the A1' frequency at zero strain, its strain derivative and the mode
    Grueneisen parameter, from the same symmetric second difference and the
    same three-strain linear fit as dft/postprocess.py,
  * the K-point gap and its strain derivative (meV per per cent),
  * the elastic constant C11+C12 from the five-point strain sweep,

plus the spin-orbit deltas of each quantity, which are what the Limitations
section quotes.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AMU = 1.66053906660e-27
EV = 1.602176634e-19
ANG = 1e-10
CM1 = 1.0 / (2.0 * np.pi * 2.99792458e10)

MATS = ['MoS2', 'WSe2']
CHA = {'MoS2': 'S', 'WSe2': 'Se'}
MASS = {'S': 32.06, 'Se': 78.97}
A0 = {'MoS2': 3.184, 'WSe2': 3.322}
U = 0.05
STRAINS = [-0.01, 0.0, 0.01]
STRAINS_EL = [-0.02, -0.01, 0.0, 0.01, 0.02]


def freq_sym(Ep, Em, E0, mred_amu, u_ang=U):
    d2 = (Ep + Em - 2.0 * E0)
    if d2 <= 0:
        return float('nan')
    w2 = d2 * EV / (mred_amu * AMU * (u_ang * ANG) ** 2)
    return float(np.sqrt(w2) * CM1)


def main():
    A = {}
    for m in MATS:
        p = os.path.join(HERE, 'soc_%s.json' % m)
        if os.path.exists(p):
            A.update(json.load(open(p)))
    out = {}
    for mat in MATS:
        mX = MASS[CHA[mat]]
        for s_ in ('sr', 'soc'):
            d = {}
            d['z_relaxed'] = A.get('%s_%s_zopt' % (mat, s_))
            # ---- A1' frozen phonon vs strain ---------------------------
            ss, wA = [], []
            for s in STRAINS:
                kb = '%s_%s_bands_e%+.3f' % (mat, s_, s)
                kp = '%s_%s_A1p_e%+.3f' % (mat, s_, s)
                km = '%s_%s_A1m_e%+.3f' % (mat, s_, s)
                if not all(k in A and A[k].get('E') is not None
                           for k in (kb, kp, km)):
                    continue
                f = freq_sym(A[kp]['E'], A[km]['E'], A[kb]['E'], 2.0 * mX)
                ss.append(s * 100.0)
                wA.append(f)
            if len(ss) == 3 and np.all(np.isfinite(wA)):
                ss_a, wA_a = np.array(ss), np.array(wA)
                d['wA'] = float(np.interp(0.0, ss_a, wA_a))
                dA = float(np.polyfit(ss_a, wA_a, 1)[0])
                d['dwA_deps'] = dA
                d['gamma_A'] = float(-dA * 100.0 / (2.0 * d['wA']))
            # ---- K gap vs strain ---------------------------------------
            sp, gap = [], []
            for s in STRAINS:
                kb = '%s_%s_bands_e%+.3f' % (mat, s_, s)
                b = A.get(kb, {}).get('bands')
                if not b:
                    continue
                sp.append(s * 100.0)
                gap.append(b['K']['cb'] - b['K']['vb'])
            if len(sp) == 3:
                sp_a, gap_a = np.array(sp), np.array(gap)
                d['gap_K'] = float(np.interp(0.0, sp_a, gap_a))
                d['dgap_deps'] = float(np.polyfit(sp_a, gap_a, 1)[0]
                                       * 1000.0)          # meV per %
            # ---- elastic constant from the five-point sweep ------------
            st, E = [], []
            for s in STRAINS_EL:
                k = ('%s_%s_bands_e%+.3f' % (mat, s_, s) if s in STRAINS
                     else '%s_%s_e%+.3f' % (mat, s_, s))
                if k in A and A[k].get('E') is not None:
                    st.append(s)
                    E.append(A[k]['E'])
            if len(st) == 5:
                c = np.polyfit(st, E, 2)
                a = A0[mat] * 1e-10
                area = np.sqrt(3) / 2.0 * a ** 2
                d['C11_plus_C12'] = float(2.0 * c[0] * EV / (2.0 * area))
            out.setdefault(mat, {})[s_] = d
        # ---- spin-orbit deltas ----------------------------------------
        if 'sr' in out.get(mat, {}) and 'soc' in out.get(mat, {}):
            sr, so = out[mat]['sr'], out[mat]['soc']
            delta = {}
            for k in ('wA', 'dwA_deps', 'gamma_A', 'gap_K', 'dgap_deps',
                      'C11_plus_C12'):
                if k in sr and k in so:
                    delta[k] = dict(sr=sr[k], soc=so[k],
                                    abs=float(so[k] - sr[k]),
                                    rel=float((so[k] - sr[k]) / sr[k])
                                    if sr[k] else None)
            out[mat]['delta'] = delta
    with open(os.path.join(HERE, 'soc_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
