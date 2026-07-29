"""Reduce the 6 x 6 x 1 cross-check run to gamma_LA and gamma_A1'.

The production numbers use a 4 x 4 x 1 mesh.  This script applies exactly the
same reduction to the denser-mesh energies in kcheck.json so that the two can
be compared directly.  Output: kcheck_summary.json
"""
import json
import os

import numpy as np

from postprocess import freq_sym, local_elastic
from tmd_common import A0, CHA, MASS

HERE = os.path.dirname(os.path.abspath(__file__))
M = 'MoS2'
U = 0.05
STRAINS = [-0.02, -0.01, 0.0, 0.01, 0.02]


def main():
    K = json.load(open(os.path.join(HERE, 'kcheck.json')))
    out = {}

    keys = ['e%+.3f' % s for s in STRAINS]
    if all(k in K for k in keys):
        E = [K[k]['E'] for k in keys]
        C = local_elastic(STRAINS, E, A0[M])
        cm, cp = C[-0.01], C[0.01]
        Am, Ap = (1 - 0.01) ** 2, (1 + 0.01) ** 2
        out['gamma_LA'] = float(-0.5 * (np.log(cp) - np.log(cm))
                                / (np.log(Ap) - np.log(Am)))
        out['C11_plus_C12'] = float(C[0.0])

    mX = MASS[CHA[M]]

    def gamma_A_sym(get):
        """A1' Grueneisen parameter from the symmetric +/- u displacement."""
        w, ss = [], []
        for s in [-0.01, 0.0, 0.01]:
            t = get(s)
            if t is None:
                return None
            Eb, Ep, Em = t
            ss.append(s * 100.0)
            w.append(freq_sym(Ep, Em, Eb, 2.0 * mX, U))
        slope = float(np.polyfit(ss, w, 1)[0])
        w0 = float(np.interp(0.0, ss, w))
        return dict(wA=w0, dwA_deps=slope,
                    gamma_A=float(-slope / (2.0 * w0) * 100.0))

    def get6(s):
        kb, kp, km = ('e%+.3f' % s, 'A1p_e%+.3f' % s, 'A1m_e%+.3f' % s)
        if not all(k in K for k in (kb, kp, km)):
            return None
        return K[kb]['E'], K[kp]['E'], K[km]['E']

    A = json.load(open(os.path.join(HERE, 'stageA.json')))

    def get4(s):
        kb = '%s_e%+.3f' % (M, s)
        kp = '%s_A1_e%+.3f' % (M, s)
        km = '%s_A1m_e%+.3f' % (M, s)
        if not all(k in A for k in (kb, kp, km)):
            return None
        return A[kb]['E'], A[kp]['E'], A[km]['E']

    r6, r4 = gamma_A_sym(get6), gamma_A_sym(get4)
    if r6:
        out['gamma_A_sym_6x6'] = r6['gamma_A']
        out['wA_sym_6x6'] = r6['wA']
    if r4:
        out['gamma_A_sym_4x4'] = r4['gamma_A']
        out['wA_sym_4x4'] = r4['wA']

    json.dump(out, open(os.path.join(HERE, 'kcheck_summary.json'), 'w'),
              indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
