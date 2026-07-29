"""Reduce the raw PySCF output to the physical constants used by the models.

Reads stageA.json (chalcogen relaxation, biaxial strain sweep, frozen phonons
at both displacement signs) and writes dft_summary.json, which materials.py
loads.

Only quantities that are well defined in a slab calculation without a vacuum
reference are exported: energy derivatives, energy differences between states
of the same calculation, and phonon frequencies.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MASS = {'Mo': 95.95, 'W': 183.84, 'S': 32.06, 'Se': 78.97}
MET = {'MoS2': 'Mo', 'WS2': 'W', 'MoSe2': 'Mo', 'WSe2': 'W'}
CHA = {'MoS2': 'S', 'WS2': 'S', 'MoSe2': 'Se', 'WSe2': 'Se'}
A0 = {'MoS2': 3.184, 'WS2': 3.183, 'MoSe2': 3.319, 'WSe2': 3.322}
MATS = ['MoS2', 'WS2', 'MoSe2', 'WSe2']

AMU = 1.66053906660e-27
EV = 1.602176634e-19
ANG = 1e-10
CM1 = 1.0 / (2.0 * np.pi * 2.99792458e10)      # rad/s -> cm^-1
STRAINS = [-0.02, -0.01, 0.0, 0.01, 0.02]
# measured zone-centre frequencies used as the anchor (see materials.PHONON)
WEXP = {'MoS2': (385.0, 403.0), 'WS2': (356.0, 417.5),
        'MoSe2': (287.0, 241.0), 'WSe2': (249.4, 250.2)}
EA = {'MoS2': 1.880, 'WS2': 2.010, 'MoSe2': 1.570, 'WSe2': 1.650}
WLA_EXP = {'MoS2': 227.6, 'WS2': 176.0, 'WSe2': 130.0}
E_LASER = 1239.84 / 532.0                       # 2.331 eV
U_DEFECT = 0.285                                # eV nm^2, see materials.py


def freq_sym(Ep, Em, E0, mred_amu, u_ang):
    """Harmonic frequency (cm^-1) from the symmetric second difference

        Phi = [E(+u) + E(-u) - 2 E(0)] / u^2 ,  omega^2 = Phi / mred ,

    which cancels any residual linear force along the mode coordinate.
    """
    d2 = (Ep + Em - 2.0 * E0)
    if d2 <= 0:
        return float('nan')
    w2 = d2 * EV / (mred_amu * AMU * (u_ang * ANG) ** 2)
    return float(np.sqrt(w2) * CM1)


def freq_onesided(Ep, E0, mred_amu, u_ang):
    d = Ep - E0
    if d <= 0:
        return float('nan')
    w2 = 2.0 * d * EV / (mred_amu * AMU * (u_ang * ANG) ** 2)
    return float(np.sqrt(w2) * CM1)


def local_elastic(strains, E, a0):
    """Elastic constant C11+C12 (N/m) at each interior strain point, from a
    local three-point quadratic fit and the strained cell area."""
    out = {}
    for i in range(1, len(strains) - 1):
        x = np.array(strains[i - 1:i + 2])
        y = np.array(E[i - 1:i + 2])
        c = np.polyfit(x - x[1], y, 2)
        d2E = 2.0 * c[0] * EV                      # J per unit strain^2
        a = a0 * 1e-10 * (1.0 + strains[i])
        area = np.sqrt(3) / 2.0 * a ** 2
        out[strains[i]] = d2E / (2.0 * area)
    return out


def main():
    A = json.load(open(os.path.join(HERE, 'stageA.json')))
    out = {}

    for m in MATS:
        d = {'a0': A0[m], 'Udef': U_DEFECT}
        mM, mX = MASS[MET[m]], MASS[CHA[m]]

        # ---------------- strain sweep ---------------------------------
        st, E, bands = [], [], []
        for x in STRAINS:
            k = '%s_e%+.3f' % (m, x)
            if k in A and A[k].get('conv'):
                st.append(x)
                E.append(A[k]['E'])
                bands.append(A[k].get('bands'))
        if len(st) >= 3:
            c = np.polyfit(st, E, 2)
            a = A0[m] * 1e-10
            area = np.sqrt(3) / 2.0 * a ** 2
            d['C11_plus_C12'] = float(2.0 * c[0] * EV / (2.0 * area))
            d['C2D'] = float(d['C11_plus_C12'] / 1.25)
        if len(st) == 5:
            C = local_elastic(st, E, A0[m])
            cm, cp = C[-0.01], C[0.01]
            Am, Ap = (1 - 0.01) ** 2, (1 + 0.01) ** 2
            d['gamma_LA'] = float(-0.5 * (np.log(cp) - np.log(cm))
                                  / (np.log(Ap) - np.log(Am)))
        if len(st) >= 3 and all(b is not None for b in bands):
            sp = np.array(st) * 100.0
            gap = np.array([b['K']['cb'] - b['K']['vb'] for b in bands])
            d['gap_K'] = float(np.interp(0.0, sp, gap))
            d['dgap_deps'] = float(np.polyfit(sp, gap, 1)[0] * 1000.0)  # meV/%
            # diagnostics only, not used by the transport model
            eQK = np.array([min(b['L']['cb'], b['Q']['cb']) - b['K']['cb']
                            for b in bands])
            d['EQK_dft_diag'] = float(np.interp(0.0, sp, eQK))
            d['dEQK_deps_diag'] = float(np.polyfit(sp, eQK, 1)[0])

        # ---------------- frozen phonons -------------------------------
        wA, wE, ss = [], [], []
        mred_A = 2.0 * mX
        mred_E = 2.0 * mX * (1.0 + 2.0 * mX / mM)
        for s in [-0.01, 0.0, 0.01]:
            kb = '%s_e%+.3f' % (m, s)
            kAp, kAm = '%s_A1_e%+.3f' % (m, s), '%s_A1m_e%+.3f' % (m, s)
            kEp, kEm = '%s_E1_e%+.3f' % (m, s), '%s_E1m_e%+.3f' % (m, s)
            if not (kb in A and kAp in A and kEp in A):
                continue
            E0 = A[kb]['E']
            U = A.get('%s_ph_e%+.3f' % (m, s), {}).get('U', 0.05)
            if kAm in A and kEm in A:
                fA = freq_sym(A[kAp]['E'], A[kAm]['E'], E0, mred_A, U)
                fE = freq_sym(A[kEp]['E'], A[kEm]['E'], E0, mred_E, U)
                d['phonon_symmetric'] = True
            else:
                fA = freq_onesided(A[kAp]['E'], E0, mred_A, U)
                fE = freq_onesided(A[kEp]['E'], E0, mred_E, U)
                d['phonon_symmetric'] = False
            ss.append(s * 100.0)
            wA.append(fA)
            wE.append(fE)
        if len(ss) >= 3 and np.all(np.isfinite(wA)) and np.all(np.isfinite(wE)):
            ss, wA, wE = np.array(ss), np.array(wA), np.array(wE)
            d['wA_dft'] = float(np.interp(0.0, ss, wA))
            d['wE_dft'] = float(np.interp(0.0, ss, wE))
            dA = float(np.polyfit(ss, wA, 1)[0])
            dE = float(np.polyfit(ss, wE, 1)[0])
            d['dwA_deps'] = dA
            d['dwE_deps'] = dE
            d['gamma_A'] = float(-dA * 100.0 / (2.0 * d['wA_dft']))
            d['gamma_E'] = float(-dE * 100.0 / (2.0 * d['wE_dft']))
        # ---- A1' from the relaxed z scan --------------------------------
        # The A1' normal coordinate is the chalcogen half-thickness, so the
        # curvature of E(z) gives the force constant at the relaxed geometry.
        zs, wz = [], []
        for s in [-0.01, 0.0, 0.01]:
            meta = A.get('%s_zscan_meta_e%+.3f' % (m, s))
            kb = '%s_e%+.3f' % (m, s)
            km = '%s_zscan_e%+.3f_d%+.3f' % (m, s, -0.04)
            kp = '%s_zscan_e%+.3f_d%+.3f' % (m, s, 0.04)
            if not (meta and kb in A and km in A and kp in A):
                continue
            dz = meta['dz']
            c = (A[kp]['E'] + A[km]['E'] - 2.0 * A[kb]['E']) / dz ** 2
            if c <= 0:
                continue
            w2 = c * EV / 1e-20 / (2.0 * mX * AMU)
            zs.append(s * 100.0)
            wz.append(float(np.sqrt(w2) * CM1))
            if abs(s) < 1e-9:
                zmin = -0.5 * (A[kp]['E'] - A[km]['E']) / (dz * c)
                d['z_shift_relaxed'] = float(zmin)
        if len(zs) == 3:
            zs, wz = np.array(zs), np.array(wz)
            d['wA_relaxed'] = float(np.interp(0.0, zs, wz))
            dAz = float(np.polyfit(zs, wz, 1)[0])
            d['dwA_deps_relaxed'] = dAz
            d['gamma_A_relaxed'] = float(-dAz * 100.0 / (2.0 * d['wA_relaxed']))
            d['gamma_A'] = d['gamma_A_relaxed']
            d['wA_dft'] = d['wA_relaxed']

        d['wE_exp_anchor'] = WEXP[m][0]
        d['wA_exp_anchor'] = WEXP[m][1]
        out[m] = d

    # ---- harmonicity check on MoS2 --------------------------------------
    harm = {}
    for uu in [0.025, 0.05, 0.075]:
        if uu == 0.05:
            kp, km = 'MoS2_A1_e+0.000', 'MoS2_A1m_e+0.000'
        else:
            kp, km = 'MoS2_A1_u%.3f' % uu, 'MoS2_A1m_u%.3f' % uu
        if kp in A and km in A and 'MoS2_e+0.000' in A:
            harm[uu] = freq_sym(A[kp]['E'], A[km]['E'], A['MoS2_e+0.000']['E'],
                                2 * MASS['S'], uu)
    if harm:
        out['_harmonicity_MoS2_A1'] = harm

    # ---- transfer of the disorder-activated Raman constant --------------
    # In the double-resonance picture C ~ |D|^2 / (omega_LA^2 Delta^2), with D
    # the gap deformation potential, omega_LA the zone-edge acoustic frequency
    # and Delta the detuning of the laser from the A exciton.  MoS2 is the
    # measured anchor; the other three are predictions of this work.
    C_A_MoS2 = 0.59
    ref = out.get('MoS2', {})
    D0 = abs(ref.get('dgap_deps', 100.0))
    w0 = WLA_EXP['MoS2']
    dl0 = abs(E_LASER - EA['MoS2'])
    for m in MATS:
        d = out[m]
        D = abs(d.get('dgap_deps', D0))
        wla = WLA_EXP.get(m)
        if wla is None:                                # MoSe2
            v = np.sqrt(d.get('C2D', 130.0) / 4.42e-6)
            v0 = np.sqrt(ref.get('C2D', 130.0) / 3.03e-6)
            wla = w0 * (v / v0) * (A0['MoS2'] / A0[m])
            d['wLA_pred'] = float(wla)
        dl = abs(E_LASER - EA[m])
        d['C_A'] = float(C_A_MoS2 * (D / D0) ** 2 * (w0 / wla) ** 2
                         * (dl0 / dl) ** 2)
        d['C_E'] = float(d['C_A'] * 1.11 / 0.59)
        d['dA'] = float(-2.20 * (D / D0))
        d['dE'] = float(-0.33 * (D / D0))

    with open(os.path.join(HERE, 'dft_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
