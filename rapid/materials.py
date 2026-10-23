"""Material parameter table for monolayer 1H-MX2 (M = Mo, W; X = S, Se).

Provenance of every group of constants is stated in PROVENANCE below and
repeated in the comment above each table.  Exactly two numbers in the whole
framework are calibrated rather than computed or measured: the neutral
point-defect potential U (transport.py) and the interface trap density n_it
(transport.py).  Both are fixed once, against published first-principles
transport results, and then held for every result reported.
"""
from __future__ import annotations

import json
import os

import numpy as np

MATERIALS = ['MoS2', 'WS2', 'MoSe2', 'WSe2']

PROVENANCE = {
    'BANDS': 'lit: Jin et al., Phys. Rev. B 90, 045422 (2014)',
    'MU_PHONON': 'lit: Jin et al., Phys. Rev. B 90, 045422 (2014), Table III,\n'
                 'the K-valley dominated column',
    'PHONON': 'lit: Michail 2024 and Mignuzzi 2015 (MoS2), Berkdemir 2013\n'
              '(WS2), Terrones 2014 DFPT values (WSe2), Tonndorf 2013 (MoSe2)',
    'EXCITON': 'lit: gauge factors, Henriquez-Guerra et al., ACS AMI 15,\n'
               '57369 (2023); A-exciton energies, Tonndorf 2013 (MoSe2, WSe2)\n'
               'and typical values (MoS2, WS2); linewidths assumed, 35-45 meV',
    'DOPING_MoS2': 'lit: Chakraborty et al., Phys. Rev. B 85, 161403 (2012)',
    'RAMAN_CAL': 'lit: Mignuzzi et al., Phys. Rev. B 91, 195411 (2015)',
    'gamma_E, gamma_A, gamma_LA, C2D, dgap_deps': 'dft: this work',
    'C_A (WS2, MoSe2, WSe2)': 'dft: predicted in this work by transfer',
    'GAMMA_MEAS': 'lit: Michail et al., ACS AMI 16, 49602 (2024): both\n'
                  'optical modes of MoS2, and the A1\' mode and the 2LA\n'
                  'overtone of WSe2. Comparison only, never substituted',
    'CHI_DOP': 'lit: Chernikov et al., Phys. Rev. Lett. 115, 126802 (2015)',
    'DEQK_DEPS, DEGK_DEPS': 'unused: strain-tuned valley shift, zero strain '
                            'in every reported result',
    'COX, STACK': 'geom: nominal gate-stack capacitance and permittivity',
    'VSAT': 'lit: Smithe et al., Nano Lett. 18, 4516 (2018), MoS2 value '
            'used for all four',
    'P_SPEC': 'assumed: fully diffuse etched edge, p = 0',
    'halo_contrast': 'assumed: 20x defect density inside the damage halo;\n'
                     'scanned from 5x to 100x',
    'Udef': 'cal: fixed once on Dossena et al., npj 2D Mater. Appl. 9, 67\n'
            '(2025), WS2 on disordered Al2O3, used as a short-range\n'
            'scattering proxy',
    'N_IT': 'cal: fixed once on measured monolayer field-effect mobility',
    'n_d (reference film)': 'lit: Yang et al., ACS AMI 18, 10161 (2026),\n'
                            'lateral-force-microscopy count, grown WS2',
    'edge shifts': 'lit: Krayev et al., Appl. Phys. Lett. 128, 203102 (2026),\n'
                   'edges of one XeF2-etched, annealed ribbon on gold, 785 nm',
    'lambda_2LA': 'assumed: zero for MoS2, where it has not been measured;\n'
                  'the value that would reverse the result is computed',
    'w_e': 'assumed: 10 nm, the resolution of the measurement; scanned\n'
           'from 5 to 20 nm',
    'halo (XeF2 etch)': 'assumed: 5 nm, no measured value; scanned from\n'
                        '1 to 200 nm',
    'halo (He-ion beam)': 'lit: Liu et al., Adv. Funct. Mater. 36, e14880\n'
                          '(2026), damage up to 150 nm',
}

# --------------------------------------------------------------------------
# physical constants (SI unless noted)
# --------------------------------------------------------------------------
HBAR = 1.054571817e-34
QE = 1.602176634e-19
M0 = 9.1093837015e-31
KB = 1.380649e-23
EPS0 = 8.8541878128e-12
C_LIGHT = 2.99792458e8
T300 = 300.0
KT300 = KB * T300 / QE          # 0.02585 eV

# --------------------------------------------------------------------------
# band-structure parameters
# masses in units of m0; valley separations in eV
# Jin, Li, Mullen, Kim, Phys. Rev. B 90, 045422 (2014)  [doi:10.1103/PhysRevB.90.045422]
# cross-checked against Kormanyos et al., 2D Mater. 2, 022001 (2015)
# --------------------------------------------------------------------------
BANDS = {
    #            m_cK  m_cQ  m_vK  m_vG   E_QK    E_GK
    'MoS2':  dict(mcK=0.51, mcQ=0.76, mvK=0.58, mvG=4.05, EQK=0.081, EGK=0.148),
    'WS2':   dict(mcK=0.31, mcQ=0.60, mvK=0.42, mvG=4.07, EQK=0.067, EGK=0.173),
    'MoSe2': dict(mcK=0.64, mcQ=0.80, mvK=0.71, mvG=7.76, EQK=0.028, EGK=0.374),
    'WSe2':  dict(mcK=0.39, mcQ=0.64, mvK=0.51, mvG=7.77, EQK=0.016, EGK=0.427),
}
VALLEY_DEG = dict(K=2, Q=6)      # K/K' and the six Q (Lambda) pockets

# Intrinsic room-temperature phonon-limited mobility, K-valley dominated
# electrons, Jin et al. 2014 Table III (used as the phonon-limited ceiling).
MU_PHONON = {'MoS2': 320.0, 'WS2': 690.0, 'MoSe2': 180.0, 'WSe2': 250.0}

# --------------------------------------------------------------------------
# Raman-active zone-centre modes (cm^-1) of the monolayer, 300 K
#   MoS2  : E' and A1' are the zero-strain positions of Michail et al., ACS
#           Appl. Mater. Interfaces 16, 49602 (2024), Table 1, mean of the
#           four devices; LA(M) at about 227 cm^-1, Mignuzzi et al., Phys.
#           Rev. B 91, 195411 (2015)
#   WS2   : Berkdemir et al., Sci. Rep. 3, 1755 (2013), Table 1, monolayer at
#           488 nm excitation (E' 355.9, A1' 417.5, LA(M) 176 cm^-1)
#   WSe2  : Terrones et al., Sci. Rep. 4, 4215 (2014), DFPT values for the
#           monolayer, which match the measured near-degenerate pair at about
#           250 cm^-1; LA(M) at about 130 cm^-1 (Michail et al. 2024)
#   MoSe2 : Tonndorf et al., Opt. Express 21, 4908 (2013), monolayer
#           (E' 287.2, A1' 240.5 cm^-1)
# LA(M) is the disorder-activated zone-edge acoustic phonon.
# --------------------------------------------------------------------------
PHONON = {
    'MoS2':  dict(wE=384.4, wA=403.0, wLA=227.0),
    'WS2':   dict(wE=355.9, wA=417.5, wLA=176.0),
    'MoSe2': dict(wE=287.2, wA=240.5, wLA=None),   # LA(M) from DFT scaling
    'WSe2':  dict(wE=249.4, wA=250.2, wLA=130.0),
}

# --------------------------------------------------------------------------
# A-exciton energy (eV) at 300 K, its biaxial strain gauge factor (meV per %
# strain, negative = redshift under tension) and a room-temperature linewidth.
# Gauge factors: Henriquez-Guerra et al., ACS Appl. Mater. Interfaces 15,
# 57369 (2023) [doi:10.1021/acsami.3c13281], Table 1, neutral A exciton.
# Energies: Tonndorf et al. 2013 PL (MoSe2 1.57 eV, WSe2 1.65 eV); MoS2 and
# WS2 are typical room-temperature values.  Gam0 is an assumed typical
# linewidth, used only to spread the detuning in the C_A transfer.
# --------------------------------------------------------------------------
EXCITON = {
    'MoS2':  dict(EA=1.880, gauge=-100.0, Gam0=0.045),
    'WS2':   dict(EA=2.010, gauge=-129.0, Gam0=0.035),
    'MoSe2': dict(EA=1.570, gauge=-64.0, Gam0=0.040),
    'WSe2':  dict(EA=1.650, gauge=-120.0, Gam0=0.038),
}

# --------------------------------------------------------------------------
# Phonon renormalisation by electron doping (cm^-1 per 1e13 cm^-2).
# MoS2 measured: Chakraborty et al., Phys. Rev. B 85, 161403(R) (2012).
# For the other three the coupling is scaled by the DFT deformation potential
# ratio (see dft/postprocess.py); this is an explicit prediction.
# --------------------------------------------------------------------------
DOPING_MoS2 = dict(dA=-2.20, dE=-0.33)

# --------------------------------------------------------------------------
# Disorder-activated Raman calibration.
# MoS2 anchor: Mignuzzi et al., Phys. Rev. B 91, 195411 (2015):
#     I(LA)/I(A1') = C_A / L_D^2 ,  C_A = 0.59 +- 0.03 nm^2 at 532 nm
#     I(LA)/I(E' ) = C_E / L_D^2 ,  C_E = 1.11 +- 0.08 nm^2 at 532 nm
# --------------------------------------------------------------------------
RAMAN_CAL = dict(C_A=0.59, C_A_err=0.03, C_E=1.11, C_E_err=0.08,
                 laser_nm=532.0)

# --------------------------------------------------------------------------
# Measured biaxial-strain mode Grueneisen parameters, where they exist, in the
# convention gamma = -(1/2 w) dw/deps used throughout.  Michail et al., ACS
# Appl. Mater. Interfaces 16, 49602 (2024) cover both optical modes of MoS2
# and, for WSe2, the A1' mode and the 2LA overtone; that last one is the only
# direct biaxial measurement of a zone-edge acoustic Grueneisen parameter in
# this family that we know of, and it is the value the computed acoustic
# strain sensitivity is checked against.  WSe2 entries are means over the five
# devices of that work (Table 2), as for MoS2 (Table 1).  The E' mode of WS2
# has been measured biaxially by Michail et al., J. Phys. Chem. C 127, 3506
# (2023), but the value is not reproduced here and is not used.  We know of no
# measurement for MoSe2.  The calculated values are used throughout; these
# entries serve as a comparison and set the low end of the acoustic bracket.
# --------------------------------------------------------------------------
GAMMA_MEAS = {
    'MoS2':  dict(gE=0.56, gA=0.31),
    'WS2':   dict(),
    'MoSe2': dict(),
    'WSe2':  dict(gA=0.27, gLA=0.46),
}

# elastic / lattice
LATT = {'MoS2': 3.184, 'WS2': 3.183, 'MoSe2': 3.319, 'WSe2': 3.322}
def _rho2d(name):
    """Areal mass density (kg/m^2) from the lattice constant and masses."""
    import numpy as _np
    AMU = 1.66053906660e-27
    m = {'MoS2': 95.95 + 2 * 32.06, 'WS2': 183.84 + 2 * 32.06,
         'MoSe2': 95.95 + 2 * 78.97, 'WSe2': 183.84 + 2 * 78.97}[name]
    a = LATT[name] * 1e-10
    return m * AMU / (_np.sqrt(3) / 2 * a ** 2)


RHO2D = {k: _rho2d(k) for k in MATERIALS}

# environment
EPS_SIO2 = 3.9
EPS_HFO2 = 22.0
EPS_TOP = 1.0
EPS_ENV = (EPS_SIO2 + EPS_TOP) / 2.0

# Single calibrated defect potential, eV nm^2 (see PROVENANCE).
U_DEFECT = 0.285

DFT_FILE = os.path.join(os.path.dirname(__file__), '..', 'dft', 'dft_summary.json')


def load_dft(path: str | None = None) -> dict:
    """Load the post-processed first-principles summary, if it exists."""
    path = path or DFT_FILE
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


class Material:
    """Bundle of all parameters needed by the forward models."""

    def __init__(self, name: str, dft: dict | None = None):
        if name not in MATERIALS:
            raise KeyError(name)
        self.name = name
        d = dft if dft is not None else load_dft()
        self.dft = d.get(name, {})
        b = BANDS[name]
        self.mcK, self.mcQ = b['mcK'], b['mcQ']
        self.mvK, self.mvG = b['mvK'], b['mvG']
        self.EQK, self.EGK = b['EQK'], b['EGK']
        self.mu_ph = MU_PHONON[name]
        p = PHONON[name]
        self.wE = self.dft.get('wE_exp_anchor', p['wE'])
        self.wA = self.dft.get('wA_exp_anchor', p['wA'])
        self.wLA = p['wLA'] if p['wLA'] is not None else self.dft.get('wLA_pred')
        x = EXCITON[name]
        self.EA0, self.gaugeA, self.GamX0 = x['EA'], x['gauge'], x['Gam0']
        self.a0 = LATT[name]
        self.rho2d = RHO2D[name]
        # Grueneisen parameters: measured value where one exists, otherwise
        # the value computed in this work.  Both are kept.
        self.gE_dft = self.dft.get('gamma_E', float('nan'))
        self.gA_dft = self.dft.get('gamma_A', float('nan'))
        # The two strain sensitivities used in the charge/strain separation must be
        # internally consistent, so both come from the same calculation.  The
        # published biaxial measurements, which are limited by strain transfer
        # through the substrate, are kept for comparison in GAMMA_MEAS.
        gm = GAMMA_MEAS[name]
        # No silent substitution: if the first-principles value is missing the
        # constructor fails loudly rather than borrowing another material's.
        if not (self.gE_dft == self.gE_dft and self.gA_dft == self.gA_dft):
            raise ValueError(
                'missing first-principles Grueneisen parameters for %s; '
                'run dft/postprocess.py' % name)
        self.gE, self.gA = self.gE_dft, self.gA_dft
        self.gE_meas = gm.get('gE')
        self.gA_meas = gm.get('gA')
        # elastic modulus, 2D (N/m), from the biaxial strain sweep
        if 'C2D' not in self.dft:
            raise ValueError('missing C2D for %s; run dft/postprocess.py'
                             % name)
        self.C2D = self.dft['C2D']
        # doping coefficients
        self.dA = self.dft.get('dA', DOPING_MoS2['dA'])
        self.dE = self.dft.get('dE', DOPING_MoS2['dE'])
        # disorder-activated Raman constants (nm^2)
        self.C_A = self.dft.get('C_A', RAMAN_CAL['C_A'])
        self.C_E = self.dft.get('C_E', RAMAN_CAL['C_E'])
        # neutral point-defect scattering potential (eV nm^2).  Calibrated
        # once against the first-principles disordered-WS2 transport result of
        # Dossena et al., npj 2D Mater. Appl. 9, 67 (2025) and held fixed.
        self.Udef = self.dft.get('Udef', U_DEFECT)

    # ---- derived quantities -------------------------------------------
    @property
    def vLA(self) -> float:
        """Longitudinal sound velocity (m/s) from the 2D elastic modulus."""
        return float(np.sqrt(self.C2D / self.rho2d))

    def __repr__(self):
        return f'<Material {self.name}>'


def all_materials(dft: dict | None = None):
    d = dft if dft is not None else load_dft()
    return {m: Material(m, d) for m in MATERIALS}
