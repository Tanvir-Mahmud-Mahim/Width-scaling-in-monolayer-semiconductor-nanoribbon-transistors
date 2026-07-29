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
    'MU_PHONON': 'lit: Jin et al., Phys. Rev. B 90, 045422 (2014)',
    'PHONON': 'lit: Mignuzzi 2015, Berkdemir 2013, Terrones 2014',
    'EXCITON': 'lit: Henriquez-Guerra et al., ACS AMI 15, 57369 (2023)',
    'DOPING_MoS2': 'lit: Chakraborty et al., Phys. Rev. B 85, 161403 (2012)',
    'RAMAN_CAL': 'lit: Mignuzzi et al., Phys. Rev. B 91, 195411 (2015)',
    'gamma_E, gamma_A, gamma_LA, C2D, dgap_deps': 'dft: this work',
    'C_A (WS2, MoSe2, WSe2)': 'dft: predicted in this work by transfer',
    'GAMMA_MEAS': 'lit: Michail et al., ACS AMI 16, 49602 (2024), '
                  'MoS2 and WSe2 only',
    'CHI_DOP': 'lit: Chernikov et al., Phys. Rev. Lett. 115, 126802 (2015)',
    'DEQK_DEPS, DEGK_DEPS': 'unused: strain-tuned valley shift, zero strain '
                            'in every reported result',
    'COX, STACK': 'geom: nominal gate-stack capacitance and permittivity',
    'VSAT': 'lit: Smithe et al., Nano Lett. 18, 4516 (2018), MoS2 value '
            'used for all four',
    'P_SPEC': 'assumed: fully diffuse etched edge, p = 0',
    'halo_contrast': 'assumed: 20x defect density inside the damage halo',
    'Udef': 'cal: fixed once on Dossena et al., npj 2D Mater. Appl. 9, 67 (2025)',
    'N_IT': 'cal: fixed once on measured monolayer field-effect mobility',
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
# Raman-active zone-centre modes (cm^-1), 300 K, monolayer, on SiO2
#   MoS2  : Mignuzzi PRB 91, 195411 (2015); Michail APL 108, 173102 (2016)
#   WS2   : Berkdemir Sci. Rep. 3, 1755 (2013)
#   WSe2  : Terrones Sci. Rep. 4, 4215 (2014)
#   MoSe2 : Tonndorf Opt. Express 21, 4908 (2013)
# LA(M) is the disorder-activated zone-edge acoustic phonon.
# --------------------------------------------------------------------------
PHONON = {
    'MoS2':  dict(wE=385.0, wA=403.0, wLA=227.6),
    'WS2':   dict(wE=356.0, wA=417.5, wLA=176.0),
    'MoSe2': dict(wE=287.0, wA=241.0, wLA=None),   # LA(M) from DFT scaling
    'WSe2':  dict(wE=249.4, wA=250.2, wLA=130.0),
}

# --------------------------------------------------------------------------
# A-exciton energy (eV) on SiO2 at 300 K and its biaxial strain gauge factor
# (meV per % strain, negative = redshift under tension).
# Gauge factors: Henriquez-Guerra et al., ACS Appl. Mater. Interfaces 15,
# 57369 (2023) [doi:10.1021/acsami.3c13281]
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
#     phonon correlation length L_C = alpha * L_D, alpha = 0.5 (A1'), 0.8 (E')
# --------------------------------------------------------------------------
RAMAN_CAL = dict(C_A=0.59, C_A_err=0.03, C_E=1.11, C_E_err=0.08,
                 alpha_A=0.5, alpha_E=0.8, laser_nm=532.0)

# --------------------------------------------------------------------------
# Measured biaxial-strain mode Grueneisen parameters, where they exist.  When
# Direct biaxial-strain measurements exist only for MoS2 (both optical modes)
# and for the A1' mode of WSe2: Michail et al., ACS Appl. Mater. Interfaces
# 16, 49602 (2024).  That work does not cover WS2 or MoSe2, and no biaxial
# measurement exists for the zone-edge acoustic branch of any of the four, so
# the calculated values are used throughout and these entries serve only as a
# comparison.
# --------------------------------------------------------------------------
GAMMA_MEAS = {
    'MoS2':  dict(gE=0.56, gA=0.31),
    'WS2':   dict(),
    'MoSe2': dict(),
    'WSe2':  dict(gA=0.28),
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
        # The two lever arms used in the charge/strain separation must be
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
        self.alpha_A = RAMAN_CAL['alpha_A']
        self.alpha_E = RAMAN_CAL['alpha_E']
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
