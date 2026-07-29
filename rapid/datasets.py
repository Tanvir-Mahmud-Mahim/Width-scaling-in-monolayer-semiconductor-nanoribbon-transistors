"""Published data sets used to calibrate and to test the framework.

Every entry records the source so that the provenance of each number used in
the manuscript can be traced.  Values were read from the tables, text and
figure annotations of the cited works.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Nattoo et al., ACS Appl. Mater. Interfaces 17, 47347 (2025)
# doi:10.1021/acsami.5c04483
# Few-layer MoS2 grown by ALD and by RF sputtering, before and after a 900 C
# rapid thermal anneal.  Two independent disorder-activated Raman metrics and
# the transfer-length-method effective mobility.
# ---------------------------------------------------------------------------
NATTOO = [
    dict(label='ALD as-grown',       R_sh=0.325, R_sh_e=0.031,
         R_LA=0.043, R_LA_e=0.003, mu=0.16, mu_e=0.07),
    dict(label='ALD annealed',       R_sh=0.230, R_sh_e=0.068,
         R_LA=0.032, R_LA_e=0.002, mu=0.44, mu_e=0.09),
    dict(label='sputtered as-grown', R_sh=0.687, R_sh_e=0.022,
         R_LA=0.128, R_LA_e=0.004, mu=0.030, mu_e=0.010),
    dict(label='sputtered annealed', R_sh=0.507, R_sh_e=0.012,
         R_LA=0.087, R_LA_e=0.007, mu=np.nan, mu_e=np.nan),
]

# ---------------------------------------------------------------------------
# Krayev et al., Appl. Phys. Lett. 128, 203102 (2026)  doi:10.1063/5.0321304
# Gap-mode tip-enhanced Raman maps of monolayer MoS2 nanoribbons on gold.
# ---------------------------------------------------------------------------
KRAYEV = dict(
    ribbon_width_nm=60.0,
    edge_dwA=-0.5,          # A1' shift at the ribbon edge, cm^-1
    edge_dw2LA=0.0,         # 2LA(M) unchanged at the edge
    spot_dwA=-0.6,          # inhomogeneity spot, cm^-1
    spot_dw2LA=-2.8,        # 2LA(M) shift at the same spot, cm^-1
    inhom_size_nm=(50.0, 100.0),
)

# ---------------------------------------------------------------------------
# Peng et al., 2D Mater. 13, 025005 (2026)  doi:10.1088/2053-1583/ae33d0
# Wide-field hyperspectral differential reflectance of monolayer WS2.
# ---------------------------------------------------------------------------
PENG = dict(
    EA_grain=1.98,          # A-exciton energy inside the grain, eV
    EA_gb=2.03,             # at the grain boundary, eV
    dE_gb=0.050,            # blue shift, eV
    strain_gb_reported=-0.6,  # percent compressive, as estimated in that work
    vdoped_peak=[(0.0, 1.95), (0.4, 1.94), (0.7, 1.93), (12.0, 1.90),
                 (17.0, 1.89), (30.0, 1.87)],   # nominal V %, DR peak eV
)

# ---------------------------------------------------------------------------
# Pena et al., Nature Nanotechnology 21, 803 (2026)
# doi:10.1038/s41565-026-02161-w   Monolayer TMD nanoribbon transistors.
# ---------------------------------------------------------------------------
PENA = dict(
    mu_FE=(30.0, 60.0),                    # cm^2 / V s at 300 K
    width_no_degradation_nm=(75.0, 850.0),
    smallest_width_nm=25.0,
    Ion=dict(MoS2_HfO2=560.0, MoS2_SiO2_43nm=620.0,
             MoS2_75nm=400.0, MoS2_25nm=310.0,
             WS2=420.0, WSe2=130.0),        # uA/um at Vds = 1 V
    Rc_ohm_um=560.0,
    edge_roughness_nm=3.0,
)

# ---------------------------------------------------------------------------
# Yang, Pena et al., ACS Appl. Mater. Interfaces 18, 10161 (2026)
# doi:10.1021/acsami.5c19328   Lateral-force-microscopy defect counting.
# ---------------------------------------------------------------------------
YANG_DEFECTS = {
    'WS2 SS-CVD': (1.3e12, 0.5e12),
    'WSe2 SS-CVD': (2.2e12, 0.3e12),
    'WSe2 MOCVD': (5.3e13, 1.6e13),
    'WSe2 device': (5.9e12, 1.3e12),
}

# ---------------------------------------------------------------------------
# Dossena et al., npj 2D Mater. Appl. 9, 67 (2025)  doi:10.1038/s41699-025-00587-9
# First-principles quantum transport in disordered WS2/Al2O3 stacks.
# ---------------------------------------------------------------------------
DOSSENA = dict(mu_phonon=312.0,
               points=[(2.0e13, 56.5), (4.0e13, 8.5)])   # (n_defect, mu)

# ---------------------------------------------------------------------------
# Mignuzzi et al., Phys. Rev. B 91, 195411 (2015) doi:10.1103/PhysRevB.91.195411
# ---------------------------------------------------------------------------
MIGNUZZI = dict(C_A=0.59, C_A_err=0.03, C_E=1.11, C_E_err=0.08,
                alpha_A=0.5, alpha_E=0.8, wLA=227.6, laser_nm=532.0)

# ---------------------------------------------------------------------------
# Michail et al., ACS Appl. Mater. Interfaces 16, 49602 (2024)
# doi:10.1021/acsami.4c07216    Direct biaxial strain calibration.
# ---------------------------------------------------------------------------
MICHAIL_BIAX = {
    'MoS2': dict(dwE=-4.3, dwE_e=0.1, dwA=-2.5, dwA_e=0.3,
                 gE=0.56, gE_e=0.02, gA=0.31, gA_e=0.02),
    'WSe2': dict(dwE=np.nan, dwE_e=np.nan, dwA=-1.3, dwA_e=0.1,
                 gE=np.nan, gE_e=np.nan, gA=0.28, gA_e=0.02),
}
# No published direct biaxial-strain measurement of the mode Grueneisen
# parameters of monolayer WS2 or MoSe2 is available, so MICHAIL_BIAX above
# (MoS2 and WSe2 only) is the complete set of measured comparison points.

# ---------------------------------------------------------------------------
# Chakraborty et al., Phys. Rev. B 85, 161403(R) (2012)
# doi:10.1103/PhysRevB.85.161403   Phonon renormalisation by electron doping.
# ---------------------------------------------------------------------------
CHAKRABORTY = dict(n=1.8e13, dwA=-4.0, dwE=-0.6, dGamA=6.0)

# ---------------------------------------------------------------------------
# Jin et al., Phys. Rev. B 90, 045422 (2014) doi:10.1103/PhysRevB.90.045422
# Intrinsic phonon-limited mobilities, K-valley dominated (cm^2 / V s).
# ---------------------------------------------------------------------------
JIN_MU = {'MoS2': dict(e=320.0, h=270.0), 'WS2': dict(e=690.0, h=540.0),
          'MoSe2': dict(e=180.0, h=90.0), 'WSe2': dict(e=250.0, h=270.0)}

# ---------------------------------------------------------------------------
# Liu et al., Adv. Funct. Mater. 36, 2514880 (2026) doi:10.1002/adfm.202514880
# Helium-ion patterned MoS2 nanoribbons: damage halo beyond the exposed edge.
# ---------------------------------------------------------------------------
HALO = dict(HIM_nm=150.0, RIE_nm=5.0)

# ---------------------------------------------------------------------------
# Henriquez-Guerra et al., ACS Appl. Mater. Interfaces 15, 57369 (2023)
# doi:10.1021/acsami.3c13281    A-exciton biaxial gauge factors, meV per %.
# ---------------------------------------------------------------------------
GAUGE = {'WS2': (-129.0, 3.0), 'WSe2': (-120.0, 3.0),
         'MoS2': (-100.0, 3.0), 'MoSe2': (-64.0, 4.0)}
