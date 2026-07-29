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
# That work reports two very different gate stacks: a 96 nm SiO2 back gate and
# a 5.5 to 7.5 nm HfO2 local back gate of about 1.5 nm equivalent oxide
# thickness.  Only the high-kappa devices are modelled quantitatively here.
# The reason is the gate overdrive.  On the high-kappa stack an overdrive of
# order 1 V is the operating point, so the 1.5 V used throughout this work is
# a fair comparison.  The back-gated devices are driven to tens of volts on a
# 96 nm oxide to reach the same sheet density, and the per-device gate voltage
# is not reported, so any model current quoted for them would rest on an
# assumed bias.  Their on-currents are recorded below for reference only.
# Channels were defined by electron-beam lithography and a XeF2 dry chemical
# etch, not by reactive-ion etching.
PENA = dict(
    mu_FE=(30.0, 60.0),                    # cm^2 / V s at 300 K, 75 nm on SiO2
    width_no_degradation_nm=(75.0, 850.0),
    smallest_width_nm=25.0,
    # name: (material, carrier, W_nm, L_nm, stack, I_on uA/um at Vds = 1 V)
    devices=(
        ('MoS2_50nm_HfO2', 'MoS2', 'e', 50.0, 50.0, 'HfO2_EOT1p5', 460.0),
        ('WS2_50nm_HfO2', 'WS2', 'e', 50.0, 50.0, 'HfO2_EOT1p5', 420.0),
        ('WSe2_50nm_HfO2', 'WSe2', 'h', 50.0, 50.0, 'HfO2_EOT1p5', 130.0),
    ),
    # Back-gated on a 96 nm SiO2 oxide, at an unreported gate overdrive:
    # (material, W_nm, L_nm, I_on uA/um at Vds = 1 V).  Quoted, not modelled.
    backgated=(
        ('MoS2', 25.0, 50.0, 310.0),
        ('MoS2', 43.0, 300.0, 620.0),
        ('MoS2', 75.0, 300.0, 400.0),
    ),
    # 60 nm MoS2 on the high-kappa stack, 560 uA/um, channel length not stated
    # in the paper, so it is quoted but not modelled.
    Ion_60nm_HfO2=560.0,
    # Transfer-length extraction on 75 nm MoS2 ribbons with Au contacts,
    # quoted per contact at the highest gate overdrive.
    Rc_ohm_um=560.0,
    Rc_fit_ohm_um=(190.0, 370.0),
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
# Shift rates are cm^-1 per per cent of biaxial strain; the Grueneisen
# parameters are quoted in the same convention used here, gamma = -(1/2 w)
# dw/deps, so that they can be compared directly with the calculation.
# The WSe2 measurement includes the 2LA overtone, which is the acoustic
# channel this work relies on, so it is the one direct test of the acoustic
# lever arm that exists.
MICHAIL_BIAX = {
    'MoS2': dict(dwE=-4.3, dwE_e=0.1, dwA=-2.5, dwA_e=0.3,
                 gE=0.56, gE_e=0.02, gA=0.31, gA_e=0.02,
                 dw2LA=np.nan, dw2LA_e=np.nan,
                 gLA=np.nan, gLA_e=np.nan),
    'WSe2': dict(dwE=np.nan, dwE_e=np.nan, dwA=-1.3, dwA_e=0.1,
                 gE=np.nan, gE_e=np.nan, gA=0.26, gA_e=0.02,
                 dw2LA=-2.3, dw2LA_e=0.1,
                 gLA=0.45, gLA_e=0.02),
}
# Michail et al., J. Phys. Chem. C 127, 3506 (2023) doi:10.1021/acs.jpcc.2c06933
# report the first controlled pure biaxial strain measurement on monolayer
# WS2 and extract the Grueneisen parameter of the in-plane E' mode.  The
# numerical value is behind a paywall and is therefore cited for the
# existence of the measurement only, and is not used in any calculation.
# No biaxial-strain measurement of monolayer MoSe2 has been published.

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
# Damage-halo widths. The helium-ion value is measured: Liu et al. report
# "damage up to 150 nm beyond the patterned edge". No halo width has been
# reported for a XeF2-etched ribbon; the 5 nm figure is a modelling choice for
# a gentle etch, motivated by the absence of defect-activated Raman modes and
# the unbroadened tip-enhanced photoluminescence reported for those ribbons,
# and it is scanned rather than relied upon.
HALO = dict(HIM_nm=150.0, GENTLE_nm=5.0)

# ---------------------------------------------------------------------------
# Liu, Gu and Ye, IEEE Electron Device Lett. 33, 1273 (2012)
# doi:10.1109/LED.2012.2202630
# Width-driven threshold shift in MoS2 nanoribbon transistors. The flakes are
# 6, 6 and 11 nm thick, so these are multilayer bodies, not monolayers, and
# the comparison in the article is an onset and order-of-magnitude check
# rather than a quantitative one. Channel length 1 um, 300 nm SiO2 back gate.
# The paper gives no per-width threshold voltage, only the three statements
# below, so nothing more can be read out of it.
# ---------------------------------------------------------------------------
LIU2012 = dict(
    widths_nm=(2000.0, 1000.0, 500.0, 200.0, 100.0, 80.0, 60.0),
    thickness_nm=(6.0, 6.0, 11.0),
    Lch_nm=1000.0, tox_nm=300.0,
    W_constant_above_nm=500.0,     # V_T flat for wider ribbons
    W_onset_nm=200.0,              # positive shift first observed
    dVT_total_V=50.0,              # -20 V to +30 V for the 6 nm body, at 60 nm
    narrowest_nm=60.0,
)

# ---------------------------------------------------------------------------
# Henriquez-Guerra et al., ACS Appl. Mater. Interfaces 15, 57369 (2023)
# doi:10.1021/acsami.3c13281    A-exciton biaxial gauge factors, meV per %.
# ---------------------------------------------------------------------------
GAUGE = {'WS2': (-129.0, 3.0), 'WSe2': (-120.0, 3.0),
         'MoS2': (-100.0, 3.0), 'MoSe2': (-64.0, 4.0)}
