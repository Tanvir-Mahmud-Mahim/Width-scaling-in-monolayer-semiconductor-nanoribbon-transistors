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
         R_LA=0.128, R_LA_e=0.005, mu=0.030, mu_e=0.010),
    dict(label='sputtered annealed', R_sh=0.507, R_sh_e=0.012,
         R_LA=0.087, R_LA_e=0.007, mu=np.nan, mu_e=np.nan),
]

# ---------------------------------------------------------------------------
# Krayev et al., Appl. Phys. Lett. 128, 203102 (2026)  doi:10.1063/5.0321304
# (author-posted full text: poplab.stanford.edu/pdfs/Krayev-TMDnanoribbonTERS-apl26.pdf)
# Monolayer MoS2 grown by solid-source CVD on 300 nm SiO2, patterned by
# electron-beam lithography and dry-etched with XeF2, template-stripped onto
# gold (100 nm Au evaporated, flipped, stripped) and imaged by gap-mode
# tip-enhanced Raman spectroscopy at 785 nm (below the A exciton), spatial
# resolution about 10 nm.  The ribbons analysed here are from the second
# batch, which had an additional overnight vacuum anneal (250 C) that removed
# resist residues.  Edge values (Fig. 4 of that work, averages of about ten
# pixels): A1' redshift of about 0.5 cm^-1, 2LA(M) unchanged.  Interior
# inhomogeneity (Fig. 3, spot 2 against spot 1): redshifts of 2.8 cm^-1 for
# 2LA(M) and 0.6 cm^-1 for A1'.  The authors read the edge shift as
# consistent with negative fixed charge, likely from fabrication and/or
# atmospheric exposure.  The 60 nm width is that of the first-batch ribbon
# (Fig. 2); the width of the second-batch ribbon is not stated.  All seven
# authors are also authors of the transistor study below (same lithography
# and XeF2 etch).
# ---------------------------------------------------------------------------
KRAYEV = dict(
    ribbon_width_nm=60.0,   # first-batch ribbon (Fig. 2), reference only
    laser_nm=785.0,
    resolution_nm=10.0,
    n_ribbons=1, n_edges=2,
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
    EA_grain=2.00,          # A-exciton energy inside the grain, eV
    EA_gb=2.05,             # at the grain boundary, eV
    dE_gb=0.050,            # blue shift, eV
    strain_gb_reported=-0.6,  # percent compressive, as estimated in that work
)

# ---------------------------------------------------------------------------
# Sousa et al., Nanoscale 16, 12923 (2024)  doi:10.1039/D4NR00837E
# (preprint arXiv:2401.11984).  Tip-enhanced Raman and photoluminescence of
# grown MoS2 monolayers on fused silica, 632.8 nm excitation.  At the edge of
# sample A1, which atomic force microscopy shows to be suspended by about
# 2 nm, E' redshifts by 1.5 cm^-1, the 2LA band by 8.5 cm^-1, and no
# substantial A1' shift is seen.  The authors read the E' shift as about
# 0.7 % strain and attribute the 2LA change to a change in the relative
# weights of its resonant components rather than to strain.  Used in the
# text as qualitative context only: it is a grown, not an etched, edge and
# no A1' value is given.
# ---------------------------------------------------------------------------
SOUSA_EDGE = dict(dwE=-1.5, dw2LA=-8.5, dwA_detected=False,
                  strain_reported_pct=0.7, laser_nm=632.8,
                  suspension_nm=2.0)

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
# etch, not by reactive-ion etching.  The same work reports no defect-
# activated Raman modes, an unbroadened A exciton and a slightly larger
# trion-to-exciton ratio in the ribbons than in wide regions, which its
# authors note potentially suggests doping from the edges, and no measurable
# off-state edge conduction.  No damage-halo width is reported.
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
    # quoted per contact at the highest gate overdrive.  That work gives it
    # as an upper bound, so the modelled currents that include it are lower
    # bounds on what the model predicts for a better contact.
    Rc_ohm_um=560.0,
    Rc_fit_ohm_um=(190.0, 370.0),
    # Per material: the MoS2 bound above, and for WS2 the value that work
    # reports for its WS2 ribbons, R_c ~ 675 +/- 268 ohm um.  No value is
    # reported for the p-type WSe2 contact, which is therefore resolved as a
    # barrier (A12); for the series-resistor estimate it carries the MoS2
    # bound, stated as such.
    Rc_by_material={'MoS2': 560.0, 'WS2': 675.0, 'WSe2': 560.0},
    Rc_WS2_err_ohm_um=268.0,
    edge_roughness_nm=3.0,
)

# ---------------------------------------------------------------------------
# Yang, Pena et al., ACS Appl. Mater. Interfaces 18, 10161 (2026)
# doi:10.1021/acsami.5c19328   Lateral-force-microscopy (LFM) counting of
# surface defects (one side of the monolayer; the work estimates the total at
# about twice the LFM count).  Values are the LFM densities as reported, with
# the upper error bar.  SS-CVD: solid-source CVD.
# ---------------------------------------------------------------------------
YANG_DEFECTS = {
    'WSe2 self-flux': (7.3e11, 1.8e11),
    'WS2 SS-CVD': (1.3e12, 0.5e12),
    'WSe2 SS-CVD': (2.2e12, 0.3e12),
    'WSe2 MOCVD': (5.3e13, 1.6e13),
    'WSe2 device': (5.9e12, 1.3e12),
}

# ---------------------------------------------------------------------------
# Dossena et al., npj 2D Mater. Appl. 9, 67 (2025)  doi:10.1038/s41699-025-00587-9
# First-principles quantum transport in WS2 on Al2O3, with the disorder placed
# in the oxide interface; the two points are two interface models at nearly
# the same carrier density.
# ---------------------------------------------------------------------------
DOSSENA = dict(mu_phonon=312.0,
               points=[(2.0e13, 56.5), (4.0e13, 8.5)])   # (n_defect, mu)

# ---------------------------------------------------------------------------
# Mignuzzi et al., Phys. Rev. B 91, 195411 (2015) doi:10.1103/PhysRevB.91.195411
# ---------------------------------------------------------------------------
MIGNUZZI = dict(C_A=0.59, C_A_err=0.03, C_E=1.11, C_E_err=0.08,
                wLA=227.0, laser_nm=532.0)

# ---------------------------------------------------------------------------
# Michail et al., ACS Appl. Mater. Interfaces 16, 49602 (2024)
# doi:10.1021/acsami.4c07216    Direct biaxial strain calibration.
# ---------------------------------------------------------------------------
# Shift rates are cm^-1 per per cent of biaxial strain; the Grueneisen
# parameters are quoted in the same convention used here, gamma = -(1/2 w)
# dw/deps, so that they can be compared directly with the calculation.
# The WSe2 measurement includes the 2LA overtone, which is the acoustic
# channel this work relies on, so it is the one direct test of the acoustic
# strain sensitivity that we know of.
MICHAIL_BIAX = {
    'MoS2': dict(dwE=-4.3, dwE_e=0.1, dwA=-2.5, dwA_e=0.3,
                 gE=0.56, gE_e=0.02, gA=0.31, gA_e=0.02,
                 dw2LA=np.nan, dw2LA_e=np.nan,
                 gLA=np.nan, gLA_e=np.nan),
    # WSe2: means over the five devices WT1-WT5 of Table 2 (515 nm), the same
    # averaging the work applies to MoS2 (Table 1) and to the WSe2 A1' value
    # it quotes, -1.3(1) cm^-1/%, which is used as stated; other errors are
    # standard errors of the mean.
    'WSe2': dict(dwE=np.nan, dwE_e=np.nan, dwA=-1.3, dwA_e=0.1,
                 gE=np.nan, gE_e=np.nan, gA=0.27, gA_e=0.01,
                 dw2LA=-2.37, dw2LA_e=0.15,
                 gLA=0.46, gLA_e=0.03),
}
# Michail et al., J. Phys. Chem. C 127, 3506 (2023) doi:10.1021/acs.jpcc.2c06933
# report the first controlled pure biaxial strain measurement on monolayer
# WS2 and extract the Grueneisen parameter of the in-plane E' mode (equal to
# that of bulk WS2, per the abstract).  It is cited for the existence of the
# measurement only and is not used in any calculation.  We know of no
# biaxial-strain measurement of monolayer MoSe2.

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
# Liu et al., Adv. Funct. Mater. 36, e14880 (2026) doi:10.1002/adfm.202514880
# MoS2 nanoribbons patterned directly with a focused helium-ion beam (beam-
# induced etching with XeF2 gas): damage halo beyond the exposed edge.
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
# below, so nothing more can be read out of it.  Widths trimmed by BCl3/Ar
# plasma dry etching.  The authors attribute the
# shift to edge depletion by electric fields or adsorbed molecules; the 50 V
# total refers to one 6 nm thick set of ribbons.
# ---------------------------------------------------------------------------
LIU2012 = dict(
    etch='BCl3/Ar plasma',
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


# ---------------------------------------------------------------------------
# Electrical cross-checks: threshold voltage against ribbon width.
# ---------------------------------------------------------------------------
# Pena et al., Nat. Nanotechnol. 21, 803 (2026), Extended Data Fig. 4
# ("Width-dependent threshold voltage across monolayer MoS2 nanoribbons").
# Monolayer MoS2 ribbons made by the same electron-beam lithography and XeF2
# etch as the ribbons of the tip-enhanced Raman data, on a ~5.5 nm HfO2 local
# back gate; V_T,cc at a constant current of 100 nA/um, forward sweep.  The
# caption states that V_T,cc increases as W_ch is reduced and attributes it,
# with earlier work, to negative charge at the edges.  The paper gives the
# EOT as ~1.5 nm for its HfO2 back gates without naming the thickness; the
# Methods tie that figure to the 7.5 nm films, so for the 5.5 nm films the
# same permittivity gives ~1.1 nm.  Both are used.
#
# The values below were DIGITISED BY US from the published image
# (41565_2026_2161_Fig8_ESM.jpg, 832 x 663 px): marker centroids located by
# colour against the axis ticks (V: 1.0 V = 56.5 px ... -1.0 V = 505 px, i.e.
# 224 px per volt; L: 0 at 164 px, 1 um at 797 px).  Reading uncertainty is
# about 0.01 V for isolated markers and about 0.02 V for the few markers
# partly hidden by others; both are far below the device-to-device spread.
# (W_ch nominal nm, L_ch um, V_T,cc V)
PENA_VT = (
    # W ~ 850 nm (grey triangles, 18 devices)
    (850, 1.00, -0.05), (850, 1.00, -0.35),
    (850, 0.50, -0.35), (850, 0.50, -0.47), (850, 0.50, -0.74),
    (850, 0.30, -0.20), (850, 0.30, -0.49), (850, 0.30, -0.58),
    (850, 0.20, -0.49), (850, 0.20, -0.52),
    (850, 0.10, -0.05), (850, 0.10, -0.36), (850, 0.10, -0.62),
    (850, 0.07, -0.42), (850, 0.07, -0.60), (850, 0.07, -0.89),
    (850, 0.05, -1.11), (850, 0.03, -0.90),
    # W ~ 75 nm (blue circles, 15 devices)
    (75, 1.00, 0.27), (75, 1.00, -0.03), (75, 1.00, -0.12),
    (75, 0.50, -0.18), (75, 0.50, -0.21),
    (75, 0.30, -0.01), (75, 0.30, -0.26), (75, 0.30, -0.33), (75, 0.30, -0.43),
    (75, 0.20, 0.01), (75, 0.20, -0.16), (75, 0.20, -0.45),
    (75, 0.10, -0.09),
    (75, 0.07, -0.10), (75, 0.07, -0.17),
    # W ~ 25 nm (purple squares, 4 devices)
    (25, 0.30, 0.96), (25, 0.30, 0.30),
    (25, 0.05, 0.63), (25, 0.05, 0.43),
)
PENA_VT_META = dict(etch='XeF2', stack='HfO2, ~5.5 nm', EOT_nm=(1.1, 1.5),
                    I_cc_nA_per_um=100.0, figure='Extended Data Fig. 4',
                    read_uncertainty_V=0.02)

# Lan et al., Nat. Commun. 17, 7386 (2026), doi:10.1038/s41467-026-74342-z.
# Monolayer MoS2 ribbons defined by Cl2/O2 inductively coupled plasma reactive
# ion etching, 3 nm HfO2 grown at 90 C (kappa ~ 10 as stated), L = 55 to
# 75 nm; "a positive V_T,cc shift of ~0.25 V" between 540 nm and 35 nm wide
# devices (cumulative distributions of many devices), V_T,cc at 100 nA/um and
# V_DS = 0.1 V.  The authors attribute the trend to stronger gate control in
# narrow channels, not to edge charge, so the value is used here as an upper
# bound on any edge-charge contribution.
LAN2026 = dict(etch='Cl2/O2 ICP-RIE', t_HfO2_nm=3.0, kappa=10.0,
               W_wide_nm=540.0, W_narrow_nm=35.0, dVT_V=0.25)

# Kotekar-Patil et al., Appl. Phys. Lett. 114, 013508 (2019),
# doi:10.1063/1.5079860.  Exfoliated monolayer MoS2, SF6 plasma etch, 300 nm
# SiO2 back gate, W = 50 nm, L = 500 nm.  Same flakes before the etch: V_th -30 to
# -10 V; the three ribbons: 20, 31 and 33.4 V.  W is six times smaller than
# the oxide thickness, so fringing fields make the parallel-plate Eq. (1)
# inapplicable; the sign is used, not the size.
KOTEKAR2019 = dict(etch='SF6 plasma', t_SiO2_nm=300.0, W_nm=50.0,
                   Vth_unetched_V=(-30.0, -10.0), Vth_ribbon_V=(20.0, 31.0, 33.4))
