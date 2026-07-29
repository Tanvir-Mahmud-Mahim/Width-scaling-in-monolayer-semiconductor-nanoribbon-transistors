"""Export every number quoted in the manuscript as a LaTeX macro.

This guarantees that the text, the tables and the figures can never disagree:
paper/numbers.tex is regenerated from results.json on every run.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import datasets as D, transport
from .materials import all_materials, MATERIALS

ROOT = os.path.join(os.path.dirname(__file__), '..')


def _fmt_sci(x, nd=1):
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0 ** e
    return r'%.*f\times 10^{%d}' % (nd, m, e)


def main():
    with open(os.path.join(ROOT, 'results.json')) as fh:
        r = json.load(fh)
    mats = all_materials()
    L = []

    def cmd(name, val):
        L.append(r'\newcommand{\%s}{%s}' % (name, val))

    # ---- adjoint verification
    av = r['adjoint_verification']['MoS2']
    cmd('AdjJacErr', _fmt_sci(av['jacobian_vs_fd']))
    cmd('AdjDotErr', _fmt_sci(av['dot_product_error']))
    cmd('AdjMapErr', _fmt_sci(av['map_gradient_error']))

    # ---- conditioning
    c = r['conditioning']['MoS2']
    cmd('CondNum', '%.1f' % c['condition_number'])
    cmd('SigNd', '%.2f' % c['sigma_post'][0])
    cmd('SigEps', '%.3f' % c['sigma_post'][1])
    cmd('SigN', '%.3f' % c['sigma_post'][2])
    cmd('NdRelErr', '%.0f' % c['nd_rel_err_percent'])

    # ---- map recovery
    mr = r['map_recovery']
    cmd('RmseNd', '%.3f' % mr['rmse'][0])
    cmd('RmseEps', '%.3f' % mr['rmse'][1])
    cmd('RmseN', '%.3f' % mr['rmse'][2])
    Jv = np.array(mr['J'])
    conv = int(np.argmax(Jv <= 1.01 * Jv[-1]))
    cmd('MapIter', '%d' % conv)
    cmd('MapIterMax', '%d' % (len(Jv) - 1))

    # ---- Nattoo
    na = r['nattoo']
    cmd('NattooRatio', '%.1f' % na['channel_ratio_mean'])
    cmd('NattooRatioSD', '%.1f' % na['channel_ratio_std'])
    cmd('NattooScatter', '%.0f' % (100 * na['channel_ratio_rel_scatter']))
    cmd('NattooRatioALD', '%.1f' % na['channel_ratio_ALD'])
    cmd('NattooRatioSputt', '%.1f' % na['channel_ratio_sputtered'])
    for row, tag in zip(na['rows'], ['ALDag', 'ALDan', 'SPag', 'SPan']):
        cmd('nd' + tag, _fmt_sci(row['nd']))
        cmd('LD' + tag, '%.1f' % row['LD_nm'])
        cmd('muceil' + tag, '%.0f' % row['mu_ceiling'])
    defs = [row['mu_ceiling'] / row['mu_meas'] for row in na['rows']
            if row['mu_meas'] == row['mu_meas']]
    cmd('DeficitLo', '%.0f' % min(defs))
    cmd('DeficitHi', '%.0f' % max(defs))

    # ---- Krayev
    kv = r['krayev']
    cmd('EdgeN', _fmt_sci(kv['edge']['n_cm2']))
    cmd('EdgeNerr', _fmt_sci(kv['edge']['n_err'] * 1e13))
    cmd('EdgeStrain', '%.3f' % kv['edge']['strain_pct'])
    cmd('EdgeStrainErr', '%.3f' % kv['edge']['strain_err'])
    cmd('EdgeStrainBound', '%.2f' % (2.0 * kv['edge']['strain_err']))
    cmd('SpotStrain', '%.3f' % kv['spot']['strain_pct'])
    cmd('SpotStrainErr', '%.3f' % kv['spot']['strain_err'])
    cmd('SpotN', _fmt_sci(abs(kv['spot']['n_cm2'])))
    cmd('NaiveStrain', '%.2f' % kv['naive_strain_only_pct'])
    cmd('GammaLA', '%.2f' % kv['gamma_LA'])
    cmd('dwTwoLA', '%.1f' % abs(kv['dw2LA_deps']))
    cmd('dwAeps', '%.1f' % abs(kv['dwA_deps']))
    cmd('LeverRatio', '%.1f' % (abs(kv['dw2LA_deps'])
                                / abs(kv['dwA_deps'])))
    kr = r['krayev_robust']
    cmd('EdgeNspread', '%.0f' % (100.0 * (max(d['edge_n'] for d in kr)
                                          - min(d['edge_n'] for d in kr))
                                 / kv['edge']['n_cm2']))
    od = r['overtone_doping']
    cmd('OvertoneBound', '%.0f' % (100.0 * (od[-1]['edge_n'] - od[0]['edge_n'])
                                   / od[0]['edge_n']))
    cmd('NaiveTwoLA', '%.1f' % abs(kv['naive_strain_only_pct']
                                   * kv['dw2LA_deps']))

    # ---- the one direct biaxial measurement of an acoustic overtone
    mbw = D.MICHAIL_BIAX['WSe2']
    cmd('GammaLAmeasWSe', '%.2f' % mbw['gLA'])
    cmd('GammaLAmeasWSeErr', '%.2f' % mbw['gLA_e'])
    cmd('GammaAmeasWSe', '%.2f' % mbw['gA'])
    cmd('dwTwoLAmeasWSe', '%.1f' % abs(mbw['dw2LA']))
    cmd('dwAmeasWSe', '%.1f' % abs(mbw['dwA']))
    cmd('LeverRatioMeasWSe', '%.1f' % (abs(mbw['dw2LA']) / abs(mbw['dwA'])))
    _gLwse = mats['WSe2'].dft.get('gamma_LA', float('nan'))
    _wLwse = mats['WSe2'].wLA
    _gAwse, _wAwse = mats['WSe2'].gA, mats['WSe2'].wA
    cmd('LeverRatioPredWSe', '%.1f' % (2.0 * _gLwse * _wLwse
                                       / (_gAwse * _wAwse)))
    cmd('GammaLApredWSe', '%.2f' % _gLwse)
    cmd('dwAmeasMoS', '%.1f' % abs(D.MICHAIL_BIAX['MoS2']['dwA']))
    cmd('GammaAmeasMoS', '%.2f' % D.MICHAIL_BIAX['MoS2']['gA'])
    cmd('GammaEmeasMoS', '%.2f' % D.MICHAIL_BIAX['MoS2']['gE'])
    cmd('StrainTransferLo', '%.0f' % 87.0)
    cmd('StrainTransferHi', '%.0f' % 98.0)

    # ---- Peng
    pg = r['peng']
    cmd('GBstrain', '%.2f' % abs(pg['gb_strain_pct']))
    cmd('GBstrainRef', '%.1f' % abs(pg['gb_strain_reported']))

    # ---- transport validation
    tv = r['transport_validation']
    cmd('DosOneRef', '%.1f' % tv['dossena'][0]['mu_ref'])
    cmd('DosOnePred', '%.0f' % tv['dossena'][0]['mu_pred'])
    cmd('DosTwoRef', '%.1f' % tv['dossena'][1]['mu_ref'])
    cmd('DosTwoPred', '%.0f' % tv['dossena'][1]['mu_pred'])
    cmd('YangWSTwo', '%.0f' % tv['yang']['WS2 SS-CVD']['mu_pred'])
    cmd('YangWSeTwo', '%.0f' % tv['yang']['WSe2 SS-CVD']['mu_pred'])

    # ---- ribbons
    rb = r['ribbons']
    cmd('SigmaLine', _fmt_sci(rb['sigma_line_cm']))
    cmd('Wedge', '%.0f' % rb['w_edge_nm'])
    cmd('Vov', '%.1f' % rb['Vov'])
    for tag, key in (('SiOThreeH', 'SiO2_300nm'), ('SiONinety', 'SiO2_90nm'),
                     ('SiOThirty', 'SiO2_30nm'), ('HfO', 'HfO2_EOT1p5')):
        sp = rb['split'][key]
        cmd('WcHalo' + tag, '%.0f' % sp['halo_only']
            if sp['halo_only'] == sp['halo_only'] else '<5')
        cmd('WcChg' + tag, '%.0f' % sp['charge_only'])
        cmd('WcBoth' + tag, '%.0f' % sp['both'])
    ws = rb['wedge_scan']
    cmd('WcHfOwFive', '%.0f' % ws['5']['HfO2_EOT1p5'])
    cmd('WcHfOwTwenty', '%.0f' % ws['20']['HfO2_EOT1p5'])
    W = np.array(rb['curves']['MoS2']['W'])
    wordw = {25: 'Twentyfive', 43: 'Fortythree', 75: 'Seventyfive',
             850: 'Wide'}
    for w, wd in wordw.items():
        cmd('IonMoS' + wd, '%.0f' % np.interp(
            w, W, np.array(rb['curves']['MoS2']['I'])))
    cmd('IonWSTwo', '%.0f' % np.interp(
        43, W, np.array(rb['curves']['WS2']['I'])))
    cmd('IonWSeTwo', '%.0f' % np.interp(
        43, W, np.array(rb['curves']['WSe2']['I'])))
    cmd('WcGentle', '%.0f' % rb['halo']['GENTLE']['Wc'])
    cmd('WcHIM', '%.0f' % rb['halo']['HIM']['Wc'])
    for tag, key in (('SiOThreeH', 'SiO2_300nm'), ('SiONinety', 'SiO2_90nm'),
                     ('SiOThirty', 'SiO2_30nm'), ('HfO', 'HfO2_EOT1p5')):
        cmd('Wc' + tag, '%.0f' % rb['cox'][key]['Wc'])
        cmd('dVT' + tag, '%.2f' % rb['cox'][key]['dVT_25nm'])
    for m in MATERIALS:
        cmd('Wc' + m.replace('2', 'Two'), '%.0f' % rb['Wc'][m])

    # ---- material constants
    for m in MATERIALS:
        mm = mats[m]
        tag = m.replace('2', 'Two')
        cmd('gE' + tag, '%.2f' % mm.gE)
        cmd('gA' + tag, '%.2f' % mm.gA)
        cmd('CA' + tag, '%.2f' % mm.C_A)
        cmd('Udef' + tag, '%.3f' % (mm.Udef if mm.Udef else float('nan')))
        cmd('CTwoD' + tag, '%.0f' % mm.C2D)
        cmd('wEdft' + tag, '%.0f' % mm.dft.get('wE_dft', float('nan')))
        cmd('wAdft' + tag, '%.0f' % mm.dft.get('wA_dft', float('nan')))
        cmd('gap' + tag, '%.2f' % mm.dft.get('gap_K', float('nan')))
        cmd('dgap' + tag, '%.0f' % abs(mm.dft.get('dgap_deps', float('nan'))))
        cmd('gLA' + tag, '%.2f' % mm.dft.get('gamma_LA', float('nan')))
        cmd('gAdft' + tag, '%.2f' % mm.dft.get('gamma_A', float('nan')))
        cmd('gLApred' + tag, '%.2f' %
            r['predictions'][m].get('gamma_LA', float('nan')))

    # ---- self-consistent device solver
    sc = r['self_consistent']
    v = sc['verify']
    cmd('ScPsiErr', _fmt_sci(max(v['surface_potential_rel_err'], 1e-17)))
    cmd('ScElecErr', _fmt_sci(max(v['electrostatics_rel_err'], 1e-17)))
    cmd('ScContErr', _fmt_sci(max(v['continuity_rel_err'], 1e-17)))
    cmd('ScSquareErr', '%.3f' % (100 * v['square_law_rel_err']))
    cmd('ScCq', '%.0f' % (1e6 * v['Cq_F_cm2']))
    cmd('ScCqCorr', '%.1f' % (100 * v['quantum_capacitance_correction']))
    cmd('ScRatio', '%.2f' % sc['compare']['ratio_mean'])
    # triode factor the square-law compact form leaves out
    cmd('TriodeFactor', '%.2f' % ((rb['Vov'] - 0.5) / rb['Vov']))
    cmd('ScRatioSpread', '%.2f' % sc['compare']['ratio_spread'])
    cmd('WcSelfCons', '%.1f' % sc['Wc_self_consistent'])
    cmd('WcCompact', '%.1f' % sc['Wc_compact'])
    cmd('WcModelDiff', '%.0f' % (100 * abs(sc['Wc_self_consistent']
                                           - sc['Wc_compact'])
                                 / sc['Wc_compact']))
    for key, tag in (('MoS2_50nm_HfO2', 'MoSFiftyHK'),
                     ('WS2_50nm_HfO2', 'WSTwo'),
                     ('WSe2_50nm_HfO2', 'WSeTwo')):
        d = sc['devices'][key]
        cmd('IonScIdeal' + tag, '%.0f' % d['I_ideal'])
        cmd('IonSc' + tag, '%.0f' % d['I_with_Rc'])
        cmd('MuSc' + tag, '%.0f' % d['mu'])
        cmd('IonMeas' + tag, '%.0f' % d['measured'])
        cmd('IonScFrac' + tag, '%.2f' % d['ratio'])
    for Lx, tag in (('50', 'Short'), ('300', 'Mid'), ('1000', 'Long')):
        cmd('WcL' + tag, '%.0f' % sc['Wc_vs_L'][Lx])
    cmd('WMeasHK', '%.0f' % sc['devices']['MoS2_50nm_HfO2']['W_nm'])
    cmd('LMeasHK', '%.0f' % sc['devices']['MoS2_50nm_HfO2']['Lch_nm'])
    cmd('IonScFracMin', '%.2f' % sc['ratio_min'])
    cmd('IonScFracMax', '%.2f' % sc['ratio_max'])
    # Deviation of the model from the two n-type measurements, with the
    # measured contact resistance included, which is what the text quotes.
    _rc = [d['I_with_Rc'] / d['measured'] for d in sc['devices'].values()
           if d['carrier'] == 'e']
    _dev = sorted(100.0 * abs(x - 1.0) for x in _rc)
    cmd('IonScDevMin', '%.0f' % _dev[0])
    cmd('IonScDevMax', '%.0f' % _dev[1])
    cmd('IonScSense', 'above' if min(_rc) > 1.0 else 'below')
    _w = sc['devices']['WSe2_50nm_HfO2']
    cmd('IonScFracWSe', '%.1f' % (_w['I_with_Rc'] / _w['measured']))
    # ---- energy-resolved Boltzmann mobility
    er = r['energy_resolved']
    cmd('ErVerify', _fmt_sci(max(er['verify']['rel_err'], 1e-17)))
    cmd('ErRatioMin', '%.2f' % er['ratio_min'])
    cmd('ErRatioMax', '%.2f' % er['ratio_max'])
    cmd('ErDropMax', '%.0f' % (100 * (1.0 - er['ratio_min'])))
    cmd('ErRiseMax', '%.0f' % (100 * (er['ratio_max'] - 1.0)))
    for name, tag in (('MoS2', 'MoS'), ('WS2', 'WS'), ('MoSe2', 'MoSe'),
                      ('WSe2', 'WSe')):
        cmd('ErRatio' + tag, '%.2f' % er['materials'][name]['ratio_25nm'])
        cmd('MuEr' + tag, '%.0f' % er['materials'][name]['mu_integral_25nm'])
    dos_shift = max(abs(a['mu_integral'] - a['mu_single']) / a['mu_single']
                    for a in er['dossena'])
    cmd('ErDossenaShift', _fmt_sci(max(dos_shift, 1e-17)))

    # ---- quantum transport
    q = r['quantum']
    v = q['verify']
    cmd('QtRectErr', _fmt_sci(v['rect_barrier_max_abs_err']))
    cmd('QtTunnelErr', '%.1f' % (100 * v['tunnelling_max_rel_err']))
    cmd('QtFlatErr', _fmt_sci(max(v['flat_max_dev_from_unity'], 1e-17)))
    cmd('QtBallErr', _fmt_sci(v['ballistic_rel_err']))
    cmd('QtDiffErr', _fmt_sci(v['diffusive_rel_err']))
    cmd('QtVinj', '%.1f' % (v['v_inj_m_s'] / 1e3))
    cmd('LamScreen', '%.2f' % q['screening_length_nm'])
    cmd('ThfO', '%.1f' % q['t_hfo2_nm'])
    bal = q['ballistic']
    for name, tag in (('MoS2', 'MoS'), ('WS2', 'WS'), ('MoSe2', 'MoSe'),
                      ('WSe2', 'WSe')):
        cmd('Lhalf' + tag, '%.1f' % bal[name]['L_half_nm'])
        cmd('Ballist' + tag, '%.1f' % (100 * bal[name]['T_at_300nm']))
        cmd('IBall' + tag, '%.0f' % bal[name]['I_ballistic'])
    cmd('LhalfMin', '%.1f' % min(b['L_half_nm'] for b in bal.values()))
    cmd('LhalfMax', '%.1f' % max(b['L_half_nm'] for b in bal.values()))
    cmd('BallistMin', '%.1f' % (100 * min(b['T_at_300nm']
                                          for b in bal.values())))
    cmd('BallistMax', '%.1f' % (100 * max(b['T_at_300nm']
                                          for b in bal.values())))
    cmd('BallistFiftyMin', '%.0f' % (100 * min(b['T_at_50nm']
                                               for b in bal.values())))
    cmd('BallistFiftyMax', '%.0f' % (100 * max(b['T_at_50nm']
                                               for b in bal.values())))
    rq = q['contact_quantum']
    cmd('RqMin', '%.0f' % min(x['R_quantum'] for x in rq.values()))
    cmd('RqMax', '%.0f' % max(x['R_quantum'] for x in rq.values()))
    cmd('RqMoS', '%.0f' % rq['MoS2']['R_quantum'])
    cmd('RqRatio', '%.0f' % (q['R_measured_ohm_um']
                             / min(x['R_quantum'] for x in rq.values())))
    cmd('PhiEqMoS', '%.2f' % q['phi_from_measured_Rc']['MoS2'])
    cmd('PhiEqWS', '%.2f' % q['phi_from_measured_Rc']['WS2'])
    bvr = q['barrier_vs_resistor']
    cmd('BvrFree', '%.0f' % bvr['I_transparent'])
    cmd('BvrResistor', '%.0f' % bvr['I_resistor'])
    cmd('BvrBarrier', '%.0f' % bvr['I_barrier'])
    cmd('BvrRatio', '%.1f' % bvr['ratio'])
    w = q['wse2']
    cmd('PhiWSe', '%.2f' % w['phi_b_eV'])
    cmd('IWSeFree', '%.0f' % w['I_transparent'])
    cmd('IWSePhi', '%.0f' % w['I_at_phi'])
    cmd('RcWSePhi', '%.0f' % w['R_contact_ohm_um'])

    cmd('RcMeas', '%.0f' % sc['Rc_measured_ohm_um'])

    # k-mesh cross-check of the two strain derivatives that matter
    kc = os.path.join(ROOT, 'dft', 'kcheck_summary.json')
    if os.path.exists(kc):
        with open(kc) as fh:
            k6 = json.load(fh)
        if 'gamma_LA' in k6:
            cmd('GammaLAkSix', '%.2f' % k6['gamma_LA'])
        if 'gamma_A_sym_6x6' in k6:
            cmd('GammaAkSix', '%.2f' % k6['gamma_A_sym_6x6'])
        if 'gamma_A_sym_4x4' in k6:
            cmd('GammaAkFour', '%.2f' % k6['gamma_A_sym_4x4'])

    cmd('NitSiO', _fmt_sci(transport.N_IT_SIO2))
    cmd('NitHfO', _fmt_sci(transport.N_IT_HFO2))
    cmd('Udefect', '%.3f' % mats['MoS2'].Udef)
    cmd('LambdaAOne', '%.1f' % abs(kv['dwA_dn']))
    cmd('HaloContrast', '%.0f' % 20.0)
    cmd('CAmeas', '%.2f' % D.MIGNUZZI['C_A'])
    cmd('CAmeasErr', '%.2f' % D.MIGNUZZI['C_A_err'])
    # on-current drop and critical-width drift across the inverted film range
    from .materials import all_materials as _am
    _m = mats['MoS2']
    _st = transport.STACK['HfO2_EOT1p5']
    _base = dict(halo_nm=D.HALO['GENTLE_nm'], sigma_line_cm=rb['sigma_line_cm'],
                 Cox=transport.COX['HfO2_EOT1p5'], Vov=rb['Vov'], Vds=1.0,
                 Lch_nm=300.0, n_it_cm2=_st['nit'], eps_env=_st['eps'])
    _lo, _hi = na['rows'][1]['nd'], na['rows'][2]['nd']
    _Ilo = float(transport.ribbon_current_density_uA_um(_m, 25.0, _lo, **_base))
    _Ihi = float(transport.ribbon_current_density_uA_um(_m, 25.0, _hi, **_base))
    cmd('IdropFilm', '%.1f' % (_Ilo / _Ihi))
    _Wlo = float(transport.critical_width(_m, _lo, **_base))
    _Whi = float(transport.critical_width(_m, _hi, **_base))
    cmd('WcDriftFilm', '%.0f' % (100.0 * abs(_Whi - _Wlo) / _Wlo))
    # threshold shift predicted at the widths where the multilayer ribbons of
    # Liu et al. were measured: the onset width and the narrowest device
    _liu = D.LIU2012
    cmd('dVTLiuOnset', '%.1f' % transport.threshold_shift(
        rb['sigma_line_cm'], _liu['W_onset_nm'], transport.COX['SiO2_300nm']))
    cmd('dVTLiuNarrow', '%.0f' % transport.threshold_shift(
        rb['sigma_line_cm'], _liu['narrowest_nm'], transport.COX['SiO2_300nm']))
    cmd('LiuOnset', '%.0f' % _liu['W_onset_nm'])
    cmd('LiuNarrow', '%.0f' % _liu['narrowest_nm'])
    cmd('LiuTotal', '%.0f' % _liu['dVT_total_V'])
    cmd('LiuThick', '%.0f' % min(_liu['thickness_nm']))

    out = os.path.join(ROOT, 'paper', 'numbers.tex')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        fh.write('%% generated by rapid/numbers.py -- do not edit\n')
        fh.write('\n'.join(L) + '\n')
    print('wrote', out, len(L), 'macros')


if __name__ == '__main__':
    main()
