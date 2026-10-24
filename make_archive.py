"""Assemble the public data-and-benchmark archive and the code repository.

Two artefacts are produced under dist/:

  zenodo/  the archive of record.  Raw first-principles output, the reduced
           constants, every digitised published benchmark used in the article,
           the machine-readable results file, the figure sources, and the
           metadata files Zenodo reads.
  github/  the code repository: the same source tree plus the stored
           first-principles output needed to reproduce every number without
           re-running density functional theory.

Both are then zipped.  Running this script twice gives byte-identical
directory contents.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, 'dist')

# The concept DOI always resolves to the newest version; the version DOI names
# one exact snapshot.  Zenodo lets a version DOI be reserved before the files
# are uploaded, so the deposit can carry its own DOI, and the manuscript cites
# that same version rather than a moving target.
ZENODO_CONCEPT_DOI = '10.5281/zenodo.21670315'
# The revised article (first revision) cites the concept DOI, so the files of
# every version carry it; v5, the version that accompanied the original
# submission, is 10.5281/zenodo.22053747.
ZENODO_DOI = ZENODO_CONCEPT_DOI
GITHUB_URL = ('https://github.com/Tanvir-Mahmud-Mahim/'
              'Width-scaling-in-monolayer-semiconductor-nanoribbon-transistors')
ARTICLE = ('Two Raman peaks and transistor thresholds quantify the fixed '
           'negative charge at etched edges of monolayer MoS2')
TITLE = 'Data, benchmarks and code for: ' + ARTICLE
AUTHORS = [
    dict(name='Mahim, Tanvir M.',
         affiliation='Department of Electrical and Electronic Engineering, '
                     'BRAC University, Dhaka 1212, Bangladesh'),
    dict(name='Rahman, M. Mosaddequr',
         affiliation='Department of Electrical and Electronic Engineering, '
                     'BRAC University, Dhaka 1212, Bangladesh'),
]


def _fresh(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _copytree(src, dst, skip=('__pycache__',)):
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in sorted(files):
            if f.endswith('.pyc'):
                continue
            if f == 'run.log':      # console duplicate of the .abo record
                continue
            rel = os.path.relpath(os.path.join(root, f), src)
            _copy(os.path.join(root, f), os.path.join(dst, rel))


# ---------------------------------------------------------------------------
# published benchmarks, exported as flat CSV so they can be read without Python
# ---------------------------------------------------------------------------
def export_benchmarks(out_dir):
    from rapid import datasets as D
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def dump(name, header, rows, note):
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['# ' + note])
            w.writerow(header)
            w.writerows(rows)
        written.append((name, note))

    dump('krayev2026_tip_enhanced_raman_shifts.csv',
         ['location', 'delta_omega_A1p_cm-1', 'delta_omega_2LA_cm-1'],
         [['ribbon edge', D.KRAYEV['edge_dwA'], D.KRAYEV['edge_dw2LA']],
          ['interior inhomogeneity', D.KRAYEV['spot_dwA'],
           D.KRAYEV['spot_dw2LA']]],
         'Tip-enhanced Raman shifts (785 nm excitation) at the edges and at '
         'one interior inhomogeneity of one CVD-grown monolayer MoS2 '
         'nanoribbon patterned by electron-beam lithography and XeF2 etching, '
         'vacuum annealed and imaged on gold, read from Krayev et al. (2026), '
         'Appl. Phys. Lett. 128, 203102. No uncertainty is stated for the '
         'shifts. Inhomogeneity %.0f-%.0f nm.'
         % tuple(D.KRAYEV['inhom_size_nm']))

    dump('pena2026_threshold_voltage_vs_width_digitised.csv',
         ['width_nm', 'channel_length_um', 'V_T_cc_V'],
         [list(r) for r in D.PENA_VT],
         'Threshold voltage V_T,cc (constant current 100 nA/um, forward '
         'sweep) of monolayer MoS2 transistors made by e-beam lithography '
         'and XeF2 etching on ~5.5 nm HfO2, DIGITISED by the authors of this '
         'archive from Extended Data Fig. 4 of Pena et al., Nat. '
         'Nanotechnol. 21, 803 (2026), doi:10.1038/s41565-026-02161-w. '
         'Reading uncertainty about 0.01-0.02 V.')

    dump('threshold_shift_other_device_studies.csv',
         ['study', 'etch', 'gate', 'quantity', 'value'],
         [['Lan et al., Nat. Commun. 17, 7386 (2026)', D.LAN2026['etch'],
           '3 nm HfO2, kappa ~10', 'dV_T,cc 540 nm -> 35 nm (V)',
           D.LAN2026['dVT_V']],
          ['Kotekar-Patil et al., Appl. Phys. Lett. 114, 013508 (2019)',
           D.KOTEKAR2019['etch'], '300 nm SiO2',
           'V_th of 50 nm ribbons (V)',
           ' '.join('%g' % v for v in D.KOTEKAR2019['Vth_ribbon_V'])],
          ['Kotekar-Patil et al., Appl. Phys. Lett. 114, 013508 (2019)',
           'unetched', '300 nm SiO2', 'V_th range of unetched devices (V)',
           '%g to %g' % D.KOTEKAR2019['Vth_unetched_V']],
          ['Liu, Gu and Ye, IEEE Electron Device Lett. 33, 1273 (2012)',
           D.LIU2012['etch'], '300 nm SiO2, 6-11 nm thick (multilayer) bodies',
           'width below which V_th first shifts positive (nm)',
           '%g' % D.LIU2012['W_onset_nm']],
          ['Liu, Gu and Ye, IEEE Electron Device Lett. 33, 1273 (2012)',
           D.LIU2012['etch'], '300 nm SiO2, 6 nm thick body',
           'total V_th shift, wide ribbons -> %g nm (V)'
           % D.LIU2012['narrowest_nm'],
           '%g' % D.LIU2012['dVT_total_V']]],
         'Threshold-voltage values quoted from the text of the cited works.')

    dump('nattoo2025_disorder_activated_raman_and_mobility.csv',
         ['film', 'R_shoulder', 'R_shoulder_err', 'R_LA', 'R_LA_err',
          'mobility_cm2_V-1_s-1'],
         [[e['label'], e['R_sh'], e.get('R_sh_e', ''), e['R_LA'],
           e.get('R_LA_e', ''), e['mu']] for e in D.NATTOO],
         'Disorder-activated Raman intensity ratios and measured mobilities '
         'of atomic-layer-deposited and sputtered MoS2 films, read from '
         'Nattoo et al. (2025).')

    rows = [[n, m, car, w, L, stack, meas]
            for n, m, car, w, L, stack, meas in D.PENA['devices']]
    rows += [['%s_%.0fnm_SiO2_backgated' % (m, w), m, 'e', w, L,
              'SiO2_96nm', meas] for m, w, L, meas in D.PENA['backgated']]
    dump('pena2026_nanoribbon_on_currents.csv',
         ['device', 'material', 'carrier', 'width_nm', 'channel_length_nm',
          'gate_stack', 'on_current_uA_per_um'],
         rows,
         'On-current per unit width of monolayer nanoribbon transistors at '
         'Vds = 1 V, read from Pena et al. (2026). Only the high-kappa '
         'devices are modelled in the article: the 96 nm SiO2 back-gated '
         'devices are driven at a gate overdrive that is not reported device '
         'by device.')

    mb = D.MICHAIL_BIAX
    rows = []
    for m, d in mb.items():
        rows.append([m, d.get('gE', ''), d.get('gE_e', ''), d.get('gA', ''),
                     d.get('gA_e', ''), d.get('gLA', ''), d.get('gLA_e', '')])
    dump('michail2024_measured_gruneisen_biaxial.csv',
         ['material', 'gamma_Ep', 'gamma_Ep_err', 'gamma_A1p',
          'gamma_A1p_err', 'gamma_LA', 'gamma_LA_err'],
         rows,
         'Mode Grueneisen parameters measured under biaxial strain, read from '
         'Michail et al. (2024), in the convention gamma = -(1/2 w) dw/deps. '
         'That work covers both optical modes of MoS2 and the A1p mode and '
         'the 2LA overtone of WSe2. WSe2 values are means over the five devices of that work. Blank entries were not measured.')

    dump('gauge_factors_exciton.csv',
         ['material', 'gauge_meV_per_percent', 'gauge_err'],
         [[m, v[0], v[1]] for m, v in D.GAUGE.items()],
         'Measured neutral A-exciton biaxial gauge factors (substrate '
         'deformation), read from Henriquez-Guerra et al. (2023), ACS Appl. '
         'Mater. Interfaces 15, 57369, Table 1, used as the independent check '
         'on the computed gap deformation potentials.')

    dump('halo_widths.csv', ['process', 'halo_width_nm'],
         [['resist-masked XeF2 etch (modelling choice, not measured)', D.HALO['GENTLE_nm']],
          ['XeF2-assisted helium-ion-beam etching (measured)', D.HALO['HIM_nm']]],
         'Process-induced damage halo widths. Only the helium-ion value is '
         'measured; the gentle-etch value is a modelling choice that is '
         'scanned over two orders of magnitude in the article.')

    dump('mignuzzi2015_calibration.csv', ['quantity', 'value', 'err', 'units'],
         [['C_A', D.MIGNUZZI['C_A'], D.MIGNUZZI['C_A_err'], 'nm^2']],
         'Disorder-activated Raman calibration constant of monolayer MoS2 at '
         '532 nm, I(LA)/I(A1p) = C_A n_d, read from Mignuzzi et al. (2015), '
         'Phys. Rev. B 91, 195411.')

    dump('yang2026_lfm_defect_densities.csv',
         ['film', 'n_d_cm-2', 'err_upper_cm-2'],
         [[k, v[0], v[1]] for k, v in D.YANG_DEFECTS.items()],
         'Surface defect densities counted by lateral force microscopy, read '
         'from Yang et al. (2026), ACS Appl. Mater. Interfaces 18, 10161. '
         'SS-CVD is solid-source CVD. The WS2 value sets the point-defect '
         'density of the reference film in the article.')
    return written


# ---------------------------------------------------------------------------
def build_zenodo():
    out = _fresh(os.path.join(DIST, 'zenodo'))

    # raw and reduced first-principles output
    for f in ('stageA.json', 'conv.json', 'kcheck.json', 'dft_summary.json',
              'kcheck_summary.json'):
        src = os.path.join(HERE, 'dft', f)
        if os.path.exists(src):
            _copy(src, os.path.join(out, 'first_principles', f))
    for f in ('soc_MoS2.json', 'soc_WSe2.json', 'soc_summary.json'):
        src = os.path.join(HERE, 'dft', 'soc', f)
        if os.path.exists(src):
            _copy(src, os.path.join(out, 'first_principles', 'soc', f))

    # every number quoted in the article
    _copy(os.path.join(HERE, 'results.json'),
          os.path.join(out, 'results', 'results.json'))
    _copy(os.path.join(HERE, 'paper', 'numbers.tex'),
          os.path.join(out, 'results', 'numbers.tex'))
    _copy(os.path.join(HERE, 'map_recovery.npz'),
          os.path.join(out, 'results', 'map_recovery.npz'))
    for f in ('exact_cache.json', 'fringe_cache.json'):
        _copy(os.path.join(HERE, f), os.path.join(out, 'results', f))
        _copy(os.path.join(HERE, f), os.path.join(out, 'code', f))

    # digitised published benchmarks
    bench = export_benchmarks(os.path.join(out, 'benchmarks'))

    # figures as published
    for f in sorted(os.listdir(os.path.join(HERE, 'figs'))):
        if f.endswith('.pdf'):
            _copy(os.path.join(HERE, 'figs', f),
                  os.path.join(out, 'figures', f))

    # the code, so the archive is self-contained
    for d in ('rapid', 'dft'):
        _copytree(os.path.join(HERE, d), os.path.join(out, 'code', d))
    _copy(os.path.join(HERE, 'finalize.sh'),
          os.path.join(out, 'code', 'finalize.sh'))

    _copy(os.path.join(HERE, 'LICENSE'), os.path.join(out, 'LICENSE'))
    _write_zenodo_metadata(out, bench)
    return out


def _write_zenodo_metadata(out, bench):
    meta = {
        'title': TITLE,
        'upload_type': 'dataset',
        'creators': AUTHORS,
        'description': (
            '<p>Data, benchmarks and code accompanying the article '
            f'"{ARTICLE}".</p>'
            '<p><b>first_principles/</b> raw PySCF output: the chalcogen-height '
            'relaxation, the five-point biaxial strain sweep with band '
            'energies at the high-symmetry points, the symmetric frozen-phonon '
            'energies for the E′ mode, the relaxed-geometry z scans for '
            'the A₁′ mode, the cutoff / k-mesh / vacuum convergence '
            'study, and the 6x6x1 k-mesh cross-check, together with the '
            'reduced constants used by the models, and the fully '
            'relativistic ABINIT spin-orbit cross-check (scalar and spinor '
            'runs of the same fully relativistic pseudopotentials) with its '
            'reduced spin-orbit deltas.</p>'
            '<p><b>benchmarks/</b> every published measurement used in the '
            'article, as flat CSV with the source stated in the header: '
            'tip-enhanced Raman shifts at nanoribbon edges, disorder-activated '
            'Raman ratios and mobilities of deposited films, nanoribbon '
            'on-currents, measured mode Grüneisen parameters, A-exciton '
            'gauge factors, damage-halo widths, the Raman calibration '
            'constant, and the threshold voltages of nanoribbon transistors '
            'against width (digitised) with the threshold values quoted by three '
            'further device studies, and the defect densities counted by '
            'lateral force microscopy that set the reference film.</p>'
            '<p><b>results/</b> the machine-readable results file from which '
            'every number, table and figure in the article is generated, and '
            'the generated LaTeX macro file, and the cached intermediate '
            'results of the self-consistent electrostatic inversion of the '
            'threshold data (exact_cache.json) and of the line-capacitance '
            'correction (fringe_cache.json).</p>'
            '<p><b>figures/</b> the figures as published.</p>'
            '<p><b>code/</b> the complete analysis code. The development '
            'repository is at ' + GITHUB_URL + '</p>'),
        'access_right': 'open',
        'license': 'Apache-2.0',
        'keywords': ['2D materials', 'transition metal dichalcogenides',
                     'monolayer MoS2', 'nanoribbon transistors',
                     'Raman spectroscopy', 'Grüneisen parameters',
                     'density functional theory', 'edge charge',
                     'width scaling'],
        'related_identifiers': [
            dict(identifier=GITHUB_URL, relation='isSupplementTo',
                 scheme='url'),
        ],
    }
    with open(os.path.join(out, '.zenodo.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    cff = f"""cff-version: 1.2.0
message: "If you use this dataset, please cite it as below."
title: "{TITLE}"
type: dataset
authors:
  - family-names: Mahim
    given-names: "Tanvir M."
    affiliation: "BRAC University"
  - family-names: Rahman
    given-names: "M. Mosaddequr"
    affiliation: "BRAC University"
license: Apache-2.0
doi: {ZENODO_DOI}
repository-code: "{GITHUB_URL}"
"""
    with open(os.path.join(out, 'CITATION.cff'), 'w') as fh:
        fh.write(cff)

    lines = [
        '# Data and benchmarks', '',
        TITLE, '',
        f'DOI: {ZENODO_DOI}', f'Code repository: {GITHUB_URL}', '',
        'This archive contains everything needed to check or rebuild every',
        'number in the article without re-running density functional theory.',
        '',
        '## first_principles/', '',
        '| file | contents |', '| --- | --- |',
        '| `stageA.json` | chalcogen-height relaxation, biaxial strain sweep '
        'with band energies at G, K, M, Lambda and Q, symmetric frozen-phonon '
        'energies, relaxed-geometry z scans, for all four monolayers |',
        '| `conv.json` | convergence against density cutoff, k-mesh and vacuum '
        'thickness |',
        '| `kcheck.json` | the MoS2 strain sweep and A1′ frozen phonons '
        'repeated on a 6x6x1 k-mesh |',
        '| `dft_summary.json` | the reduced constants used by the models |',
        '| `kcheck_summary.json` | the same reduction applied to the 6x6x1 run |',
        '| `soc/soc_MoS2.json`, `soc/soc_WSe2.json` | ABINIT total energies '
        'and band edges of the spin-orbit cross-check, scalar and spinor |',
        '| `soc/soc_summary.json` | the reduced spin-orbit deltas of the '
        'strain derivatives |',
        '',
        'Energies are in eV, lengths in angstrom, frequencies in cm^-1.',
        '',
        '## benchmarks/', '',
        'Every published measurement used in the article, one CSV per source.',
        'The first line of each file states the source and any caveat.', '',
        '| file | source |', '| --- | --- |',
    ]
    for name, note in bench:
        lines.append('| `%s` | %s |' % (name, note.split('.')[0] + '.'))
    lines += [
        '',
        '## results/', '',
        '`results.json` holds every computed quantity quoted in the article,',
        'keyed by the numerical experiment that produced it. `numbers.tex` is',
        'the generated LaTeX macro file, so each macro in the manuscript can be',
        'traced to a value here. `map_recovery.npz` holds the true and',
        'recovered fields of the synthetic hyperspectral recovery test.',
        '`exact_cache.json` holds the intermediate results of stage A18, the',
        'self-consistent electrostatic inversion of the threshold data, and',
        '`fringe_cache.json` the line-capacitance factors of the gate stacks;',
        'delete them to recompute.', '',
        '## figures/', '',
        'The figures as published, in vector PDF.', '',
        '## code/', '',
        'The complete analysis code. See the repository for a full description',
        'of the layout and of how to reproduce the article end to end.', '',
        '## License', '',
        'Apache License 2.0.  See LICENSE in the code repository.',
    ]
    with open(os.path.join(out, 'README.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
def build_github():
    out = _fresh(os.path.join(DIST, 'github'))
    # the raw ABINIT run outputs stay in the Zenodo archive of record; the
    # repository carries the driver, the pseudopotentials and the reduced
    # results, which is everything needed to re-run the stage
    for d in ('rapid', 'dft', 'figs'):
        _copytree(os.path.join(HERE, d), os.path.join(out, d),
                  skip=('__pycache__', 'runs'))
    for f in ('finalize.sh', 'results.json', 'map_recovery.npz',
              'exact_cache.json', 'fringe_cache.json'):
        _copy(os.path.join(HERE, f), os.path.join(out, f))
    # The manuscript sources and the built PDFs are deliberately left out of
    # the public repository; the archive holds code, data and figures only.
    export_benchmarks(os.path.join(out, 'benchmarks'))
    _copy(os.path.join(HERE, 'README.md'), os.path.join(out, 'README.md'))
    _copy(os.path.join(HERE, 'LICENSE'), os.path.join(out, 'LICENSE'))
    _copy(os.path.join(HERE, 'dist', 'zenodo', 'CITATION.cff'),
          os.path.join(out, 'CITATION.cff'))
    with open(os.path.join(out, '.gitignore'), 'w') as fh:
        fh.write('__pycache__/\n*.pyc\n*.aux\n*.log\n*.out\n*.blg\n*.bbl\n'
                 '*Notes.bib\ndist/\n')
    return out


def main():
    os.makedirs(DIST, exist_ok=True)
    z = build_zenodo()
    g = build_github()
    for d, name in ((z, 'zenodo_data_and_benchmarks'), (g, 'github_repo')):
        base = os.path.join(DIST, name)
        if os.path.exists(base + '.zip'):
            os.remove(base + '.zip')
        subprocess.run(['zip', '-qr', base + '.zip', os.path.basename(d)],
                       cwd=DIST, check=True)
        print('wrote', base + '.zip')


if __name__ == '__main__':
    main()
