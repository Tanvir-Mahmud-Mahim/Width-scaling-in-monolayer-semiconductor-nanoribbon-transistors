# Width scaling in monolayer semiconductor nanoribbon transistors

Complete code, data and manuscript sources for

> **Two Raman phonons measure the fixed edge charge that limits width scaling in monolayer semiconductor nanoribbon transistors**
> Tanvir M. Mahim and M. Mosaddequr Rahman, Department of Electrical and Electronic Engineering, BRAC University, Dhaka 1212, Bangladesh.

Data and benchmarks of record: [10.5281/zenodo.21687328](https://doi.org/10.5281/zenodo.21687328)

---

## What the work does

Monolayer transition metal dichalcogenide nanoribbon transistors now operate at
channel widths of 25 nm, but what physically stops the width from shrinking
further has not been settled. Three candidates compete: geometric edge
roughness, process damage extending beyond the patterned line, and fixed charge
trapped at the etched edge. Distinguishing them needs a measurement of the edge
charge density, and none existed.

A phonon frequency responds to strain and to carrier density at once, so a
single Raman mode cannot separate the two. This work uses a pair chosen so that
the two effects decouple:

* the first-order **A₁′** mode, whose out-of-plane symmetry makes it couple
  selectively to the K-valley electron density, and
* the disorder-activated **2LA(M)** acoustic overtone, which is strongly strain
  sensitive and, to the accuracy of any published measurement, charge
  insensitive.

First-principles calculations on MoS₂, WS₂, MoSe₂ and WSe₂ give the two strain
lever arms. The acoustic overtone turns out to be about **4.1 times** more
strain sensitive than A₁′, which is what makes the pair well conditioned.
Applying the pair to published tip-enhanced Raman maps of patterned MoS₂
nanoribbons gives a local excess of **2.3 × 10¹² ± 7.6 × 10¹¹ cm⁻²** of band
electrons at the edge with strain below 0.03 %, while an interior inhomogeneity
in the same data is pure 0.134 % tension with no significant charge.

Carried into an electrostatic and transport model, charge of that size explains
a decade-old depletion-to-enhancement transition observed when nanoribbons are
trimmed below about 200 nm on thick oxide, and is negligible on a thin
high-κ gate. There, the patterning damage halo takes over and sets a critical
width of about **17 nm**. As-grown defect density mainly rescales the current
rather than the width limit.

---

## Repository layout

```
dft/            first-principles drivers and their stored output
  tmd_common.py            monolayer cell construction and the SCF helper
  run_stageA.py            chalcogen relaxation, biaxial strain sweep,
                           symmetric frozen phonons, band energies
  run_stageB2.py           negative-displacement frozen phonons
  run_stageC.py            chalcogen-vacancy supercells
  run_stageD.py            relaxed-geometry z scans for the A1' mode
  run_conv.py              cutoff, k-mesh and vacuum convergence study
  run_kcheck.py            the MoS2 sweep repeated on a 6x6x1 k-mesh
  postprocess.py           reduces raw output to the constants used downstream
  postprocess_kcheck.py    the same reduction for the k-mesh cross-check
  *.json                   the stored output, so nothing has to be re-run

rapid/          the analysis package
  materials.py    parameter table, with a provenance tag on every entry
  datasets.py     every published measurement used, with its source
  spectra.py      the forward model and its analytic Jacobian
  adjoint.py      adjoint-state inversion and its three verification tests
  transport.py    scattering channels, energy-resolved Boltzmann mobility,
                  nanoribbon device layer
  device.py       self-consistent Poisson and drift-diffusion device solve
  quantum.py      NEGF transmission, Schottky contact, Landauer current
  analysis.py     every numerical experiment reported in the article
  figures.py      main-text figures
  device_fig.py   the three-dimensional device schematic
  supplement.py   supplementary figures and tables
  numbers.py      exports every quoted number as a LaTeX macro

benchmarks/     published measurements as flat CSV, one file per source
figs/           the figures as published, vector PDF
finalize.sh     rebuilds every number and figure end to end
results.json    every computed quantity quoted in the article
```

The manuscript sources and the built PDFs are not part of this repository. The
article itself is the published record; everything needed to reproduce the
numbers and the figures behind it is here.

---

## Reproducing the article

Nothing has to be re-run: the first-principles output is stored in `dft/`.

```bash
pip install numpy scipy matplotlib devsim
bash finalize.sh
```

`devsim` needs a BLAS/LAPACK library; on Debian or Ubuntu
`apt install libopenblas-dev liblapack-dev` is enough.

`finalize.sh` reduces the stored density functional theory output, runs every
numerical experiment, regenerates all eight figures and three tables, and
exports every quoted number as a LaTeX macro. If a `paper/` directory with the
manuscript sources is present it also typesets the PDFs and reports the page
count; without one it stops after the macros, which is the normal case here.

To run the individual stages:

```bash
cd dft && python postprocess.py          # raw output -> constants
cd .. && python -m rapid.analysis        # -> results.json
python -m rapid.figures                  # main-text figures
python -m rapid.supplement               # supplementary figures and tables
python -m rapid.numbers                  # -> numbers.tex, one macro per quantity
```

Re-running the density functional theory needs only PySCF and takes a few hours
on two cores:

```bash
pip install pyscf ase
cd dft
python run_stageA.py && python run_stageB2.py && python run_stageD.py
python run_stageC.py && python run_conv.py && python run_kcheck.py
python postprocess.py && python postprocess_kcheck.py
```

---

## Method summary

**First principles.** Periodic PBE calculations in PySCF with
Goedecker-Teter-Hutter pseudopotentials and a DZVP-MOLOPT-SR basis, a 60 Ha
density cutoff, a 4 × 4 × 1 k-mesh and a 15 Å c axis. The E′ force
constant comes from a symmetric ±0.05 Å frozen displacement, which cancels
the residual linear force present at finite strain. The A₁′ normal coordinate
is the chalcogen half-thickness itself, so a three-point parabolic scan of that
one coordinate gives both the relaxed geometry and the force constant at that
geometry. The acoustic Grüneisen parameter follows from the strain dependence
of the elastic constant through ω_LA(M) ∝ √C₁₁. A 6 × 6 × 1 cross-check
confirms that the strain derivatives move by only a few per cent.

**Separation.** The two Raman shifts form a 2 × 2 system whose lower-right
entry is zero, which is the physical content of the method. Because the
overtone is measured not to shift at the edge, the strain is pinned near zero
for any non-zero acoustic Grüneisen parameter, so the edge result does not
depend on the calculated value. The assumption that the overtone is charge
insensitive is bounded rather than asserted.

**Adjoint inversion.** The same forward model extends to a full hyperspectral
image, where defect density, strain and carrier density are recovered jointly.
The forward operator is an explicit analytic chain, so its linearisation is
available in closed form and the gradient is assembled in one adjoint sweep;
the Gauss-Newton system is solved matrix free by conjugate gradients. Three
checks run automatically: the analytic Jacobian against a central finite
difference, the dot-product identity between the separately implemented forward
and adjoint operators, and the assembled map gradient against a directional
finite difference of the full objective.

**Transport.** Phonon, neutral point-defect and screened interface-charge
scattering combined by Matthiessen's rule, plus diffuse edge scattering and a
damage halo treated as two parallel conductors, closed with the measured
monolayer MoS₂ saturation velocity. The mobility comes from the full Boltzmann
integral, carrying a relaxation time at every carrier energy, which matters
because the screened Coulomb rate follows the carrier wavevector and the
diffuse edge scattering rate follows the carrier speed. Every current the
article compares with measurement then comes from a self-consistent solve of
the channel: the surface band bending and the electron quasi-Fermi potential
are carried as coupled fields, so the quantum capacitance and the fall of the
channel charge towards the drain enter without approximation. The coupled
system is solved by Newton's method in [DEVSIM](https://devsim.org), and four
checks run with the code: the band bending against an independent bisection
solve of the gate balance, the residual of the gate balance itself,
source-to-drain current continuity, and the long-channel square law once the
quantum capacitance is placed in series with the oxide.

**Quantum transport.** `rapid/quantum.py` removes what a semiclassical solve
cannot describe. The current is written as a two-dimensional Landauer
integral; the channel enters through a transmission λ/(λ+L) built from the
same energy-resolved relaxation times, and the contact through the
transmission of the metal-to-monolayer Schottky barrier, obtained from a
non-equilibrium Green's function solution of the effective-mass equation, so
thermionic emission and tunnelling are one calculation. Four checks run with
the code: the transmission of a rectangular barrier against the closed-form
result, a flat potential transmitting to one, the non-degenerate ballistic
limit against q n_s √(k_BT/2πm\*), and the long-channel limit against the
Boltzmann conductivity. That last identity is what fixes λ = (π/2)vτ, with no
freedom left. The 300 nm channels of the measured devices have a transmission
of 1 to 2 %, so they are firmly diffusive; a barrier-free contact to a 2D
channel cannot go below 28 to 39 Ω·µm; and the p-type WSe₂ current corresponds
to a barrier of 0.26 eV. The compact expression is retained only to sweep the design map and
the critical-width scans; it is high by a nearly constant factor 1.56, and the
critical width, being a ratio, differs between the two models by 7 %. Exactly two constants are calibrated
rather than computed or measured, both once and then held fixed: the neutral
point-defect potential and the interface trap density of each gate stack.
`rapid/materials.py` carries a provenance tag on every entry, reproduced as
Table S3 of the supplement.

---

## Data

The archive of record, including the raw first-principles output and every
digitised published benchmark, is deposited at Zenodo:
**[10.5281/zenodo.21687328](https://doi.org/10.5281/zenodo.21687328)**.

`benchmarks/` in this repository holds the same benchmark files as flat CSV,
each with its source stated in the header line.

---

## Citing

If you use this code or these data, please cite the article and the Zenodo
deposit. `CITATION.cff` holds machine-readable metadata.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
