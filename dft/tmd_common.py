"""Common utilities: build monolayer TMD cells for PySCF periodic DFT."""
import numpy as np
from pyscf.pbc import gto, dft

# PBE-relaxed in-plane lattice constants (A) used as starting points.
A0 = {'MoS2': 3.184, 'WS2': 3.183, 'MoSe2': 3.319, 'WSe2': 3.322}
# starting chalcogen half-thickness (A)
Z0 = {'MoS2': 1.564, 'WS2': 1.571, 'MoSe2': 1.675, 'WSe2': 1.680}
MET = {'MoS2': 'Mo', 'WS2': 'W', 'MoSe2': 'Mo', 'WSe2': 'W'}
CHA = {'MoS2': 'S', 'WS2': 'S', 'MoSe2': 'Se', 'WSe2': 'Se'}
MASS = {'Mo': 95.95, 'W': 183.84, 'S': 32.06, 'Se': 78.97}

VAC = 15.0

def build_cell(mat, strain=0.0, dz=0.0, a=None, z=None, basis='gth-dzvp-molopt-sr',
               ke_cutoff=100.0, disp=None, sc=(1, 1), remove=None, verbose=3):
    """Monolayer 1H-MX2. strain = biaxial in-plane strain (fraction).
    disp: dict of atom index -> 3-vector Cartesian displacement (A)."""
    a = (A0[mat] if a is None else a) * (1.0 + strain)
    z = (Z0[mat] if z is None else z) + dz
    c = VAC
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    M, X = MET[mat], CHA[mat]
    frac = [(M, [1 / 3., 2 / 3., 0.5]),
            (X, [2 / 3., 1 / 3., 0.5 + z / c]),
            (X, [2 / 3., 1 / 3., 0.5 - z / c])]
    atoms = []
    nx, ny = sc
    for ix in range(nx):
        for iy in range(ny):
            for s, f in frac:
                fr = np.array([(f[0] + ix) / nx, (f[1] + iy) / ny, f[2]])
                atoms.append([s, np.dot(fr, np.array([lat[0] * nx, lat[1] * ny, lat[2]]))])
    if nx * ny > 1:
        lat = np.array([lat[0] * nx, lat[1] * ny, lat[2]])
    if disp:
        for i, d in disp.items():
            atoms[i][1] = atoms[i][1] + np.asarray(d)
    if remove:
        atoms = [a for i, a in enumerate(atoms) if i not in set(remove)]
    cell = gto.Cell()
    cell.a = lat
    cell.atom = [(s, tuple(r)) for s, r in atoms]
    cell.basis = basis
    cell.pseudo = 'gth-pbe'
    cell.ke_cutoff = ke_cutoff
    cell.dimension = 3
    cell.verbose = verbose
    cell.build()
    return cell

def run_scf(cell, nk=4, conv_tol=1e-7, dm0=None, max_cycle=60):
    kpts = cell.make_kpts([nk, nk, 1])
    mf = dft.KRKS(cell, kpts)
    mf.xc = 'pbe'
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    e = mf.kernel(dm0=dm0)
    return mf, e
