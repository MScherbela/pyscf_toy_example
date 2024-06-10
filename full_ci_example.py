#%%
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import pyscf.fci.cistring
import numpy as np
import matplotlib.pyplot as plt

# Get LiH molecule
R = np.array([
    [0.0, 0.0, 0.0],
    [3.015, 0, 0]
    ]
    )
Z = [3, 1]
mol = pyscf.gto.Mole(
    atom=[(Z_, R_) for Z_, R_ in zip(Z, R)],
    basis="ccpvdz",
    unit="bohr")
mol.build()

# Get an approximate solution to the SE in the form of a single Slater determinant
# consisting of functions that each depend only on 1 coordinate each (=orbitals)
# This part is very cheap
print("Running Hartree-Fock...")
hf = pyscf.scf.RHF(mol)
E_hf = hf.kernel()
n_orbitals = hf.mo_coeff.shape[1]
print(f"No. of orbitals: {n_orbitals}")

# Generate the space of all possible Slater determinants you could build,
# and find the many-body eigenstates as a linear combinations of this space
# This part is extremely expensive, because the space scales factorially with the number of electrons
# and single particle basis functions
print("Running FCI...")
fci = pyscf.fci.FCI(mol, hf.mo_coeff)
n_states = 10
fci.nroots = n_states
energies_fci, fci_vecs = fci.kernel()
orb_occupations = pyscf.fci.cistring.gen_occslst(range(19), 2)

print(f"E_HF  = {E_hf:.4f} Ha")
print(f"E_FCI = {energies_fci[0]:.4f} Ha")


#%%
# As a simple example plot slices of the resulting orbitals and many-body-wavefunctions.
# Instead of slicing and plotting, we could in principle also run Markov Chain Monte Carlo (MCMC),
# to draw samples from this wavefunction (to get psi(x, t=0)) or draw samples at a later point in time
# by drawing samples from some time-evolved linear combination of states

n_el = sum(mol.nelec)
n_up = n_el // 2
n_plot = 500
x_plot = np.linspace(-2, 5, n_plot)
np.random.seed(12345)
r = np.random.normal(size=[1, n_el, 3])
r = np.tile(r, (n_plot, 1, 1))
r[:, 0, 0] = x_plot

# Evaluate basis functions
basis_funcs = np.stack([mol.eval_gto("GTOval", r[:, i, :]) for i in range(n_el)], axis=-2)

# Evaluate single-particle orbitals (linear comb. of basis functions)
molecular_orbitals = basis_funcs @ hf.mo_coeff

# Evaluate all (combinatorially many) slater determinants one could build from these orbitals
slater_matrices_up = np.stack([molecular_orbitals[:, :n_up, ind_orb] for ind_orb in orb_occupations], axis=-2)
slater_matrices_up = np.moveaxis(slater_matrices_up, -2, -3)
slater_matrices_dn = np.stack([molecular_orbitals[:, n_up:, ind_orb] for ind_orb in orb_occupations], axis=-2)
slater_matrices_dn = np.moveaxis(slater_matrices_dn, -2, -3)
dets_up = np.linalg.det(slater_matrices_up)
dets_dn = np.linalg.det(slater_matrices_dn)


plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

# Plot slices through the 3D orbitals
for ind_orbital in range(4):
    axes[0].plot(x_plot, molecular_orbitals[:, 0, ind_orbital], label=f"Orbital {ind_orbital}")
axes[0].legend()
axes[0].set_title(f"Molecular orbitals (lowest 4 out of {n_orbitals})")

# Plot slices through the 3*n_electron-dimensional wavefunctions
for ind_state, (E, ci_coeff) in enumerate(zip(energies_fci, fci_vecs)):
    psi = np.einsum("bu,bd,ud->b", dets_up, dets_dn, ci_coeff)
    axes[1].plot(x_plot, psi, label=f"$\Psi_{{{ind_state}}}$, E = {E:.4f}")
axes[1].legend()
axes[1].set_title(f"Many-electron eigenstates (lowest {n_states})")

for ax in axes:
    for R_ in R:
        ax.axvline(R_[0], color='gray', ls='--')
    ax.set_xlabel("x-coordinate of electron 0")

fig.suptitle("Full CI for Lithium Hydride (LiH, 4 electrons)", fontsize=16)
fig.savefig("slices_plots.png", dpi=200, bbox_inches="tight")


