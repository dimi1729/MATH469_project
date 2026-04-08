from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Parameters
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 64, 64  # Grid points
dx, dy = Lx / Nx, Ly / Ny
dt = 1e-4  # Time step
M_p = 1.0  # Mobility coefficient
num_steps = 1000  # Number of time steps

phi_p = np.random.rand(Ny, Nx) * 0.1 + 0.5


def free_energy_derivative(phi):
    """
    df/dphi_p for a double-well potential
    f = phi^2 * (1-phi)^2
    df/dphi = 2*phi*(1-phi)^2 - 2*phi^2*(1-phi)
    """
    return 2 * phi * (1 - phi) ** 2 - 2 * phi**2 * (1 - phi)


def compute_gradient(field, dx, dy):
    """
    Compute gradient using central differences
    Don't really get why but this is how you do a gradient apparently
    """
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)

    grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
    grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dy)

    # Periodic boundary conditions
    grad_x[:, 0] = (field[:, 1] - field[:, -1]) / (2 * dx)
    grad_x[:, -1] = (field[:, 0] - field[:, -2]) / (2 * dx)
    grad_y[0, :] = (field[1, :] - field[-1, :]) / (2 * dy)
    grad_y[-1, :] = (field[0, :] - field[-2, :]) / (2 * dy)

    return grad_x, grad_y


def compute_divergence(field_x, field_y, dx, dy):
    """Compute divergence using central differences"""
    div = np.zeros_like(field_x)

    div[:, 1:-1] += (field_x[:, 2:] - field_x[:, :-2]) / (2 * dx)
    div[1:-1, :] += (field_y[2:, :] - field_y[:-2, :]) / (2 * dy)

    # Periodic boundary conditions
    div[:, 0] += (field_x[:, 1] - field_x[:, -1]) / (2 * dx)
    div[:, -1] += (field_x[:, 0] - field_x[:, -2]) / (2 * dx)
    div[0, :] += (field_y[1, :] - field_y[-1, :]) / (2 * dy)
    div[-1, :] += (field_y[0, :] - field_y[-2, :]) / (2 * dy)

    return div


for step in range(num_steps):
    # Compute chemical potential: mu = df/d phi_p
    mu = free_energy_derivative(phi_p)

    # Compute gradient of chemical potential
    grad_mu_x, grad_mu_y = compute_gradient(mu, dx, dy)

    # Multiply by mobility
    flux_x = M_p * grad_mu_x
    flux_y = M_p * grad_mu_y

    # Compute divergence of flux
    dphi_dt = compute_divergence(flux_x, flux_y, dx, dy)

    phi_p = phi_p + dt * dphi_dt

    # Print progress
    if step % 100 == 0:
        print(f"Step {step}/{num_steps}, mean(phi_p) = {np.mean(phi_p):.4f}")

# Visualize final result
plt.figure(figsize=(8, 6))
plt.imshow(phi_p, cmap="viridis", origin="lower")
plt.colorbar(label="φ_p")
plt.title("Phase Field φ_p at final time")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("phase_field.png", dpi=300)
print("Simulation complete. Result saved to 'phase_field.png'")


def fef(phi_p: Callable, phi_r: Callable) -> float:
    """
    Function to calculate free energy functional from protein abundance
    and RNA abundance (phi_p and phi_r) in the condensate
    """
    ...
