import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv
from numpy import exp, sin, cos, pi

# Parameters
k = 0.1 + 0.1j
a = 1.0
num_points = 200
r_max = 3.0
l_max = 10  # Truncation of the series

# Create a grid in polar coordinates
r = np.linspace(0.01, r_max, num_points)  # Avoid r=0
theta = np.linspace(0, pi, num_points)
R, Theta = np.meshgrid(r, theta)

# Convert to Cartesian coordinates for plotting
X = R * np.sin(Theta)
Z = R * np.cos(Theta)

def incident_field(r, theta, k):
    z = r * np.cos(theta)
    return exp(1j * k * z)

def scattered_field(R, Theta, k, a, l_max):
    psi_sc = np.zeros_like(R, dtype=complex)
    for l in range(l_max + 1):
        ka = k * a
        kr_flat = k * R.flatten()  # Flatten the radius grid

        jl_ka = np.sqrt(pi / (2 * ka)) * jv(l + 0.5, ka)
        yl_ka = np.sqrt(pi / (2 * ka)) * yv(l + 0.5, ka)

        # Hankel function of the first kind: h_l^(1) = j_l + i * yl_ka
        hl1_ka_complex = jl_ka + 1j * yl_ka

        if np.abs(hl1_ka_complex) < 1e-9:  # Avoid division by zero
            al = 0
        else:
            al = - (2 * l + 1) * (1j)**l * jl_ka / hl1_ka_complex

        jl_kr_flat = np.sqrt(pi / (2 * kr_flat)) * jv(l + 0.5, kr_flat)
        yl_kr_flat = np.sqrt(pi / (2 * kr_flat)) * yv(l + 0.5, kr_flat)
        hl1_kr_complex_flat = jl_kr_flat + 1j * yl_kr_flat
        hl1_kr_complex_grid = hl1_kr_complex_flat.reshape(R.shape) # Reshape to the grid

        legendre_val = np.polynomial.legendre.legval(np.cos(Theta), np.polynomial.legendre.Legendre.basis(l).coef[::-1])
        psi_sc += al * hl1_kr_complex_grid * legendre_val

    return psi_sc
# Calculate the incident and scattered fields on the grid
psi_inc = incident_field(R, Theta, k)
psi_sc = scattered_field(R, Theta, k, a, l_max)
psi_total = psi_inc + psi_sc

# Apply the boundary condition (approximately, as the grid might not hit r=a exactly)
boundary_mask = np.isclose(R, a, atol=r_max / num_points)
psi_total[boundary_mask] = 0  # Enforce Dirichlet boundary condition

# Visualize the real part of the total field
plt.figure(figsize=(8, 8))
plt.imshow(psi_total.real, extent=[-r_max, r_max, -r_max, r_max], origin='lower', cmap='coolwarm')
plt.colorbar(label='Real part of Total Field')
circle = plt.Circle((0, 0), a, edgecolor='black', facecolor='none', linestyle='--')
plt.gca().add_patch(circle)
plt.title(f'Scattering from a Sphere (k={k:.2f}), Real Part of Total Field')
plt.xlabel('x')
plt.ylabel('z')
plt.xlim([-r_max, r_max])
plt.ylim([-r_max, r_max])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()

# Visualize the magnitude of the total field
plt.figure(figsize=(8, 8))
plt.imshow(np.abs(psi_total), extent=[-r_max, r_max, -r_max, r_max], origin='lower', cmap='viridis')
plt.colorbar(label='Magnitude of Total Field')
circle = plt.Circle((0, 0), a, edgecolor='black', facecolor='none', linestyle='--')
plt.gca().add_patch(circle)
plt.title(f'Scattering from a Sphere (k={k:.2f}), Magnitude of Total Field')
plt.xlabel('x')
plt.ylabel('z')
plt.xlim([-r_max, r_max])
plt.ylim([-r_max, r_max])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()