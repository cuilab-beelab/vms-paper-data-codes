import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
sigma = 0.5 / 1000
DFL = 2.905 * sigma
delta = 1e-6  # for finite difference
N = 1

# Meshgrid setup
x_min, x_max = -0.01, 0.01
y_min, y_max = -0.01, 0.01
number_of_mesh_x = 1000
number_of_mesh_y = 1000
x = np.linspace(x_min, x_max, number_of_mesh_x)
y = np.linspace(y_min, y_max, number_of_mesh_y)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# 2D PSF
def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def psi_shifted(X, Y, shift, sigma):
    return Sai(X + shift, Y, sigma)

def Pr(X, Y, theta, sigma):
    p1 = psi_shifted(X, Y, theta / 2, sigma)
    p2 = psi_shifted(X, Y, -theta / 2, sigma)
    return 0.5 * (p1 + p2)

def dlogPr_dtheta(X, Y, theta, sigma, delta):
    pr_plus = Pr(X, Y, theta + delta, sigma)
    pr_minus = Pr(X, Y, theta - delta, sigma)
    return (np.log(pr_plus + 1e-20) - np.log(pr_minus + 1e-20)) / (2 * delta)

def fisher_information_2D(X, Y, theta, sigma, delta):
    pr = Pr(X, Y, theta, sigma)
    dlog = dlogPr_dtheta(X, Y, theta, sigma, delta)
    integrand = (dlog**2) * pr
    return np.sum(integrand) * dx * dy

# Scan over theta values
theta_vals = np.linspace(1e-6, 0.0035, 100)  # avoid zero to prevent log singularity
F_vals = [N * fisher_information_2D(X, Y, theta, sigma, delta) for theta in theta_vals]
F_vals = np.array(F_vals)
# Plotting
plt.figure(figsize=(7, 5))
plt.plot(theta_vals / DFL, F_vals*1e-6, linewidth=2)
plt.xlabel('Separation θ / DFL', fontsize=12)
plt.ylabel('Fisher Information ℱ(θ)', fontsize=12)
plt.title('Fisher Information vs. Separation Distance', fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()

# Create DataFrame
df = pd.DataFrame({
    'theta (microns)': theta_vals / DFL,
    'Fisher Information': F_vals*1e-6
})

# Path to save the file
save_path = r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Fisher\DI.xlsx'

# Save to Excel
df.to_excel(save_path, index=False)

print(f'Data saved to: {save_path}')
