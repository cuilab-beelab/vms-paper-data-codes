import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import pandas as pd
# Parameters
sigma = 0.5 / 1000
DFL = 2.905 * sigma
N = 1
delta = 1e-8  # for finite difference
m_max = 1  # max photon number to sum over

# Probability mass function Pr(m|theta)
def Q(theta, sigma):
    return theta**2 / (16 * sigma**2)

def Pr_theta(theta, sigma):
    q = Q(theta, sigma)
    return 1 - np.exp(-q)

# Finite difference of Pr(m|theta)
def dPr_dtheta(theta, sigma, delta):
    pr_plus = Pr_theta(theta + delta, sigma)
    pr_minus = Pr_theta(theta - delta, sigma)
    return (pr_plus - pr_minus) / (2 * delta)

# Fisher information
def fisher_discrete(theta, sigma, delta):
    pr = Pr_theta(theta, sigma)
    dpr = dPr_dtheta(theta, sigma, delta)
    fisher= (dpr**2) / pr
    return fisher

# Scan θ values
theta_vals = np.linspace(1e-6, 0.0035, 100)
F_vals = [fisher_discrete(theta, sigma, delta) for theta in theta_vals]
F_vals = np.array(F_vals)
# Plot
plt.figure(figsize=(7, 5))
plt.plot(theta_vals / DFL, F_vals*1e-6, 'b-', linewidth=2)
plt.xlabel('Separation θ / DFL', fontsize=12)
plt.ylabel('Fisher Information ℱ(θ)', fontsize=12)
plt.title('Discrete Fisher Information vs. Separation θ', fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()

# Create DataFrame
df = pd.DataFrame({
    'theta (microns)': theta_vals / DFL,
    'Fisher Information': F_vals*1e-6
})

# Path to save the file
save_path = r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Fisher\FI_physical_MS.xlsx'

# Save to Excel
df.to_excel(save_path, index=False)

print(f'Data saved to: {save_path}')
