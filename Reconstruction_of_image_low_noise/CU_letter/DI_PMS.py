import math
import numpy as np
from sympy import symbols, hermite, expand
from scipy.special import hermite as scipy_hermite
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image
from scipy.special import j1
import time

from PIL import Image
import numpy as np

img = Image.open(r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Comparison_DI_vMS_pMS_DeC\CU_300_300.png').convert('L')
object_matrix = np.array(img, dtype=float)

object_matrix=(object_matrix-np.min(object_matrix))/(np.max(object_matrix)-np.min(object_matrix))
object_matrix = (object_matrix > 0.5).astype(float)

B = np.empty_like(object_matrix)          # allocate output array

for j in range(object_matrix.shape[1]):   # for each column
    B[:, j] = object_matrix[::-1, j]      # copy column j, but reversed top-to-bottom
object_matrix=B
# Start timing
tic = time.time()
# Define the variable
u = symbols('u')

# Function to calculate the alpha_mk coefficient for the Hermite polynomial H_m(x)
def alpha(m, k):
    # Get the Hermite polynomial H_m(u)
    H = hermite(m, u)
    # Expand H_m(u) to get the power series representation
    H_expanded = expand(H)
    # Extract the coefficients of the expanded polynomial
    coeffs = H_expanded.as_coefficients_dict()
    # Return the coefficient alpha_mk or 0 if it does not exist
    return coeffs.get(u**k, 0)

def photo_current(m, n, sigma, DFL, object_matrix):
    x_min, x_max, y_min, y_max = -0.003, 0.003, -0.003, 0.003
    number_of_mesh_x, number_of_mesh_y = 300, 300
    dx, dy = (x_max - x_min) / number_of_mesh_x, (y_max - y_min) / number_of_mesh_y
    x = np.linspace(x_min, x_max, number_of_mesh_x)
    y = np.linspace(y_min, y_max, number_of_mesh_y)
    X, Y = np.meshgrid(x, y)  # Create meshgrid for vectorized operations
    Bases = 1 / (math.sqrt(math.factorial(m) * math.factorial(n))) * (X / (2 ** 0.5 * sigma)) ** m * (
                Y / (2 ** 0.5 * sigma)) ** n * np.exp(-1 * (X ** 2 + Y ** 2) / (4 * sigma ** 2))
    return np.sum(Bases * object_matrix) * dx * dy

# Function to calculate the beta coefficient
def beta(m, n, sigma, DFL,object_matrix):
    beta_mn = 0
    for k in range(m+1):
        for l in range(n+1):
            b_mn = (math.sqrt(math.factorial(k) * math.factorial(l))) * \
                   alpha(m, k) * alpha(n, l) * \
                   photo_current(k, l, sigma, DFL,object_matrix)
            beta_mn += b_mn
    return beta_mn

# Function to reconstruct the image
def Recontructed_image(x, y, m, n, sigma, DFL,object_matrix):
    # Use scipy's hermite function for numerical computation
    Hm = scipy_hermite(m)(x / (2**0.5 * sigma))
    Hn = scipy_hermite(n)(y / (2**0.5 * sigma))
    E_mn = beta(m, n, sigma, DFL,object_matrix) / \
           (2**(m+n+2) * math.factorial(m) * math.factorial(n) * np.pi * sigma**2) * \
           Hm * Hn * np.exp(-1 * (x**2 + y**2) / (4 * sigma**2))
    return E_mn

def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (x**2 + y**2) / (2 * sigma ** 2))

# Parameters
sigma = 0.5 / 1000
radius = 0.003/2.5
DFL=1.5/1000

# Define the range for x and y
x_min = -0.003
x_max = 0.003
y_min = -0.003
y_max = 0.003
number_of_mesh_x = 300
number_of_mesh_y = 300
m_max=20
n_max=20
# Create meshgrid for x and y
x = np.linspace(x_min, x_max, number_of_mesh_x)
y = np.linspace(y_min, y_max, number_of_mesh_y)
X, Y = np.meshgrid(x, y)
E_grid = object_matrix
Sai_grid = Sai(X - X[number_of_mesh_x//2, number_of_mesh_x//2], Y - Y[number_of_mesh_y//2, number_of_mesh_y//2], sigma)
E_DI = fftconvolve(E_grid, Sai_grid, mode='same')
photon_number=10**(4)
E_DI_norm = E_DI / np.sum(E_DI) * photon_number
sdeviation = 1 / np.sqrt(E_DI_norm)
E_DI_noisy = E_DI + np.random.normal(0, sdeviation)
E_DI_noisy = (E_DI_noisy - E_DI_noisy.min()) / (E_DI_noisy.max() - E_DI_noisy.min())
E_DI=E_DI_noisy

# ── Precompute Hermite‐Gauss basis components ──
from scipy.special import eval_hermite

# 1D coordinates scaled by (2*sigma)
xi = x / (2**0.5 * sigma)        # shape (50,), dtype float64
yj = y / (2**0.5 * sigma)        # shape (50,), dtype float64
# stack returns the right shape, but might infer object dtype
Hx = np.stack([eval_hermite(m, xi) for m in range(m_max)], axis=0)
Hy = np.stack([eval_hermite(n, yj) for n in range(n_max)], axis=0)
# **Immediately** cast them to float64
Hx = Hx.astype(np.float64)       # now dtype float64
Hy = Hy.astype(np.float64)
# Gaussian envelope — also ensure it's float64
exp_factor = np.exp(-(X**2 + Y**2) / (4 * sigma**2)).astype(np.float64)
# ── Precompute object‐independent β‐coeffs ──
m_max, n_max = Hx.shape[0], Hy.shape[0]

# factorial lookup
fact = np.array([math.factorial(i) for i in range(max(m_max,n_max))], dtype=float)

# alpha_table[m,k] = α(m,k)
alpha_table = np.zeros((m_max, m_max), dtype=float)
for m in range(m_max):
    for k in range(m_max):
        alpha_table[m,k] = float(alpha(m,k))

# build 4-D mixing tensor C[m,n,k,l] = √(k! l!) α(m,k) α(n,l)
C = np.zeros((m_max, n_max, m_max, n_max), dtype=float)
for m in range(m_max):
    for n in range(n_max):
        for k in range(m_max):
            for l in range(n_max):
                C[m,n,k,l] = math.sqrt(fact[k]*fact[l]) \
                            * alpha_table[m,k] \
                            * alpha_table[n,l]

# precompute denominator denom[m,n] = prefac/(2^(m+n+2)*m!*n!)
prefac = 1/(np.pi * sigma**2)
denom = np.zeros((m_max,n_max), dtype=float)
denom_s = np.zeros((m_max,n_max), dtype=float)
for m in range(m_max):
    for n in range(n_max):
        denom[m,n] = prefac / (2**(m+n+2) * fact[m] * fact[n])


# End timing
toc = time.time()

# Calculate elapsed time
elapsed_time = toc - tic
print("Elapsed time:", elapsed_time, "seconds")

# Start timing
tic = time.time()


# 1) fill photocurrent matrix P[k,ℓ]
P = np.zeros((m_max, n_max), dtype=float)
for k in range(m_max):
    for ℓ in range(n_max):
        P[k,ℓ] = photo_current(k, ℓ, sigma, DFL, object_matrix)

# 2) contract to get all β in one shot
β = np.tensordot(C, P, axes=([2,3], [0,1]))   # shape (m_max,n_max)

# 3) apply your denominator to get the final normalization
norm = β * denom                            # shape (m_max,n_max)

# 4) rebuild ReIm with your precomputed bases
ReIm = np.zeros_like(exp_factor, dtype=np.complex128)
for m in range(m_max):
    for n in range(n_max):
        # pick off the correct 2D basis at (m,n)
        Hmn = (Hy[n][:,None] * Hx[m][None,:]) * exp_factor
        ReIm += norm[m,n] * Hmn
#ReIm = (ReIm - ReIm.min()) / (ReIm.max() - ReIm.min())
ReIm = ReIm / ReIm.max()
ReIm[ReIm < 0] = 0
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Ensure everything is float
GT   = object_matrix.astype(float)
DI   = E_DI.astype(float)
MS   = np.abs(ReIm).astype(float)

# Helper to compute each metric
def compute_metrics(name, img, reference):
    dr = reference.max() - reference.min()
    return {
        'Method': name,
        'PSNR(dB):H-B':    psnr(reference, img, data_range=dr),
        'SSIM:H-B':         ssim(reference, img, data_range=dr),
        'RSE(normRMSE):L-B': np.linalg.norm(img - reference) / np.linalg.norm(reference),
        'RSP(Pearsonr):H-B': np.corrcoef(reference.ravel(), img.ravel())[0,1],
        'StdDev(norm):H-B':   np.std(img) / np.std(reference)
    }

# Compute for both methods
results = [
    compute_metrics('Direct Imaging', DI, GT),
    compute_metrics('Mode Sorting',  MS, GT)
]

df = pd.DataFrame(results).set_index('Method')

# Print with full values
print(df.round(6).to_string())
# End timing
toc = time.time()

# Calculate elapsed time
elapsed_time = toc - tic
print("Elapsed time:", elapsed_time, "seconds")

contour = plt.contourf(x/DFL, y/DFL, abs(ReIm), levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('Reconstructed Image')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Create a new figure for the second plot
plt.figure()

# Plot for DI (the variable you want to plot in a new window)
contour_di = plt.contourf(x/DFL, y/DFL, abs(E_DI), levels=50, cmap='viridis')
plt.colorbar(contour_di)
plt.title('Direct Imaging')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Create a new figure for the second plot
plt.figure()

# Plot for DI (the variable you want to plot in a new window)
contour_di = plt.contourf(x/DFL, y/DFL, abs(object_matrix), levels=50, cmap='viridis')
plt.colorbar(contour_di)
plt.title('Object')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
