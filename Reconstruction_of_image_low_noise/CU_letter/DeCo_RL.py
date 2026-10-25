import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open(r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Comparison_DI_vMS_pMS_DeC\CU_300_300.png').convert('L')
object_matrix = np.array(img, dtype=float)

object_matrix=(object_matrix-np.min(object_matrix))/(np.max(object_matrix)-np.min(object_matrix))
object_matrix = (object_matrix > 0.5).astype(float)

# Define PSF
def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (x**2 + y**2) / (2 * sigma ** 2))

# Meshgrid setup
x_min, x_max = -0.003, 0.003
y_min, y_max = -0.003, 0.003
number_of_mesh_x = 300
number_of_mesh_y = 300
x = np.linspace(x_min, x_max, number_of_mesh_x)
y = np.linspace(y_min, y_max, number_of_mesh_y)
X, Y = np.meshgrid(x, y)

# Parameters
sigma = 0.5 / 1000
radius = 0.003/2.5

# Create object and PSF
E_grid = object_matrix
Sai_grid = Sai(X - X[number_of_mesh_x//2, number_of_mesh_x//2],
               Y - Y[number_of_mesh_y//2, number_of_mesh_y//2], sigma)

# Simulate blurred image
from scipy.signal import convolve2d
E_DI = convolve2d(E_grid, Sai_grid, mode='same', boundary='wrap')
# E_DI = (E_DI - E_DI.min()) / (E_DI.max() - E_DI.min())
photon_number=10**4
E_DI_norm = E_DI / np.sum(E_DI) * photon_number
sdeviation = 1 / np.sqrt(E_DI_norm)
E_DI_noisy = E_DI + np.random.normal(0, sdeviation)
E_DI_noisy = (E_DI_noisy - E_DI_noisy.min()) / (E_DI_noisy.max() - E_DI_noisy.min())
E_DI=E_DI_noisy


import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import sobel

def gradient_and_divergence(f):
    eps = 1e-12

    # Compute gradients
    grad_x = sobel(f, axis=1, mode='wrap')  # horizontal gradient
    grad_y = sobel(f, axis=0, mode='wrap')  # vertical gradient

    # Normalize gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + eps
    norm_grad_x = grad_x / grad_mag
    norm_grad_y = grad_y / grad_mag

    # Compute divergence of normalized gradient
    div_x = np.gradient(norm_grad_x, axis=1)
    div_y = np.gradient(norm_grad_y, axis=0)
    divergence = div_x + div_y

    return divergence

def richardson_lucy_tv_step(f_n, S_C, psf, psf_flipped, lam=0.002, eps=1e-12):
    conv = fftconvolve(f_n, psf, mode='same')
    ratio = S_C / (conv + eps)
    correction = fftconvolve(ratio, psf_flipped, mode='same')

    # TV regularization
    div_term = gradient_and_divergence(f_n)
    denominator = 1 - lam * div_term
    denominator = np.clip(denominator, 1e-2, None)

    # Update
    f_n_plus1 = correction * f_n / denominator
    return np.clip(f_n_plus1, 0, None)

def richardson_lucy_tv_deconv(S_C, psf, iterations, lam=0.002, threshold=1e-4):
    f_n = np.full_like(S_C, 0.5)
    psf_flipped = np.flip(psf, (0, 1))

    for i in range(iterations):
        f_n_plus1 = richardson_lucy_tv_step(f_n, S_C, psf, psf_flipped, lam)

        # Convergence check from Eq. (24)
        diff = np.sum(np.abs(f_n_plus1 - f_n)) / np.sum(np.abs(f_n))
        f_n = f_n_plus1
        if diff < threshold:
            print(f"Converged at iteration {i+1} with diff={diff:.2e}")
            break

    return (f_n - f_n.min()) / (f_n.max() - f_n.min())

E_recovered = richardson_lucy_tv_deconv(E_DI, Sai_grid, iterations=150, lam=0.002)
E_recovered = (E_recovered - E_recovered.min()) / (E_recovered.max() - E_recovered.min())

import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Ensure everything is float
GT   = object_matrix.astype(float)
DI   = E_DI.astype(float)
MS   = np.abs(E_recovered).astype(float)

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
# ---- Visualization ----
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(E_grid)
axs[0].set_title('Original Object')
axs[1].imshow(E_DI)
axs[1].set_title('Blurred Image')
axs[2].imshow(E_recovered)
axs[2].set_title('Recovered (Deconvolved)')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
