import math
import numpy as np
from sympy import symbols, hermite, expand
from scipy.special import hermite as scipy_hermite
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image
from scipy.signal import convolve2d
# Define the variable
u = symbols('u')
import scipy.ndimage
import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import time

# Start timing
tic = time.time()

image_path = r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Comparison_DI_vMS_pMS_DeC\Logo_200.jpg' # Replace 'your_image_name.jpg' with your actual image file name
# Load the image from the specified path
image = Image.open(image_path)

# Convert the image to a numpy array (RGB)
object_matrix = np.array(image)

# Custom grayscale mapping function
def custom_rgb_to_gray(rgb_image):
    # Create an empty array for the grayscale image
    gray_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))

    # Iterate through each pixel in the RGB image
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            r, g, b = rgb_image[i, j]

            # Custom logic for assigning grayscale values
            if r == 255 and g == 0 and b == 0:  # Red has highest value
                gray_image[i, j] = 255
            elif r == 255 and g == 255 and b == 255:  # White has middle value
                gray_image[i, j] = 5
            elif r == 0 and g == 0 and b == 255:  # Blue has lowest value
                gray_image[i, j] = 1
            else:
                # For other colors (if any), apply a basic grayscale conversion
                gray_image[i, j] = 1 * r + 0.00 * g + 0.00 * b

    return gray_image

# Apply the custom grayscale conversion
object_matrix = custom_rgb_to_gray(object_matrix)

object_matrix=(object_matrix-np.min(object_matrix))/(np.max(object_matrix)-np.min(object_matrix))
object_matrix = (object_matrix > 0.667).astype(float)+1e-17

from scipy.signal import fftconvolve
import numpy as np


def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (x ** 2 + y ** 2) / (2 * sigma ** 2))

# ---- New function with only one input ----
def l1_deconv_ista_single(blurred):

    sigma = 0.75 / 1000

    x_min = -0.006
    x_max = 0.006
    y_min = -0.006
    y_max = 0.006
    number_of_mesh_x = 66
    number_of_mesh_y = 66
    # Create meshgrid for x and y
    x = np.linspace(x_min, x_max, number_of_mesh_x)
    y = np.linspace(y_min, y_max, number_of_mesh_y)
    X, Y = np.meshgrid(x, y)
    Sai_grid = Sai(X - X[number_of_mesh_x // 2, number_of_mesh_x // 2],
                   Y - Y[number_of_mesh_y // 2, number_of_mesh_y // 2], sigma)
    # Normalize PSF
    Sai_grid /= np.sum(Sai_grid)

    psf_padded = np.zeros_like(blurred)
    cx, cy = psf_padded.shape[0] // 2, psf_padded.shape[1] // 2
    px, py = np.array(Sai_grid.shape) // 2
    psf_padded[cx - px:cx - px + Sai_grid.shape[0], cy - py:cy - py + Sai_grid.shape[1]] = Sai_grid
    H_f = fft2(ifftshift(psf_padded))
    H_f_conj = np.conj(H_f)

    # Precomputed step size
    Lipschitz = np.max(np.abs(H_f) ** 2)
    step_size = 0.9 / Lipschitz
    def soft_threshold(x, thresh):
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

    lam = 1e-5
    n_iter = 1000
    # Initialize recovered image
    f = np.full_like(blurred, 0.5)

    for _ in range(n_iter):
        F_f = fft2(f)
        Hf = np.real(ifft2(H_f * F_f))
        residual = Hf - blurred
        grad = np.real(ifft2(H_f_conj * fft2(residual)))
        f = soft_threshold(f - step_size * grad, lam * step_size)
        f = np.clip(f, 0, None)

    # Normalize output
    return (f - f.min()) / (f.max() - f.min() + 1e-12)

def final_DI_image(object_matrix):
    # Parameters
    photon_number = 10 ** 17
    sigma = 0.75 / 1000

    x_min = -0.006
    x_max = 0.006
    y_min = -0.006
    y_max = 0.006
    number_of_mesh_x = 66
    number_of_mesh_y = 66
    # Create meshgrid for x and y
    x = np.linspace(x_min, x_max, number_of_mesh_x)
    y = np.linspace(y_min, y_max, number_of_mesh_y)
    X, Y = np.meshgrid(x, y)
    E_grid = object_matrix
    Sai_grid = Sai(X - X[number_of_mesh_x // 2, number_of_mesh_x // 2],
                   Y - Y[number_of_mesh_y // 2, number_of_mesh_y // 2], sigma)
    E_DI = fftconvolve(E_grid, Sai_grid, mode='same')
    # E_DI_norm = E_DI / np.sum(E_DI) * photon_number
    # sdeviation = 1 / np.sqrt(E_DI_norm)
    # E_DI_noisy = E_DI + np.random.normal(0, sdeviation)
    # E_DI = (E_DI_noisy - E_DI_noisy.min()) / (E_DI_noisy.max() - E_DI_noisy.min())
    E_DI = (E_DI - E_DI.min()) / (E_DI.max() - E_DI.min())
    return E_DI

sub_arrays = []
sub_arrays_with_border = []
for i in range(4):
    for j in range(4):
        sub_arrays.append(object_matrix[i*50:(i+1)*50, j*50:(j+1)*50])
        sub_array = object_matrix[i*50:(i+1)*50, j*50:(j+1)*50]  # Extract 50x50 sub-array
        padded_sub_array = np.pad(sub_array, pad_width=8, mode='constant', constant_values=0)  # Add 5 rows/columns of zeros
        sub_arrays_with_border.append(padded_sub_array)

DI_sub_arrays = [final_DI_image(sub) for sub in sub_arrays]
DI_sub_arrays_with_border = [final_DI_image(sub) for sub in sub_arrays_with_border]

reconstructed_DI_rows = []
reconstructed_DI_rows_with_border = []
for i in range(4):
    row = np.hstack(DI_sub_arrays[i*4:(i+1)*4])
    row_with_border = np.hstack(DI_sub_arrays_with_border[i*4:(i+1)*4])
    reconstructed_DI_rows.append(row)
    reconstructed_DI_rows_with_border.append(row_with_border)
DI_final_array = np.vstack(reconstructed_DI_rows)
DI_final_array_with_border = np.vstack(reconstructed_DI_rows_with_border)

sub_arrays = []
for i in range(4):
    for j in range(4):
        sub_arrays.append(DI_final_array_with_border[i*66:(i+1)*66, j*66:(j+1)*66])

DeCo_sub_arrays = [l1_deconv_ista_single(sub)[8:-8, 8:-8] for sub in sub_arrays]  # Crop to 50x50

reconstructed_DeCo_rows = []
for i in range(4):
    row = np.hstack(DeCo_sub_arrays[i*4:(i+1)*4])
    reconstructed_DeCo_rows.append(row)

DeCo_final_array = np.vstack(reconstructed_DeCo_rows)

import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Ensure everything is float
GT   = object_matrix.astype(float)
DI   = DI_final_array.astype(float)
MS   = np.abs(DeCo_final_array).astype(float)

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
x_min = -0.006
x_max = 0.006
y_min = -0.006
y_max = 0.006
number_of_mesh_x = 200
number_of_mesh_y = 200

# Create meshgrid for x and y
x = np.linspace(x_min, x_max, number_of_mesh_x)
y = np.linspace(y_min, y_max, number_of_mesh_y)
X, Y = np.meshgrid(x, y)

# End timing
toc = time.time()

# Calculate elapsed time
elapsed_time = toc - tic
print("Elapsed time:", elapsed_time, "seconds")

contour = plt.contourf(x*1000, y*1000, DeCo_final_array, levels=200, cmap='viridis')
plt.colorbar(contour)
plt.title('Reconstructed Image')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure()

contour_di = plt.contourf(x*1000, y*1000, DI_final_array, levels=200, cmap='viridis')
plt.colorbar(contour_di)
plt.title('Direct Imaging')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure()

contour_obj = plt.contourf(x*1000, y*1000, object_matrix, levels=200, cmap='viridis')
plt.colorbar(contour_obj)
plt.title('Object')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
