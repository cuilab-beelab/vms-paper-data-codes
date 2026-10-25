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
# E_DI=E_DI_noisy

import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftshift, ifftshift
Sai_grid /= np.sum(Sai_grid)

def pad_psf(psf, shape):
    pad = np.zeros(shape, psf.dtype)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, -ph//2, axis=0)
    pad = np.roll(pad, -pw//2, axis=1)
    return pad

psf_padded = pad_psf(Sai_grid, E_DI_noisy.shape)
H_f        = fft2(psf_padded)
H_f_conj   = np.conj(H_f)

# compute Lipschitz‐based step size
Lipschitz = np.max(np.abs(H_f)**2)
step_size = 0.9 / Lipschitz
def soft_threshold(x, thresh):
    """Soft-thresholding (proximal operator for L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def l1_deconv_ista(blurred, psf, lam, step_size, n_iter):
    """
    L1-regularized deconvolution via ISTA:
        min_f 0.5 ||h*f - g||_2^2 + lam * ||f||_1

    blurred   : observed image g (2D array)
    psf       : point-spread function (2D, same shape or smaller)
    lam       : L1 weight (>=0)
    step_size : gradient descent step (<= 1 / ||H||^2)
    n_iter    : number of ISTA iterations
    """
    # pad PSF to image size and compute its FFT once
    img_shape = blurred.shape
    psf_padded = np.zeros_like(blurred)
    cx, cy = img_shape[0]//2, img_shape[1]//2
    px, py = np.array(psf.shape)//2
    psf_padded[cx-px:cx-px+psf.shape[0], cy-py:cy-py+psf.shape[1]] = psf
    H_f = fft2(ifftshift(psf_padded))
    H_f_conj = np.conj(H_f)

    # initialize f (e.g. constant)
    f = np.full_like(blurred, 0.5)

    # FFT of the blurred image
    G_f = fft2(blurred)

    for i in range(n_iter):
        # compute H f_k via FFT
        F_f = fft2(f)
        Hf = np.real(ifft2(H_f * F_f))

        # gradient of 0.5||Hf - g||^2 = H^T*(Hf - g)
        residual = Hf - blurred
        grad = np.real(ifft2(H_f_conj * fft2(residual)))

        # gradient descent step
        f = f - step_size * grad

        # proximal L1 step
        f = soft_threshold(f, lam * step_size)

        # enforce non-negativity (optional)
        f = np.clip(f, 0, None)

    # normalize to [0,1] for visualization
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    return f

blurred = E_DI_noisy      # your noisy blurred image
psf     = Sai_grid        # your PSF

E_recovered_l1 = l1_deconv_ista(
    blurred    = E_DI_noisy,
    psf        = Sai_grid,
    step_size  = step_size,
    lam        = 1e-5,    # lighter sparsity
    n_iter     = 1000
)


# visualize alongside your other methods:
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16,4))
ax[0].imshow(object_matrix);      ax[0].set_title('Ground Truth')
ax[1].imshow(blurred);      ax[1].set_title('Blurred + Noisy')
ax[2].imshow(E_recovered_l1);    ax[2].set_title('L1-ISTA')
for a in ax: a.axis('off')
plt.tight_layout()
plt.show()
