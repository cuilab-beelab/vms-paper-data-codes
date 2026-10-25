import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve, convolve2d
from scipy.fft import fft2, ifft2, ifftshift
import os
import pandas as pd
from skimage.transform import resize
# ——— PSF helper ———
def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# ——— pad PSF to image size and center it ———
def pad_psf(psf, shape):
    pad = np.zeros(shape, psf.dtype)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, -ph//2, axis=0)
    pad = np.roll(pad, -pw//2, axis=1)
    return pad

# ——— soft-threshold (prox of L1) ———
def soft_threshold(x, thresh):
    """Soft-thresholding (proximal operator for L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def tv_deconv(blurred, lam, step_size, n_iter):
    """
    gradient-descent on 0.5||Hf–g||^2 + lam*TV(f)
    """
    f = blurred.copy()
    for i in range(n_iter):
        # fidelity gradient via FFT
        Hf       = np.real(ifft2(H_f * fft2(f)))
        grad_data = np.real(ifft2(H_f_conj * fft2(Hf - blurred)))
        # TV gradient (Rudin-Osher-Fatemi)
        fx, fy = np.gradient(f, axis=1), np.gradient(f, axis=0)
        mag     = np.sqrt(fx*fx + fy*fy) + 1e-8
        div     = np.gradient(fx/mag, axis=1) + np.gradient(fy/mag, axis=0)
        # full gradient
        grad = grad_data - lam*div
        # descent step + clip
        f = f - step_size * grad
        f = np.clip(f, 0, 1)
    return (f - f.min())/(f.max() - f.min() + 1e-12)
N = 60
x = np.linspace(-1,1,N)
X, Y = np.meshgrid(x,x)
sigma = 0.15
center = N//2
psf = Sai(X - X[center,center], Y - Y[center,center], sigma)

# ——— file lists for 4 pairs ———
object_paths = [
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Obj_0_52.xlsx',
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Obj_90_32.xlsx',
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Obj_180_44.xlsx',
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Obj_270_44.xlsx',
]
# di_paths = [
#     r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Exp_0_52.xlsx',
#     r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Exp_90_32.xlsx',
#     r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Exp_180_44.xlsx',
#     r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\Exp_270_44.xlsx',
# ]

di_paths = [
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\DeNo_exp_0_52.xlsx',
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\DeNo_exp_90_32.xlsx',
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\DeNo_exp_180_44.xlsx',
    r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results\DeNo_exp_270_44.xlsx',
]

# ——— main processing ———
objs, di_imgs, recs = [], [], []

# PSF parameters (must match image size)
x_min, x_max = -1, 1
y_min, y_max = -1, 1
sigma = 0.15

for obj_fp, di_fp in zip(object_paths, di_paths):
    # load & normalize
    df = pd.read_excel(obj_fp, header=None, index_col=None)
    O= df.values  # shape (60,60)
    df = pd.read_excel(di_fp, header=None, index_col=None)
    D= df.values  # shape (60,60)
    O = (O - O.min())/(O.max()-O.min())
    D = (D - D.min())/(D.max()-D.min())

    # build PSF on the same grid
    ny, nx = D.shape
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    Sai_grid = Sai(
        X - X[ny//2, nx//2],
        Y - Y[ny//2, nx//2],
        sigma
    )
    Sai_grid /= np.sum(Sai_grid)
    psf_padded = pad_psf(Sai_grid, D.shape)
    H_f = fft2(psf_padded)
    H_f_conj = np.conj(H_f)

    # compute Lipschitz‐based step size
    Lipschitz = np.max(np.abs(H_f) ** 2)
    step_size = 0.9 / Lipschitz

    blurred = D  # your noisy blurred image
    psf = Sai_grid  # your PSF

    L = np.max(np.abs(H_f) ** 2)
    alpha = 0.9 / L  # gradient step for fidelity
    lambda_tv = 1e-3  # try 1e-3…1e-1 for your logo
    iters = 1000

    E_recovered_tv = tv_deconv(
        blurred=D,
        lam=lambda_tv,
        step_size=alpha,
        n_iter=iters
    )

    objs.append(O)
    di_imgs.append(D)
    recs.append(E_recovered_tv)


# ——— plot 4×3 grid ———
fig, axs = plt.subplots(4, 3, figsize=(12, 16), constrained_layout=True)

for i in range(4):
    for j, (img, title) in enumerate([
        (objs[i], 'Object'),
        (di_imgs[i], 'Blurred'),
        (recs[i], 'Recovered (TV)')
    ]):
        ax = axs[i, j]
        ax.imshow(img, interpolation='bicubic', origin='lower')
        if i == 0:
            ax.set_title(title, fontsize=12)
        ax.axis('off')

plt.show()

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def compute_metrics(name, img, reference):
    dr = reference.max() - reference.min()
    return {
        'Method': name,
        'PSNR(dB):H-B': psnr(reference, img, data_range=dr),
        'SSIM:H-B': ssim(reference, img, data_range=dr),
        'RSE(normRMSE):L-B': np.linalg.norm(img - reference) / np.linalg.norm(reference),
        'RSP(Pearsonr):H-B': np.corrcoef(reference.ravel(), img.ravel())[0, 1],
        'StdDev(norm):H-B': np.std(img) / np.std(reference)
    }
objst = objs[0]
objst = (objst > 0.5).astype(float)
di_imgst = di_imgs[0]
recst = recs[0]

objst = resize(objst, (60, 60), anti_aliasing=True)
di_imgst = resize(di_imgst, (60, 60), anti_aliasing=True)
recst = resize(recst, (60, 60), anti_aliasing=True)

objst  = (objst - objst.min()) / (objst.max() - objst.min())
di_imgst  = (di_imgst - di_imgst.min()) / (di_imgst.max() - di_imgst.min())
recst  = (recst - recst.min()) / (recst.max() - recst.min())

# Compute and print statistics for the first example
GT = objst.astype(float)
DI = di_imgst.astype(float)
REC = recst.astype(float)

results = [
    compute_metrics('Direct Imaging', DI, GT),
    compute_metrics('TV', REC, GT)
]

df = pd.DataFrame(results).set_index('Method')
print(df.round(6).to_string())