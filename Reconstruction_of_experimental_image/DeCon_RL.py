import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import sobel
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from skimage.transform import resize
# ——— PSF and TV‐Richardson-Lucy definitions (unchanged) ———
def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (x**2 + y**2) / (2 * sigma ** 2))

def gradient_and_divergence(f):
    eps = 1e-12
    gx = sobel(f, axis=1, mode='wrap')
    gy = sobel(f, axis=0, mode='wrap')
    mag = np.sqrt(gx*gx + gy*gy) + eps
    nx, ny = gx/mag, gy/mag
    div = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)
    return div

def rl_tv_step(f, S_C, psf, psf_flip, lam=0.002, eps=1e-12):
    conv = fftconvolve(f, psf, mode='same')
    corr = fftconvolve(S_C/(conv+eps), psf_flip, mode='same')
    div = gradient_and_divergence(f)
    denom = np.clip(1 - lam*div, 1e-2, None)
    return np.clip(corr * f / denom, 0, None)

def richardson_lucy_tv(S_C, psf, iters=80, lam=0.002, thresh=1e-4):
    f = np.full_like(S_C, 0.5)
    psf_flip = np.flip(psf, (0,1))
    for k in range(iters):
        f1 = rl_tv_step(f, S_C, psf, psf_flip, lam)
        if np.sum(np.abs(f1-f))/np.sum(np.abs(f)) < thresh:
            print(f"Converged at iter {k+1}")
            f = f1
            break
        f = f1
    return (f - f.min())/(f.max()-f.min())

# ——— setup mesh & PSF ———
N = 60
x = np.linspace(-1,1,N)
X, Y = np.meshgrid(x,x)
sigma = 0.15
center = N//2
psf = Sai(X - X[center,center], Y - Y[center,center], sigma)

# ——— your 4 filenames ———
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

# ——— storage lists ———
objs, di_imgs, recs = [], [], []

for obj_fp, di_fp in zip(object_paths, di_paths):
    # load & normalize
    df = pd.read_excel(obj_fp, header=None, index_col=None)
    O= df.values  # shape (60,60)
    df = pd.read_excel(di_fp, header=None, index_col=None)
    D= df.values  # shape (60,60)
    O = (O - O.min())/(O.max()-O.min())
    D = (D - D.min())/(D.max()-D.min())

    # deconvolve
    R = richardson_lucy_tv(D, psf, iters=300, lam=0.002)

    objs.append(O)
    di_imgs.append(D)
    recs.append(R)

# ——— plot a 4×3 grid ———
fig, axs = plt.subplots(4, 3, figsize=(12, 16), constrained_layout=True)

for i in range(4):
    for j, (img, title) in enumerate([
        (objs[i], 'Object'),
        (di_imgs[i], 'Blurred'),
        (recs[i], 'Recovered')
    ]):
        ax = axs[i, j]
        ax.imshow(img, interpolation='bicubic', origin='lower')
        # only top row gets column titles
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
    compute_metrics('RL', REC, GT)
]

df = pd.DataFrame(results).set_index('Method')
print(df.round(6).to_string())