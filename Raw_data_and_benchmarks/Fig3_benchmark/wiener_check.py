"""Consistency check for the CU-letter pipeline (Supplementary Section S7):
object -> Gaussian blur -> Poisson shot noise -> {VMS 20x20, Wiener} -> PSNR vs ground truth.
Conventions follow Reconstruction_of_image_low_noise/CU_letter/{VMS.py, DI_PMS.py}.
Usage: python wiener_check.py <repo_root>
"""
import math, sys
import numpy as np
from numpy.polynomial import hermite as H
from scipy.signal import fftconvolve
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

REPO = sys.argv[1]

# ---------- pipeline exactly as in VMS.py ----------
img = Image.open(f'{REPO}/Reconstruction_of_image_low_noise/CU_letter/300_300.png').convert('L')
obj = np.array(img, float)
obj = (obj - obj.min()) / (obj.max() - obj.min())
obj = (obj > 0.5).astype(float)
obj = obj[::-1, :].copy()                      # vertical flip as in VMS.py

sigma = 0.5 / 1000
n = 300
x = np.linspace(-0.003, 0.003, n); y = x.copy()
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

def Sai(X, Y, s):
    return (1 / (2 * np.pi * s**2)) * np.exp(-(X**2 + Y**2) / (2 * s**2))

cx = x[n // 2]
Sai_grid = Sai(X - cx, Y - cx, sigma)
E_blur = fftconvolve(obj, Sai_grid, mode='same')
E_blur = E_blur / E_blur.max()

def noisy(peak, rng):
    lam = np.clip(E_blur, 0, 1) * peak
    out = rng.poisson(lam).astype(float) / peak
    return (out - out.min()) / (out.max() - out.min())

# ---------- VMS 20x20, vectorized form of VMS.py ----------
m_max = 20
# alpha_table[m,k]: power-series coefficients of physicists' Hermite H_m
alpha_table = np.zeros((m_max, m_max))
for m in range(m_max):
    c = np.zeros(m + 1); c[m] = 1
    p = H.herm2poly(c)
    alpha_table[m, :len(p)] = p
fact = np.array([math.factorial(i) for i in range(m_max)], float)
C = np.einsum('k,l,mk,nl->mnkl', np.sqrt(fact), np.sqrt(fact), alpha_table, alpha_table)

# photocurrent basis: norm * H_k(x/sigma) H_l(y/sigma) * Sai(sigma)  (separable)
g1 = np.exp(-x**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
Hk_s = np.stack([H.hermval(x / sigma, np.eye(m_max)[k]) for k in range(m_max)])
W = Hk_s * g1[None, :]                         # (k, x) weighted rows
normkl = 1 / np.sqrt(2.0 ** (np.add.outer(np.arange(m_max), np.arange(m_max))) *
                     np.outer(fact, fact))

# reconstruction basis: H_m(x/(sqrt2 sigma)) etc., envelope exp(-r^2/4sigma^2)
Hk_r = np.stack([H.hermval(x / (np.sqrt(2) * sigma), np.eye(m_max)[k]) for k in range(m_max)])
env1 = np.exp(-x**2 / (4 * sigma**2))
denom = (1 / (np.pi * sigma**2)) / (2.0 ** (np.add.outer(np.arange(m_max), np.arange(m_max)) + 2)
                                    * np.outer(fact, fact))

def vms(E):
    P = normkl * (W @ E.T @ W.T).T * dx * dx   # P[k,l] with row=y(l)? check orientation
    # orientation: photo_current integrates B_kl(x,y)*E(x,y); B separable H_k(x)H_l(y)
    # E indexed [row=y, col=x]:  P[k,l] = sum_y sum_x Hl(y)g(y) Hk(x)g(x) E[y,x]
    P = (W @ E @ W.T).T * dx * dx * normkl     # P[k(x-order), l(y-order)]
    beta = np.tensordot(C, P, axes=([2, 3], [0, 1]))
    nrm = beta * denom
    R = np.einsum('mn,mi,nj->ji', nrm, Hk_r, Hk_r) * np.outer(env1, env1)
    R = R / R.max()
    Rclip = R.copy(); Rclip[Rclip < 0] = 0
    return Rclip, R

def metrics(img, ref=obj):
    dr = ref.max() - ref.min()
    return psnr(ref, img, data_range=dr), ssim(ref, img, data_range=dr), \
           np.linalg.norm(img - ref) / np.linalg.norm(ref), \
           np.corrcoef(ref.ravel(), img.ravel())[0, 1]

# ---------- Wiener ----------
otf = np.fft.fft2(np.fft.ifftshift(Sai_grid * dx * dx))   # unit-ish gain kernel

def wiener(yimg, lam):
    Yf = np.fft.fft2(yimg)
    Xf = np.conj(otf) / (np.abs(otf)**2 + lam) * Yf
    r = np.real(np.fft.ifft2(Xf))
    return r

def best_over_norms(r, ref=obj):
    outs = []
    a = (r - r.min()) / (r.max() - r.min())
    outs.append(a)
    b = np.clip(r / r.max(), 0, 1)
    outs.append(b)
    return max(metrics(o, ref)[0] for o in outs), outs

rng = np.random.default_rng(0)
Ynoisy = noisy(1e18, rng)

print('DI (noisy blurred) PSNR/SSIM/RSE/RSP: %.2f %.4f %.4f %.4f' % metrics(Ynoisy))

rec_c, rec_nc = vms(Ynoisy)
print('VMS 20x20 clip  PSNR/SSIM/RSE/RSP: %.2f %.4f %.4f %.4f' % metrics(rec_c))
print('VMS 20x20 noclp PSNR/SSIM/RSE/RSP: %.2f %.4f %.4f %.4f' % metrics(np.abs(rec_nc)))

lams = np.logspace(-16, 0, 33)
best = (-1, None)
for lam in lams:
    p, _ = best_over_norms(wiener(Ynoisy, lam))
    if p > best[0]: best = (p, lam)
print('Wiener oracle @1e18 photons: best PSNR %.2f dB at lambda=%.1e' % best)

p0, _ = best_over_norms(wiener(E_blur / 1.0, 1e-16))
print('Wiener on noise-free blurred image (lam=1e-16): PSNR %.2f dB' % p0)

# photon sweep like S13
for peak in [1e10, 1e12, 1e14, 1e16, 1e18]:
    Yp = noisy(peak, np.random.default_rng(1))
    bw = max(best_over_norms(wiener(Yp, lam))[0] for lam in np.logspace(-16, 0, 17))
    vc, _ = vms(Yp)
    print('peak=%.0e  Wiener oracle %.2f dB   VMS %.2f dB' % (peak, bw, metrics(vc)[0]))
