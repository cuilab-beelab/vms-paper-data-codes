"""SI Fig. S10: Wiener (noise-free and oracle-tuned) vs virtual mode sorting at two noise
levels. Usage: python make_figS10.py <repo_root>  (the repository top-level directory)."""
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


# ---------------- Fig. S10 montage ----------------
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8.5})
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')

rows = []
for peak, seed in [(1e18, 0), (1e12, 3)]:
    Yn = noisy(peak, np.random.default_rng(seed))
    bestP, bestim = -1, None
    for lam in np.logspace(-12, 0, 13):
        p, outs = best_over_norms(wiener(Yn, lam))
        if p > bestP: bestP, bestim = p, outs[0]
    vc, _ = vms(Yn)
    rows.append((peak, Yn, bestim, bestP, vc, metrics(vc)[0]))
wnf_p, wnf_im = -1, None
for lam in np.logspace(-16, 0, 17):
    p, outs = best_over_norms(wiener(E_blur, lam))
    if p > wnf_p: wnf_p, wnf_im = p, outs[0]
fig, axs = plt.subplots(2, 5, figsize=(10.5, 4.3))
titles0 = ['Ground truth', 'Direct image\n(peak 10$^{18}$)', 'Wiener, noise-free\noracle $\\lambda$ (%.1f dB)' % wnf_p,
           'Wiener, oracle $\\lambda$\n(%.1f dB)' % rows[0][3], 'VMS 20\u00d720\n(%.1f dB)' % rows[0][5]]
ims0 = [obj, rows[0][1], wnf_im, rows[0][2], rows[0][4]]
titles1 = ['Ground truth', 'Direct image\n(peak 10$^{12}$)', 'Wiener, oracle $\\lambda$\n(%.1f dB)' % rows[1][3],
           'VMS 20\u00d720\n(%.1f dB)' % rows[1][5], '']
ims1 = [obj, rows[1][1], rows[1][2], rows[1][4], None]
for ax, im, t in zip(axs[0], ims0, titles0):
    ax.imshow(im, cmap='viridis'); ax.set_title(t, fontsize=7.5); ax.axis('off')
for ax, im, t in zip(axs[1], ims1, titles1):
    if im is None: ax.axis('off'); continue
    ax.imshow(im, cmap='viridis'); ax.set_title(t, fontsize=7.5); ax.axis('off')
axs[0,0].text(-0.25, 0.5, 'low noise', transform=axs[0,0].transAxes, rotation=90, va='center', fontsize=9)
axs[1,0].text(-0.25, 0.5, 'high noise', transform=axs[1,0].transAxes, rotation=90, va='center', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'figS10.png'), dpi=300)
print('saved figures/figS10.png')
