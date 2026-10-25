"""Fig. 3 m-p benchmark: all methods on identical noisy realizations.
Objects: CU letter (300x300). Methods: DI, VMS20 (clip/noclip), PMS20/35 (noiseless),
RL-TV, L1-ISTA, L1-FISTA, TV, Wiener — each with an oracle hyperparameter sweep.
Seeds x photon levels; metrics follow the DI_PMS.py conventions. Outputs npz+xlsx+png.
"""
import math, sys, time, json
import numpy as np
from numpy.polynomial import hermite as H
from scipy.signal import fftconvolve
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

REPO = sys.argv[1]; OUT = sys.argv[2]

# ---------------- shared pipeline ----------------
img = Image.open(f'{REPO}/Reconstruction_of_image_low_noise/CU_letter/300_300.png').convert('L')
obj = np.array(img, float); obj = (obj - obj.min()) / (obj.max() - obj.min())
obj = (obj > 0.5).astype(float)[::-1, :].copy()
sigma = 0.5e-3; n = 300
x = np.linspace(-3e-3, 3e-3, n); dx = x[1] - x[0]
X, Y = np.meshgrid(x, x)
Sai = (1/(2*np.pi*sigma**2))*np.exp(-((X-x[n//2])**2+(Y-x[n//2])**2)/(2*sigma**2))
blur = fftconvolve(obj, Sai, mode='same'); blur /= blur.max()

def noisy(peak, rng):
    out = rng.poisson(np.clip(blur,0,1)*peak).astype(float)/peak
    return (out-out.min())/(out.max()-out.min())

def metrics(im, ref=obj):
    dr = ref.max()-ref.min()
    return dict(PSNR=psnr(ref, im, data_range=dr), SSIM=ssim(ref, im, data_range=dr),
                RSE=float(np.linalg.norm(im-ref)/np.linalg.norm(ref)),
                RSP=float(np.corrcoef(ref.ravel(), im.ravel())[0,1]))

# ---------------- VMS 20x20 (vectorized form of VMS.py) ----------------
def build_vms(m_max):
    at = np.zeros((m_max, m_max))
    for m in range(m_max):
        c = np.zeros(m+1); c[m] = 1; p = H.herm2poly(c); at[m,:len(p)] = p
    fact = np.array([math.factorial(i) for i in range(m_max)], float)
    C = np.einsum('k,l,mk,nl->mnkl', np.sqrt(fact), np.sqrt(fact), at, at)
    g1 = np.exp(-x**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    Hks = np.stack([H.hermval(x/sigma, np.eye(m_max)[k]) for k in range(m_max)])
    W = Hks*g1[None,:]
    nk = 1/np.sqrt(2.0**np.add.outer(np.arange(m_max),np.arange(m_max))*np.outer(fact,fact))
    Hkr = np.stack([H.hermval(x/(np.sqrt(2)*sigma), np.eye(m_max)[k]) for k in range(m_max)])
    env = np.exp(-x**2/(4*sigma**2))
    den = (1/(np.pi*sigma**2))/(2.0**(np.add.outer(np.arange(m_max),np.arange(m_max))+2)*np.outer(fact,fact))
    def run(E):
        P = (W @ E @ W.T).T*dx*dx*nk
        beta = np.tensordot(C,P,axes=([2,3],[0,1]))
        R = np.einsum('mn,mi,nj->ji', beta*den, Hkr, Hkr)*np.outer(env,env)
        R = R/R.max(); Rc = R.copy(); Rc[Rc<0] = 0
        return Rc, np.abs(R)
    return run
vms20 = build_vms(20)

# ---------------- PMS simulation (DI_PMS.py conventions, noiseless object) ----------------
def pms(m_max):
    fact = np.array([math.factorial(i) for i in range(m_max)], float)
    mono = np.stack([ (x/(np.sqrt(2)*sigma))**k for k in range(m_max)])
    genv = np.exp(-x**2/(4*sigma**2))
    Wm = mono*genv[None,:]
    P = (Wm @ obj @ Wm.T).T*dx*dx / np.sqrt(np.outer(fact,fact))
    at = np.zeros((m_max,m_max))
    for m in range(m_max):
        c = np.zeros(m+1); c[m]=1; p = H.herm2poly(c); at[m,:len(p)] = p
    C = np.einsum('k,l,mk,nl->mnkl', np.sqrt(fact), np.sqrt(fact), at, at)
    beta = np.tensordot(C,P,axes=([2,3],[0,1]))
    den = (1/(np.pi*sigma**2))/(2.0**(np.add.outer(np.arange(m_max),np.arange(m_max))+2)*np.outer(fact,fact))
    Hkr = np.stack([H.hermval(x/(np.sqrt(2)*sigma), np.eye(m_max)[k]) for k in range(m_max)])
    env = np.exp(-x**2/(4*sigma**2))
    R = np.einsum('mn,mi,nj->ji', beta*den, Hkr, Hkr)*np.outer(env,env)
    R = R/R.max(); R[R<0] = 0
    return R

# ---------------- deconvolution baselines ----------------
otf = np.fft.fft2(np.fft.ifftshift(Sai*dx*dx))
otf_abs2 = np.abs(otf)**2
def conv(f): return np.real(np.fft.ifft2(np.fft.fft2(f)*otf))
def convT(f): return np.real(np.fft.ifft2(np.fft.fft2(f)*np.conj(otf)))

def wiener(y, lam):
    return np.real(np.fft.ifft2(np.conj(otf)/(otf_abs2+lam)*np.fft.fft2(y)))

def rl_tv(y, lam, iters):
    f = np.clip(y.copy(), 1e-6, None); eps = 1e-12
    for _ in range(iters):
        ratio = y/(conv(f)+eps)
        gx = np.gradient(f, axis=1); gy = np.gradient(f, axis=0)
        mag = np.sqrt(gx**2+gy**2)+eps
        div = np.gradient(gx/mag, axis=1)+np.gradient(gy/mag, axis=0)
        f = f*convT(ratio)/np.clip(1-lam*div, 1e-2, None)
        f = np.clip(f, 0, None)
    return f

def ista(y, lam, iters, fista=False):
    L = otf_abs2.max(); t = 0.9/L
    f = y.copy(); z = f.copy(); tk = 1.0
    for _ in range(iters):
        g = convT(conv(z)-y)
        fn = z-t*g
        fn = np.sign(fn)*np.maximum(np.abs(fn)-lam*t, 0)
        fn = np.clip(fn, 0, None)
        if fista:
            tk1 = (1+np.sqrt(1+4*tk*tk))/2
            z = fn+((tk-1)/tk1)*(fn-f); tk = tk1
        else:
            z = fn
        f = fn
    return f

def tv_grad(y, lam, iters, step=None):
    L = otf_abs2.max(); step = step or 0.9/L
    f = y.copy(); eps = 1e-12
    for _ in range(iters):
        gx = np.gradient(f, axis=1); gy = np.gradient(f, axis=0)
        mag = np.sqrt(gx**2+gy**2)+eps
        div = np.gradient(gx/mag, axis=1)+np.gradient(gy/mag, axis=0)
        f = f-step*(convT(conv(f)-y)-lam*div)
        f = np.clip(f, 0, 1.5)
    return f

def best(recs):
    out = None
    for r in recs:
        for v in [np.clip(r/ (r.max() if r.max()>0 else 1), 0, 1), (r-r.min())/(r.max()-r.min()+1e-12)]:
            m = metrics(v)
            if out is None or m['PSNR'] > out[0]['PSNR']: out = (m, v)
    return out

# ---------------- run ----------------
res = []
pms_cache = {m: pms(m) for m in (20, 35)}
for peak in [1e12, 1e14, 1e16, 1e18]:
    for seed in range(8):
        t0 = time.time()
        Yn = noisy(peak, np.random.default_rng(100*seed+int(np.log10(peak))))
        row = dict(peak=peak, seed=seed)
        entries = {}
        entries['DI'] = metrics(Yn)
        rc, rnc = vms20(Yn)
        entries['VMS20_clip'] = metrics(rc); entries['VMS20_noclip'] = metrics(rnc)
        entries['PMS20'] = metrics(pms_cache[20]); entries['PMS35'] = metrics(pms_cache[35])
        entries['Wiener'] = best([wiener(Yn, l) for l in np.logspace(-12, 0, 13)])[0]
        entries['RL_TV'] = best([rl_tv(Yn, 0.002, it) for it in (25, 50, 100, 200)])[0]
        entries['ISTA'] = best([ista(Yn, l, 200) for l in (1e-6, 1e-5, 1e-4, 1e-3)])[0]
        entries['FISTA'] = best([ista(Yn, l, 200, fista=True) for l in (1e-6, 1e-5, 1e-4, 1e-3)])[0]
        entries['TV'] = best([tv_grad(Yn, l, 300) for l in (1e-5, 1e-4, 1e-3, 1e-2)])[0]
        for k, v in entries.items():
            res.append({**row, 'method': k, **v})
        print(peak, seed, 'done %.1fs' % (time.time()-t0), flush=True)

import pandas as pd
df = pd.DataFrame(res)
df.to_excel(OUT + '/fig3_benchmark_raw.xlsx', index=False)
agg = df.groupby(['peak', 'method']).agg(['mean', 'std']).round(4)
agg.to_excel(OUT + '/fig3_benchmark_summary.xlsx')
print(agg[['PSNR', 'SSIM']].to_string())

# panel figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
order = ['DI','VMS20_clip','VMS20_noclip','PMS20','PMS35','Wiener','RL_TV','ISTA','FISTA','TV']
fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
for ax, met, better in zip(axs, ['PSNR','SSIM','RSE','RSP'], ['higher','higher','lower','higher']):
    for meth in order:
        sub = df[df.method == meth].groupby('peak')[met]
        ax.errorbar(sub.mean().index, sub.mean().values, yerr=sub.std().values,
                    marker='o', ms=3, capsize=2, label=meth)
    ax.set_xscale('log'); ax.set_xlabel('peak photons/pixel')
    ax.set_title(f'{met} ({better} is better)')
axs[0].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(OUT + '/fig3_benchmark_panel.png', dpi=160)
print('ALL DONE')
