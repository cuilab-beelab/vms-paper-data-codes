"""Step 2: per-tile-per-field ML fits + reproduction of the Fig. 2b scalar calibration.
Outputs: out/fig2_optics_fits.csv (one row per tile per field) and summary prints.
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import math, json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.special import hermite as scipy_hermite

PTS = os.path.join(BASE, 'data', 'optics_fields')
OUT = os.path.join(BASE, 'out')
shifts = json.load(open(os.path.join(BASE, 'data', 'fig2_tilemap.json')))['shifts']

# ---- Fig. 2b scalar photocurrent (same constants as the calibration) ----
w = 0.15
xs = np.linspace(-1, 1, 900); dxs = xs[1] - xs[0]
XS, YS = np.meshgrid(xs, xs)
def Sai_n(X, Y, w):
    return (1/(2*np.pi*w**2)**0.5)*np.exp(-(X**2+Y**2)/(4*w**2))
def basis(m, n):
    Hm = scipy_hermite(m)(XS/(2**0.5*w)); Hn = scipy_hermite(n)(YS/(2**0.5*w))
    return (1/np.sqrt(2**(m+n)*math.factorial(m)*math.factorial(n)))*Hm*Hn*Sai_n(XS, YS, w)
B00, B10, B01 = np.abs(basis(0,0)), np.abs(basis(1,0)), np.abs(basis(0,1))

def scalar_photocurrent(patch900):
    p = gaussian_filter(patch900, sigma=1.0)
    p = p/p.max()
    i00 = float(np.sum(B00*np.abs(p))*dxs*dxs)
    i10 = float(np.sum(B10*np.abs(p))*dxs*dxs)
    i01 = float(np.sum(B01*np.abs(p))*dxs*dxs)
    return i00, i10, i01

# ---- ML fit on downsampled patch ----
F = 3  # downsample factor -> 300x300, pixel = 3 original px
def ds(a, f=F):
    return a[:a.shape[0]//f*f, :a.shape[1]//f*f].reshape(a.shape[0]//f, f, a.shape[1]//f, f).mean(axis=(1,3))

nn = 900//F
u = np.arange(nn) - nn/2 + .5
U, V = np.meshgrid(u, u)

def model(p):
    b0, bx, by, A1, A2, cx, cy, th, D, s = p
    ux, uy = np.cos(th), np.sin(th)
    x1, y1 = cx - D/2*ux, cy - D/2*uy
    x2, y2 = cx + D/2*ux, cy + D/2*uy
    g1 = np.exp(-((U-x1)**2+(V-y1)**2)/(2*s*s))
    g2 = np.exp(-((U-x2)**2+(V-y2)**2)/(2*s*s))
    return b0 + bx*U/nn + by*V/nn + A1*g1 + A2*g2

def fit_patch(pd, D0):
    bg = np.median(pd); amp = pd.max() - bg
    # centroid of bright region
    t = np.clip(pd - bg - 0.3*amp, 0, None)
    cy0 = float((V*t).sum()/t.sum()); cx0 = float((U*t).sum()/t.sum())
    best = None
    for Dinit in [max(D0, 2.0), max(D0*0.5, 1.0), D0*1.5 + 2]:
        p0 = [bg, 0, 0, amp, amp, cx0, cy0, 0.0, Dinit, 18.0]
        lo = [0, -50, -50, 0, 0, cx0-40, cy0-40, -0.8, 0.0, 5]
        hi = [255, 50, 50, 400, 400, cx0+40, cy0+40, 0.8, 90, 60]
        try:
            r = least_squares(lambda p: (model(p)-pd).ravel(), p0, bounds=(lo, hi), x_scale='jac', max_nfev=200)
        except Exception:
            continue
        if best is None or r.cost < best.cost: best = r
    r = best
    res = model(r.x) - pd
    sig = float(res.std())
    # CRB for D from numeric Jacobian at solution
    J = np.empty((pd.size, 10))
    eps = np.maximum(1e-4, 1e-4*np.abs(r.x))
    for i in range(10):
        pp = r.x.copy(); pp[i] += eps[i]
        J[:, i] = ((model(pp) - model(r.x))/eps[i]).ravel()
    JtJ = J.T @ J
    try:
        cov = sig**2*np.linalg.inv(JtJ + 1e-12*np.eye(10))
        crbD = float(np.sqrt(max(cov[8, 8], 0)))
    except np.linalg.LinAlgError:
        crbD = np.nan
    return r.x, sig, crbD

rows_out = []
tiles = [(r, c) for r in range(4) for c in range(5)]
for fi in range(1, 11):
    fld = np.array(Image.open(f'{PTS}/{fi}.png').convert('L'), float)
    sh = shifts[fi-1]
    fld = np.roll(fld, (sh[0], sh[1]), axis=(0, 1))
    for (tr, tc) in tiles:
        patch = fld[tr*900:(tr+1)*900, tc*900:(tc+1)*900]
        i00, i10, i01 = scalar_photocurrent(patch)
        pd = ds(patch)
        p, sig, crbD = fit_patch(pd, D0=10.0)
        rows_out.append(dict(field=fi, tile_r=tr, tile_c=tc,
                             I00=i00, I10=i10, I01=i01, r10=i10/i00, r01=i01/i00,
                             D_px=p[8]*F, theta=p[7], s_px=p[9]*F, A1=p[3], A2=p[4],
                             bg=p[0], sig_res=sig, crbD_px=crbD*F))
    print('field', fi, 'done', flush=True)

import pandas as pd
df = pd.DataFrame(rows_out)
df.to_csv(OUT + '/fig2_optics_fits.csv', index=False)
g = df.groupby(['tile_r', 'tile_c'])
summ = g.agg(D_px_mean=('D_px', 'mean'), D_px_std=('D_px', 'std'),
             s_px=('s_px', 'mean'), r10_mean=('r10', 'mean'), r10_std=('r10', 'std'),
             crbD_px=('crbD_px', 'mean'), A2_med=('A2', 'median'), A1_med=('A1', 'median'))
print(summ.round(2).to_string())
summ.to_csv(OUT + '/fig2_optics_tile_summary.csv')
