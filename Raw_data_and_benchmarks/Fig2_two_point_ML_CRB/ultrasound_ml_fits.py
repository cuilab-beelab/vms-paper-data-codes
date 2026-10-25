"""Ultrasound Fig. 2d: ML two-point fits + CRB on the 30 measured C-scan maps."""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, glob, re, os
import pandas as pd
from scipy.optimize import least_squares

UD = os.path.join(BASE, 'data', 'ultrasound_cscan')
OUT = os.path.join(BASE, 'out')
step = 0.05  # mm per pixel (scan step)

files = sorted(glob.glob(UD + '/*.xlsx'))
rows = []
for f in files:
    m = re.match(r'([\d.]+)_(\d+)', os.path.basename(f))
    gap, rep = float(m.group(1)), int(m.group(2))
    M = pd.read_excel(f, header=None).values.astype(float)
    rows.append((gap, rep, M))
print('maps:', len(rows), 'shape', rows[0][2].shape)

def make_grid(shape):
    n0, n1 = shape
    u = np.arange(n1) - n1/2 + .5
    v = np.arange(n0) - n0/2 + .5
    return np.meshgrid(u, v)

def model(p, U, V, n0, n1):
    b0, bx, by, A1, A2, cx, cy, th, D, s = p
    ux, uy = np.cos(th), np.sin(th)
    x1, y1 = cx-D/2*ux, cy-D/2*uy
    x2, y2 = cx+D/2*ux, cy+D/2*uy
    g1 = np.exp(-((U-x1)**2+(V-y1)**2)/(2*s*s))
    g2 = np.exp(-((U-x2)**2+(V-y2)**2)/(2*s*s))
    return b0 + bx*U/n1 + by*V/n0 - A1*g1 - A2*g2   # holes = dips in reflection

def fit_map(M, D0_px):
    n0, n1 = M.shape
    U, V = make_grid(M.shape)
    bg = np.median(M); amp = bg - M.min()
    t = np.clip(bg - M - 0.2*amp, 0, None)
    cy0 = float((V*t).sum()/max(t.sum(),1e-9)); cx0 = float((U*t).sum()/max(t.sum(),1e-9))
    best = None
    for Din in [max(D0_px,1.5), max(0.5*D0_px,0.8), 1.5*D0_px+1]:
      for th0 in [0.0, np.pi/2]:
        p0 = [bg,0,0,amp,amp,cx0,cy0,th0,Din,10.0]
        lo = [0,-1,-1,0,0,cx0-15,cy0-15,th0-0.9,0.0,4]
        hi = [1, 1, 1,1,1,cx0+15,cy0+15,th0+0.9,45,25]
        try:
            r = least_squares(lambda p:(model(p,U,V,n0,n1)-M).ravel(), p0, bounds=(lo,hi), x_scale='jac', max_nfev=300)
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    r = best
    sig = float((model(r.x,U,V,n0,n1)-M).std())
    J = np.empty((M.size,10)); eps = np.maximum(1e-5,1e-4*np.abs(r.x))
    for i in range(10):
        pp = r.x.copy(); pp[i]+=eps[i]
        J[:,i]=((model(pp,U,V,n0,n1)-model(r.x,U,V,n0,n1))/eps[i]).ravel()
    try:
        cov = sig**2*np.linalg.inv(J.T@J+1e-12*np.eye(10))
        crb = float(np.sqrt(max(cov[8,8],0)))
    except np.linalg.LinAlgError: crb = np.nan
    return r.x, sig, crb

out = []
for gap, rep, M in rows:
    p, sig, crb = fit_map(M, D0_px=gap/step)
    out.append(dict(gap=gap, rep=rep, D_px=p[8], D_mm=p[8]*step, theta=p[7],
                    s_px=p[9], s_mm=p[9]*step, A1=p[3], A2=p[4], sig=sig,
                    crb_px=crb, crb_mm=crb*step))
    print('gap %.3f rep %d: D=%.3f mm  s=%.2f mm  crb=%.4f mm' % (gap, rep, p[8]*step, p[9]*step, crb*step), flush=True)

df = pd.DataFrame(out)
df.to_csv(OUT+'/fig2_ultrasound_fits.csv', index=False)
t = df.groupby('gap').agg(D_mm_mean=('D_mm','mean'), D_mm_std=('D_mm','std'),
                          crb_mm=('crb_mm','mean'), s_mm=('s_mm','mean'), n=('rep','count'))
print(t.round(4).to_string())
t.to_csv(OUT+'/fig2_ultrasound_summary.csv')
