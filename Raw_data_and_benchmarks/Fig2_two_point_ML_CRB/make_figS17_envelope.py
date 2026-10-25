"""SI Fig. S17: envelope-resolution analysis and estimator calibration on the measured
two-point data (Supplementary Section S14), with the quoted summary numbers printed."""
import math, glob, re, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.special import hermite as scipy_hermite
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
UD = os.path.join(BASE, 'data', 'ultrasound_cscan')
PTS = os.path.join(BASE, 'data', 'optics_fields')
OUT = os.path.join(BASE, 'out')
TILEMAP = os.path.join(BASE, 'data', 'fig2_tilemap.json')
plt.rcParams.update({'font.size': 12})

# ---------------- ultrasound preprocessing (calibration pipeline) ----------------
def prep_us(M):
    E = gaussian_filter(M.astype(float), sigma=1)
    E = (E - E.min())/(E.max() - E.min())
    E = 1 - E
    # center on blob (generalized to any map shape)
    hp = np.argwhere(E > 0.5)
    cy, cx = E.shape[0]//2, E.shape[1]//2
    E = np.roll(E, (int(cy - hp[:,0].mean()), int(cx - hp[:,1].mean())), axis=(0,1))
    # crop central 81x81
    r0, c0 = E.shape[0]//2 - 40, E.shape[1]//2 - 40
    E = E[r0:r0+81, c0:c0+81]
    E = (E - E.min())/(E.max() - E.min())
    return E

step = 0.05  # mm/px
maps = {}
for f in sorted(glob.glob(UD + '/*.xlsx')):
    m = re.match(r'([\d.]+)_(\d+)', os.path.basename(f))
    gap, rep = float(m.group(1)), int(m.group(2))
    maps.setdefault(gap, []).append(prep_us(pd.read_excel(f, header=None).values))

def dip_contrast(prof, D_px, x0):
    """P = mean of the two window maxima around +-D/2 from center x0; V = min between."""
    n = len(prof)
    w = max(2, int(round(D_px/4)))
    i1, i2 = int(round(x0 - D_px/2)), int(round(x0 + D_px/2))
    p1 = prof[max(i1-w,0):i1+w+1].max(); p2 = prof[max(i2-w,0):min(i2+w+1,n)].max()
    V = prof[min(i1,i2):max(i1,i2)+1].min()
    P = 0.5*(p1+p2)
    return max((P - V)/P, 0.0)

us_gaps = sorted(maps)
us_C = {g: [dip_contrast(E[40,:], g/step, 40) for E in maps[g]] for g in us_gaps}
print('US dip contrast (mean %):')
for g in us_gaps: print('  %.3f mm: %.1f%%  (n=%d)' % (g, 100*np.mean(us_C[g]), len(us_C[g])))

# FWHM of a single-blob profile (smallest gap ~ PSF)
p = maps[0.085][0][40,:]
above = np.where(p >= 0.5)[0]
fwhm = (above[-1]-above[0])*step
print('measured FWHM ~ %.2f mm (quoted 1.55)' % fwhm)

# ultrasound photocurrent from maps (same mode width as the calibration)
a = 0.5*2.54/100/2; lam = 1500/3.5e6; z0 = 0.83*2.54/100
FWHM_t = 0.61*lam*z0/a; sigma_t = 2.25*FWHM_t; w_us = 0.325*sigma_t
xg = np.linspace(-0.00405/2, 0.00405/2, 81); dxg = (xg.max()-xg.min())/81
XG, YG = np.meshgrid(xg, xg)
def Sai(x, y, w): return (1/(2*np.pi*w**2)**0.5)*np.exp(-(x**2+y**2)/(4*w**2))
def pc(m, n, w, E):
    B = (1/(2**(m+n)*math.factorial(m)*math.factorial(n))**0.5)*scipy_hermite(m)(XG/(2**0.5*w))*scipy_hermite(n)(YG/(2**0.5*w))*Sai(XG,YG,w)
    return np.sum(abs(B*E))*dxg*dxg
rows = []
for g in us_gaps:
    for E in maps[g]:
        rows.append(dict(gap=g, I00=pc(0,0,w_us,E), I10=pc(1,0,w_us,E), c=E[40,40]))
ru = pd.DataFrame(rows)
ru['I10n'] = ru.I10/ru.groupby('gap').I00.transform('mean').max()
# normalization: I10 / max(I00), as in the calibration
ru['I10n'] = ru.I10/ru.I00.max()
ru['cn'] = ru.c  # already 0..1
tu = ru.groupby('gap').agg(I10m=('I10n','mean'), I10s=('I10n','std'), cm=('cn','mean'), cs=('cn','std'))

# ---------------- optics ----------------
shifts = json.load(open(TILEMAP))['shifts']
def mm01(a): return (a-a.min())/(a.max()-a.min())
def shift_center(mat, threshold=0.5):
    hp = np.argwhere(mat > threshold)
    if hp.size == 0: return mat
    sx = int((901)//2 - hp[:,0].mean()); sy = int((901)//2 - hp[:,1].mean())
    return np.roll(np.roll(mat, sx, axis=0), sy, axis=1)
scale_o = 916.2  # px/mm
opt_prof = {}; opt_C = {g: [] for g in [round(0.05+0.01*k,2) for k in range(20)]}
for fi in range(1, 11):
    fld = np.array(Image.open(f'{PTS}/{fi}.png').convert('L'), np.float32)/255.0
    sh = shifts[fi-1]
    fld = np.roll(fld, (sh[0], sh[1]), axis=(0,1))
    for tr in range(4):
        for tc in range(5):
            g = round(0.05+0.01*(tr*5+tc), 2)
            p = mm01(fld[tr*900:(tr+1)*900, tc*900:(tc+1)*900])
            p = shift_center(p); p = gaussian_filter(p, sigma=1.0); p = mm01(p); p = shift_center(p)
            prof = p[450, :]
            opt_C[g].append(dip_contrast(prof, g*scale_o, 450))
            if fi == 1: opt_prof[g] = prof
oc_m = {g: np.mean(v) for g, v in opt_C.items()}
oc_s = {g: np.std(v) for g, v in opt_C.items()}
print('Optics dip contrast: 0.18 -> %.1f%%  0.20 -> %.1f%%  (threshold 26.4%%)' % (100*oc_m[0.18], 100*oc_m[0.20]))
print('crossing between 0.18 and 0.20:', oc_m[0.18] < 0.264 < oc_m[0.20])

so = pd.read_csv(OUT + '/fig2_optics_scalar.csv')
tso = so.groupby('gap').agg(I01m=('I01n','mean'), I01s=('I01n','std'), p00m=('p00','mean'), p00s=('p00','std'))

# ---------------- figure ----------------
fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
ax = axs[0,0]
for g, stl in [(0.25,'-'), (0.7,'--'), (1.4,':')]:
    ax.plot(np.arange(81)*step - 2.0, maps[g][0][40,:], 'C0', ls=stl, lw=1.6, label=f'{g} mm')
ax.set_xlabel('Position (mm)'); ax.set_ylabel('Normalized envelope')
ax.set_title('Ultrasound: measured profiles', fontsize=13); ax.legend(fontsize=10)
ax = axs[0,1]
gg = np.array(us_gaps)
ax.errorbar(gg, [100*np.mean(us_C[g]) for g in gg], yerr=[100*np.std(us_C[g]) for g in gg],
            fmt='o-', ms=5, capsize=2, color='C0')
ax.axhline(26.4, color='grey', ls='--', lw=1.2)
ax.text(0.35, 27.5, 'Rayleigh (26.4%)', color='grey', fontsize=10.5)
ax.set_ylim(-2, 50); ax.set_xlabel('Separation (mm)'); ax.set_ylabel('Dip contrast (%)')
ax.set_title('Ultrasound: unresolved at all separations', fontsize=13)
ax = axs[0,2]
fig2d_vals = {0.085:(0.46437,0.002363),0.15:(0.480664,0.001559),0.25:(0.523928,0.003757),
              0.5:(0.59941,0.003408),0.7:(0.688629,0.003002),0.9:(0.734939,0.005223),
              1.2:(0.867176,0.002946),1.4:(0.921347,0.004)}
ug = np.array(list(fig2d_vals)); uv = np.array([v[0] for v in fig2d_vals.values()]); us_ = np.array([v[1] for v in fig2d_vals.values()])
ax.errorbar(ug, uv, yerr=us_, fmt='s-', ms=5.5, capsize=2, color='#c0392b', label='Virtual photocurrent $I_{10}$')
ax.errorbar(tu.index, tu.cm, yerr=tu.cs, fmt='o-', ms=5.5, capsize=2, color='#2980b9', label='Central pixel')
ax.set_xlabel('Separation (mm)'); ax.set_ylabel('Normalized signal')
ax.set_title('Ultrasound: estimator calibration', fontsize=13); ax.legend(fontsize=10)
ax = axs[1,0]
xo = (np.arange(900) - 450)/scale_o
for g, stl in [(0.11,'-'), (0.18,'--'), (0.24,':')]:
    ax.plot(xo, opt_prof[g], 'C0', ls=stl, lw=1.1, label=f'{g} mm')
ax.set_xlim(-0.35, 0.35)
ax.set_xlabel('Position (mm)'); ax.set_ylabel('Normalized intensity')
ax.set_title('Optics: measured profiles', fontsize=13); ax.legend(fontsize=10)
ax = axs[1,1]
gsel = sorted(fig_g for fig_g in [0.05,0.07,0.11,0.13,0.16,0.18,0.20,0.21,0.22,0.24])
ax.errorbar(gsel, [100*oc_m[g] for g in gsel], yerr=[100*oc_s[g] for g in gsel],
            fmt='o-', ms=5, capsize=2, color='C0')
ax.axhline(26.4, color='grey', ls='--', lw=1.2)
ax.set_xlabel('Separation (mm)'); ax.set_ylabel('Dip contrast (%)')
ax.set_title('Optics: Rayleigh crossing near 0.20 mm', fontsize=13)
ax = axs[1,2]
ax.errorbar(tso.index, tso.I01m, yerr=tso.I01s, fmt='s-', ms=5, capsize=2, color='#c0392b', label='Virtual photocurrent $I_{01}$')
ax.errorbar(tso.index, tso.p00m, yerr=tso.p00s, fmt='o-', ms=5, capsize=2, color='#2980b9', label='Central pixel')
ax.set_xlabel('Separation (mm)'); ax.set_ylabel('Normalized signal')
ax.set_title('Optics: estimator calibration', fontsize=13); ax.legend(fontsize=10)
for ax, let in zip(axs.ravel(), 'abcdef'):
    ax.text(-0.14, 1.06, let, transform=ax.transAxes, fontweight='bold', fontsize=15)
fig.tight_layout(); fig.savefig(os.path.join(BASE, '..', 'figures', 'figS17.png'), dpi=300)
print('saved figures/figS17.png')
