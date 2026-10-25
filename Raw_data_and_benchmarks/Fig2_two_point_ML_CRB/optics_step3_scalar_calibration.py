"""Step 3: reproduction of the Fig. 2b scalar calibration from the raw fields (consistency check)."""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import math, json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import hermite as scipy_hermite
import pandas as pd

PTS = os.path.join(BASE, 'data', 'optics_fields')
OUT = os.path.join(BASE, 'out')
shifts = json.load(open(os.path.join(BASE, 'data', 'fig2_tilemap.json')))['shifts']

w = 0.15
xs = np.linspace(-1, 1, 900); dxs = xs[1]-xs[0]
XS, YS = np.meshgrid(xs, xs)
def Sai_n(X, Y, w): return (1/(2*np.pi*w**2)**0.5)*np.exp(-(X**2+Y**2)/(4*w**2))
def basis(m, n):
    Hm = scipy_hermite(m)(XS/(2**0.5*w)); Hn = scipy_hermite(n)(YS/(2**0.5*w))
    return (1/np.sqrt(2**(m+n)*math.factorial(m)*math.factorial(n)))*Hm*Hn*Sai_n(XS,YS,w)
AB = {k: np.abs(basis(*mn)) for k, mn in {'00':(0,0),'10':(1,0),'01':(0,1)}.items()}

def shift_center(mat, threshold=0.5):
    hp = np.argwhere(mat > threshold)
    if hp.size == 0: return mat
    sx = int((900+1)//2 - hp[:,0].mean()); sy = int((900+1)//2 - hp[:,1].mean())
    return np.roll(np.roll(mat, sx, axis=0), sy, axis=1)

def mm(a): return (a-a.min())/(a.max()-a.min())

rec = []
for fi in range(1, 11):
    fld = np.array(Image.open(f'{PTS}/{fi}.png').convert('L'), np.float32)/255.0
    sh = shifts[fi-1]
    fld = np.roll(fld, (sh[0], sh[1]), axis=(0,1))
    for tr in range(4):
        for tc in range(5):
            p = fld[tr*900:(tr+1)*900, tc*900:(tc+1)*900]
            p = mm(p); p = shift_center(p)
            p = gaussian_filter(p, sigma=1.0)
            p = mm(p); p = shift_center(p)
            i = {k: float(np.sum(AB[k]*np.abs(p))*dxs*dxs) for k in AB}
            rec.append(dict(field=fi, gap=round(0.05+0.01*(tr*5+tc), 2),
                            I00=i['00'], I10=i['10'], I01=i['01'], p00=float(p[450,450])))
    print('field', fi, flush=True)

df = pd.DataFrame(rec)
g00 = df.groupby('gap')['I00'].mean(); max_I00 = g00.max()
df['I10n'] = df.I10/max_I00; df['I01n'] = df.I01/max_I00
df.to_csv(OUT+'/fig2_optics_scalar.csv', index=False)
t = df.groupby('gap').agg(I10n_m=('I10n','mean'), I10n_s=('I10n','std'),
                          I01n_m=('I01n','mean'), I01n_s=('I01n','std'),
                          p00_m=('p00','mean'), p00_s=('p00','std'))
print(t.round(4).to_string())
fig2b_vals = {0.05:0.714161,0.07:0.737810,0.11:0.754196,0.13:0.786447,0.16:0.861119,
          0.18:0.912884,0.20:0.978305,0.21:1.000000,0.22:0.987484,0.24:0.967819}
sub = t.reindex([round(k,2) for k in fig2b_vals])
for col in ['I10n_m','I01n_m']:
    c = np.corrcoef(sub[col].values, list(fig2b_vals.values()))[0,1]
    print(col, 'corr vs Fig. 2b values:', round(c,4), 'range %.3f-%.3f' % (sub[col].min(), sub[col].max()))
