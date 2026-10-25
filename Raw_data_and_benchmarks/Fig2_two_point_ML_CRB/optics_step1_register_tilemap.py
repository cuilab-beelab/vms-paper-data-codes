"""Step 1: register the 10 field acquisitions; template-match labeled crops -> tile map."""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, json, sys
from PIL import Image

PTS = os.path.join(BASE, 'data', 'optics_fields')
MIC = os.path.join(BASE, 'data', 'optics_labeled_crops')
OUT = os.path.join(BASE, 'out')

fields = []
for i in range(1, 11):
    fields.append(np.array(Image.open(f'{PTS}/{i}.png').convert('L'), float))
fields = np.array(fields)
print('fields', fields.shape)

# --- registration via FFT cross-correlation against field 1 (integer shift) ---
def xcorr_shift(a, b):
    A = np.fft.rfft2(a - a.mean()); B = np.fft.rfft2(b - b.mean())
    c = np.fft.irfft2(A * np.conj(B), s=a.shape)
    idx = np.unravel_index(np.argmax(c), c.shape)
    sh = [int(idx[0] if idx[0] <= a.shape[0]//2 else idx[0]-a.shape[0]),
          int(idx[1] if idx[1] <= a.shape[1]//2 else idx[1]-a.shape[1])]
    return sh, c.max()/(a.std()*b.std()*a.size)

shifts = [[0, 0]]
for i in range(1, 10):
    sh, cc = xcorr_shift(fields[0], fields[i])
    shifts.append(sh)
    print(f'field {i+1}: shift {sh} ncc~{cc:.3f}')

# --- template match the labeled crops into field 1 (downsample 4x for speed) ---
def ds(a, f=4): return a[:a.shape[0]//f*f, :a.shape[1]//f*f].reshape(a.shape[0]//f, f, a.shape[1]//f, f).mean(axis=(1,3))
F = ds(fields[0]); Fz = F - F.mean()
FA = np.fft.rfft2(Fz)
labels = ['0.11','0.13','0.16','0.18','0.20','0.21','0.22','0.24','0.05','0.07']
tilemap = {}
for lab in labels:
    T = np.array(Image.open(f'{MIC}/{lab}.png').convert('L'), float)
    Td = ds(T); Tz = Td - Td.mean()
    pad = np.zeros_like(F); pad[:Tz.shape[0], :Tz.shape[1]] = Tz
    c = np.fft.irfft2(FA * np.conj(np.fft.rfft2(pad)), s=F.shape)
    ij = np.unravel_index(np.argmax(c), c.shape)
    # normalized score
    sc = c[ij] / (np.linalg.norm(Tz) * np.linalg.norm(Fz) / np.sqrt(Fz.size) + 1e-9)
    tilemap[lab] = dict(row=int(ij[0]*4), col=int(ij[1]*4), score=float(c[ij]))
    print(lab, '-> top-left px', (ij[0]*4, ij[1]*4), 'peak %.3g' % c[ij])

json.dump(dict(shifts=shifts, tilemap=tilemap), open(os.path.join(BASE, 'data', 'fig2_tilemap.json'), 'w'), indent=1)

# verification montage: draw matched boxes on field 1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(fields[0], cmap='gray')
for lab, d in tilemap.items():
    ax.add_patch(plt.Rectangle((d['col'], d['row']), 900, 900, fill=False, lw=1.2))
    ax.text(d['col']+20, d['row']+120, lab, fontsize=8, color='yellow')
ax.set_title('field 1 with matched labeled crops'); ax.axis('off')
fig.tight_layout(); fig.savefig(OUT + '/fig2_tilemap_check.png', dpi=110)
print('saved tilemap')
