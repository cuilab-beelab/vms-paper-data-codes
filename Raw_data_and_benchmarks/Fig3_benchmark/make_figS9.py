"""SI Fig. S9 (DI / PMS20 / PMS35 / VMS20 / DC-RL at peak 1e18) from the benchmark."""
import math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, '..', 'figures')
OUT = os.path.join(BASE, 'out')
df = pd.read_excel(OUT + '/fig3_benchmark_raw.xlsx')
d18 = df[df.peak == 1e18]
mets = [('PSNR','PSNR (dB)',None),('SSIM','SSIM',(0,0.8)),('RSE','RSE',(0,0.8)),('RSP','RSP',(0,1.05))]
# ================= figS9 v2 (same five readouts, bench values) =================
sel = ['DI','PMS20','PMS35','VMS20_clip','RL_TV']
lab9 = ['DI','PMS\n20','PMS\n35','VMS\n20','DC\nRL']
col9 = ['#1a2a4f','#2ca02c','#145a32','#d62728','#e67e22']
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.0))
for ax,(met,yl,_),let in zip(axs, mets, 'abcd'):
    g = d18.groupby('method')[met]
    for i, meth in enumerate(sel):
        m, s = g.mean()[meth], g.std()[meth]
        ax.errorbar(i, m, yerr=s, fmt='o', ms=7, mfc=col9[i], mec=col9[i], ecolor=col9[i], capsize=2)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(lab9, fontsize=10.5)
    ax.set_ylabel(yl)
    if met == 'PSNR': ax.set_ylim(0, 18)
    elif met == 'SSIM': ax.set_ylim(0, 0.6)
    elif met == 'RSE': ax.set_ylim(0, 0.7)
    elif met == 'RSP': ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(sel)-0.4)
    ax.text(-0.22, 1.02, let, transform=ax.transAxes, fontweight='bold', fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'figS9.png'), dpi=300); plt.close(fig)

