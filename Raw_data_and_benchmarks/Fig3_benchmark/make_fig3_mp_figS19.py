"""Main-text Fig. 3 m-p and SI Fig. S19 from out/fig3_benchmark_raw.xlsx."""
import math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, '..', 'figures')
OUT = os.path.join(BASE, 'out')
df = pd.read_excel(OUT + '/fig3_benchmark_raw.xlsx')
# ================= fig3_mp v2 =================
df = pd.read_excel(OUT + '/fig3_benchmark_raw.xlsx')
d18 = df[df.peak == 1e18]
order = ['DI','PMS20','PMS35','VMS20_clip','VMS20_noclip','Wiener','RL_TV','ISTA','FISTA','TV']
labels = ['DI','PMS20','PMS35','VMS20','VMS20°','Wiener','RL','ISTA','FISTA','TV']
colors = ['#1a2a4f','#2ca02c','#7f7f7f','#d62728','#d62728','#6a3d9a','#e377c2','#8c564b','#bcbd22','#17becf']
open_face = [False]*4 + [True] + [False]*5
mets = [('PSNR','PSNR (dB)',None),('SSIM','SSIM',(0,0.8)),('RSE','RSE',(0,0.8)),('RSP','RSP',(0,1.05))]
plt.rcParams.update({'font.size': 10.5})
fig, axs = plt.subplots(1, 4, figsize=(11.4, 2.55))
for ax,(met,yl,ylim),let in zip(axs, mets, 'mnop'):
    g = d18.groupby('method')[met]
    for i, meth in enumerate(order):
        m, s = g.mean()[meth], g.std()[meth]
        fc = 'white' if open_face[i] else colors[i]
        ax.errorbar(i, m, yerr=s, fmt='o', ms=5.5, mfc=fc, mec=colors[i], ecolor=colors[i], capsize=2, lw=1.1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=7.6, rotation=55, ha='right')
    ax.get_xticklabels()[3].set_color('#d62728'); ax.get_xticklabels()[4].set_color('#d62728')
    ax.set_ylabel(yl)
    if ylim: ax.set_ylim(*ylim)
    if met == 'PSNR': ax.set_ylim(8, 16)
    ax.set_xlim(-0.7, len(order)-0.3)
    ax.text(-0.14, 1.05, let, transform=ax.transAxes, fontweight='bold', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_mp.png'), dpi=300); plt.close(fig)

# ================= figS19 v2 =================
plt.rcParams.update({'font.size': 13})
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, met, yl in zip(axs, ['PSNR','SSIM','RSE','RSP'], ['PSNR (dB)','SSIM','RSE','RSP']):
    for meth in order:
        sub = df[df.method == meth].groupby('peak')[met]
        ax.errorbar(sub.mean().index, sub.mean().values, yerr=sub.std().values,
                    marker='o', ms=3.5, capsize=2, label=meth)
    ax.set_xscale('log'); ax.set_xlabel('Peak photons per pixel')
    ax.set_ylabel(yl)
axs[0].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'figS19.png'), dpi=300)
print('v2 figures done')
