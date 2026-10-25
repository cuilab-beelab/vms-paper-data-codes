"""Main-text Fig. 2 panels f,g from the fit/CRB tables in out/."""
import math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, '..', 'figures')
OUT = os.path.join(BASE, 'out')
# ================= fig2_fg v2 =================
plt.rcParams.update({'font.size': 10.5, 'axes.linewidth': 0.8})
fo = pd.read_csv(OUT + '/fig2_optics_fits.csv')
fo['gap'] = (0.05 + 0.01*(fo.tile_r*5 + fo.tile_c)).round(2)
go = fo.groupby('gap').agg(D_px_s=('D_px','std'), crb=('crbD_px','mean'))
scale = 916.2
fu = pd.read_csv(OUT + '/fig2_ultrasound_summary.csv', index_col=0)
fu = fu[fu.index >= 0.15]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6))
ax = axs[0]
ax.semilogy(go.index, go.D_px_s/scale*1000, 'o-', ms=4, lw=1.1, color='#1f77b4', label='Two-point fit, repeatability (N = 10)')
ax.semilogy(go.index, go.crb/scale*1000, 's--', ms=3.5, lw=1.1, color='#2ca02c', label='Cramér–Rao bound (direct image)')
ax.axhline(25, color='#d62728', lw=1.3, ls=':', label='Virtual photocurrent, $\\delta D$ ($D_{\\mathrm{eff}}$/10)')
ax.axvspan(0.045, 0.095, color='orange', alpha=0.12)
ax.text(0.048, 38, 'fit floor', fontsize=9)
ax.set_xlabel('Separation (mm)'); ax.set_ylabel('$\\delta D$ (µm)')
ax.set_ylim(0.2, 80); ax.legend(fontsize=7.5, loc='center right')
ax.text(-0.17, 1.04, 'f', transform=ax.transAxes, fontweight='bold', fontsize=13)
ax = axs[1]
ax.semilogy(fu.index, fu.D_mm_std*1000, 'o-', ms=4, lw=1.1, color='#1f77b4', label='Two-point fit, repeatability (N = 4)')
ax.semilogy(fu.index, fu.crb_mm*1000, 's--', ms=3.5, lw=1.1, color='#2ca02c', label='Cramér–Rao bound (direct map)')
ax.axhline(14, color='#d62728', lw=1.3, ls=':', label='Virtual photocurrent, $\\delta D$ ($\\lambda$/31)')
ax.set_xlabel('Separation (mm)'); ax.set_ylabel('$\\delta D$ (µm)')
ax.set_ylim(1, 30); ax.legend(fontsize=7.5, loc='upper right')
ax.text(-0.17, 1.04, 'g', transform=ax.transAxes, fontweight='bold', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_fg.png'), dpi=300); plt.close(fig)

