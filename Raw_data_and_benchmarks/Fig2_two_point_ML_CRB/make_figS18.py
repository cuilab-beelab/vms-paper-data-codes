"""SI Fig. S18: scalar-calibration reproduction, two-point fits and CRB (both domains)."""
import math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, '..', 'figures')
OUT = os.path.join(BASE, 'out')
fo = pd.read_csv(OUT + '/fig2_optics_fits.csv')
fo['gap'] = (0.05 + 0.01*(fo.tile_r*5 + fo.tile_c)).round(2)
# ================= figS18 v2 =================
plt.rcParams.update({'font.size': 12})
so = pd.read_csv(OUT + '/fig2_optics_scalar.csv')
to = so.groupby('gap').agg(I10n_m=('I10n','mean'), I10n_s=('I10n','std'))
go2 = fo.groupby('gap').agg(D_px_m=('D_px','mean'), D_px_s=('D_px','std'), crb=('crbD_px','mean'))
selg = go2.index >= 0.10
gv = np.array(go2.index); dv = go2.D_px_m.values
scale2 = float(np.sum(dv[selg]*gv[selg]) / np.sum(gv[selg]**2))
D_mm = go2.D_px_m/scale2; D_sd = go2.D_px_s/scale2; crb_mm = go2.crb/scale2
fig2b_vals = {0.05:(0.714161,0.009687),0.07:(0.737810,0.017452),0.11:(0.754196,0.004203),
            0.13:(0.786447,0.026989),0.16:(0.861119,0.017898),0.18:(0.912884,0.011923),
            0.20:(0.978305,0.030395),0.21:(1.0,0.022847),0.22:(0.987484,0.074395),0.24:(0.967819,0.077661)}
fu2 = pd.read_csv(OUT + '/fig2_ultrasound_summary.csv', index_col=0)
fu2 = fu2[fu2.index >= 0.15]
fig2d_vals = {0.085:(0.46437,0.002363),0.15:(0.480664,0.001559),0.25:(0.523928,0.003757),
            0.5:(0.59941,0.003408),0.7:(0.688629,0.003002),0.9:(0.734939,0.005223),
            1.2:(0.867176,0.002946),1.4:(0.921347,0.004)}
fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.2))
ax = axs[0,0]
ax.errorbar(to.index, to.I10n_m, yerr=to.I10n_s, fmt='o-', ms=4, lw=1.1, capsize=2,
            color='#c0392b', label='reproduced from raw fields (20 gaps, N=10)')
tg = np.array(list(fig2b_vals)); tv = np.array([v[0] for v in fig2b_vals.values()]); ts = np.array([v[1] for v in fig2b_vals.values()])
ax.errorbar(tg, tv*float(to.I10n_m.loc[0.21]), yerr=ts, fmt='s', ms=4.5, capsize=2,
            color='#2c3e50', alpha=0.7, label='Fig. 2b values (rescaled)')
ax.set_xlabel('design separation (mm)'); ax.set_ylabel('$I_{10}/I_{00}$ (norm.)')
ax.set_title('Optics: VMS scalar calibration (r = 0.98)', fontsize=12.5); ax.legend(fontsize=9)
ax = axs[0,1]
ax.errorbar(go2.index, D_mm, yerr=D_sd, fmt='o', ms=4, capsize=2, color='#2980b9')
ax.plot([0,0.26],[0,0.26],'k--',lw=0.9)
ax.axvspan(0.04, 0.095, color='orange', alpha=0.12)
ax.text(0.052, 0.21, 'fit floor /\nnon-monotone', fontsize=9.5)
ax.set_xlabel('design separation (mm)'); ax.set_ylabel('fitted separation $\\hat{D}$ (mm)')
ax.set_title('Optics: two-point fit on direct images', fontsize=12.5)
ax = axs[0,2]
ax.semilogy(go2.index, D_sd*1000, 'o-', ms=4, lw=1.1, color='#2980b9', label='fit repeatability (N=10)')
ax.semilogy(go2.index, crb_mm*1000, 's--', ms=3.5, lw=1.1, color='#27ae60', label='CRB (direct image)')
ax.axhline(25, color='#c0392b', lw=1.3, ls=':', label='VMS scalar $\\delta D$ (25 µm)')
ax.set_xlabel('design separation (mm)'); ax.set_ylabel('$\\delta D$ (µm)')
ax.set_title('Optics: precision vs CRB', fontsize=12.5); ax.legend(fontsize=9)
ax = axs[1,0]
ug = np.array(list(fig2d_vals)); uv = np.array([v[0] for v in fig2d_vals.values()]); us = np.array([v[1] for v in fig2d_vals.values()])
ax.errorbar(ug, uv, yerr=us, fmt='o-', ms=4, lw=1.1, capsize=2, color='#c0392b')
ax.set_xlabel('design separation (mm)'); ax.set_ylabel('$I_{10}/I_{00}$ (norm.)')
ax.set_title('Ultrasound: VMS scalar calibration (archived)', fontsize=12.5)
ax = axs[1,1]
ax.errorbar(fu2.index, fu2.D_mm_mean, yerr=fu2.D_mm_std, fmt='o', ms=4, capsize=2, color='#2980b9')
ax.plot([0,1.6],[0,1.6],'k--',lw=0.9)
ax.set_xlabel('design separation (mm)'); ax.set_ylabel('fitted separation $\\hat{D}$ (mm)')
ax.set_title('Ultrasound: two-point fit (bias at small $D$)', fontsize=12.5)
ax = axs[1,2]
ax.semilogy(fu2.index, fu2.D_mm_std*1000, 'o-', ms=4, lw=1.1, color='#2980b9', label='fit repeatability (N=4)')
ax.semilogy(fu2.index, fu2.crb_mm*1000, 's--', ms=3.5, lw=1.1, color='#27ae60', label='CRB (direct map)')
ax.axhline(14, color='#c0392b', lw=1.3, ls=':', label='VMS scalar $\\delta D$ (14 µm)')
ax.set_xlabel('design separation (mm)'); ax.set_ylabel('$\\delta D$ (µm)')
ax.set_title('Ultrasound: precision vs CRB', fontsize=12.5); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'figS18.png'), dpi=300); plt.close(fig)

