"""SI Fig. S2: classical Fisher information of direct imaging vs binary SPADE,
normalized to the quantum Fisher information (two incoherent point sources, Gaussian PSF)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8})

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, '..', 'figures')

sigma = 0.5e-3; DFL = 2.905*sigma
x = np.linspace(-0.01, 0.01, 1000); dgr = x[1]-x[0]
X, Y = np.meshgrid(x, x)
def Pr(theta):
    return 0.5*((1/(2*np.pi*sigma**2))*np.exp(-((X+theta/2)**2+Y**2)/(2*sigma**2)) +
                (1/(2*np.pi*sigma**2))*np.exp(-((X-theta/2)**2+Y**2)/(2*sigma**2)))
def FI_DI(theta, d=1e-6):
    p = Pr(theta); dl = (np.log(Pr(theta+d)+1e-20)-np.log(Pr(theta-d)+1e-20))/(2*d)
    return np.sum(dl**2*p)*dgr*dgr
th = np.linspace(1e-6, 0.0035, 60)
fdi = np.array([FI_DI(t) for t in th])
Q = th**2/(16*sigma**2)
fbin = (1/(4*sigma**2))*Q*np.exp(-Q)/(1-np.exp(-Q))
qfi = 1/(4*sigma**2)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(th/DFL, fdi/qfi, lw=1.6, color='#1f77b4', label='Direct imaging (classical FI)')
ax.plot(th/DFL, fbin/qfi, lw=1.6, color='#d62728', label='Binary SPADE (physical mode sorting)')
ax.axhline(1, color='k', lw=1, ls='--', label='Quantum FI')
ax.set_xlabel('Separation $D$ / $D_{FL}$'); ax.set_ylabel('Fisher information (norm. to QFI)')
ax.set_xlim(0, 2.4); ax.set_ylim(0, 1.1); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'figS2.png'), dpi=300)
print('saved figures/figS2.png')
