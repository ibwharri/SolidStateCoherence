import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from Simulation import *

g = lambda omega: np.sqrt(omega)
h = lambda omega: np.sqrt(omega)
Delta = lambda omega: omega**2
kBT = 83.5 # GHz
lambda_soc = 45/2 # GHz
q_orb = .1
dgs = 1.3e6 # GHz
fgs = -1.7e6 # GHz

# Plot alpha/beta as a function of strain
# =======================================
eps_max = 2e-5
Npts = 200
eps = np.linspace(-eps_max, eps_max, Npts)
eps_cant = np.zeros((3,3))
eps_cant[0,0] = 1
eps_cant = np.einsum('i,jk->ijk',eps,eps_cant)

# Find strain for in defect coordinate system
theta_111 = np.arccos(2/np.sqrt(6))

eps_110_1 = rotate_tensor(eps_cant, np.array([0,1,0]), theta_111, 2)
axis = rotate_tensor(np.array([0,0,1]), np.array([0,1,0]), theta_111)
eps_110_2 = rotate_tensor(eps_110_1, axis, -np.pi/2-.0001, 2)
eps_100_1 = rotate_tensor(eps_110_1, axis, -np.pi/4, 2)
eps_100_2 = rotate_tensor(eps_110_1, axis, np.pi/4, 2)

# Find alpha/beta
alpha_110 = find_alpha(np.array([eps_110_1, eps_110_2]), dgs, fgs)
alpha_100 = find_alpha(np.array([eps_100_1, eps_100_2]), dgs, fgs)
alpha = np.array([alpha_110, alpha_100])
beta_110 = find_beta(np.array([eps_110_1, eps_110_2]), dgs, fgs)
beta_100 = find_beta(np.array([eps_100_1, eps_100_2]), dgs, fgs)
beta = np.array([beta_110, beta_100])

# Plot
plt.figure()
plt.subplot(121)
plt.plot(eps*1e5, alpha_110[0,:], 'crimson', label=r'$\alpha$, Orientation 1')
plt.plot(eps*1e5, beta_110[0,:], 'crimson', ls='--', label=r'$\beta$, Orientation 1')
plt.plot(eps*1e5, alpha_110[1,:], 'navy', label=r'$\alpha$, Orientation 2')
plt.plot(eps*1e5, beta_110[1,:], 'navy', ls='--', label=r'$\beta$, Orientation 2')
plt.title('110 Oriented Cantilever')
plt.xlabel(r'$\epsilon\times 10^5$')
plt.ylabel(r'Strain Respons (GHz)')
plt.legend()

plt.subplot(122)
plt.plot(eps*1e5, alpha_100[0,:], 'crimson', label=r'$\alpha$, Orientation 1')
plt.plot(eps*1e5, beta_100[0,:], 'crimson', ls='--', label=r'$\beta$, Orientation 1')
plt.plot(eps*1e5, alpha_100[1,:], 'navy', label=r'$\alpha$, Orientation 2')
plt.plot(eps*1e5, beta_100[1,:], 'navy', ls='--', label=r'$\beta$, Orientation 2')
plt.title('100 Oriented Cantilever')
plt.xlabel(r'$\epsilon \times 10^5$')
plt.legend()

# Find T2
# ================================
# Define magnetic field components
B_aligned = np.array([0,1,0])
B_misaligned = rotate_tensor(np.array([0,0,1]), np.array([0,1,0]), theta_111)

def find_coherence(B, alpha, beta, transition_pair=None):
    H = h_total(lambda_soc, q_orb, B[2], B[0], B[1], alpha, beta)
    E = np.linalg.eigvalsh(H)
    dE = (E[...,1]-E[...,0])

    # Decoherence rate without degeneracy correction
    L = getLindbladian( H, g, h, Delta, kBT, transition_pair)
    #T1_0 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'z' )
    T1 = thermalisedOrbitalDephasingRate(H, L, kBT, 'z')
    T2 = thermalisedOrbitalDephasingRate(H, L, kBT, 'y')

    return dE, T1, T2

# Plot
plt.figure()
ax_110_dE = plt.subplot(321)
ax_100_dE = plt.subplot(322)
ax_110_T1 = plt.subplot(323)
ax_100_T1 = plt.subplot(324)
ax_110_T2 = plt.subplot(325)
ax_100_T2 = plt.subplot(326)
colors = ['crimson', 'navy']

# Aligned 110
dE, T1, T2 = find_coherence(B_aligned, alpha_110, beta_110, np.array([[0,1],[2,3]]))
[ax_110_dE.plot(eps*1e5, dE[i,:], col, ls='-') for i,col in enumerate(colors)]
[ax_110_T1.plot(eps*1e5, T1[i,:]/1e6, col, ls='-') for i,col in enumerate(colors)]
[ax_110_T2.plot(eps*1e5, T2[i,:]/1e6, col, ls='-', label='Orientation {}, Aligned'.format(i+1)) for i,col in enumerate(colors)]

# Misaligned 110
dE, T1, T2 = find_coherence(B_misaligned, alpha_110, beta_110)
[ax_110_dE.plot(eps*1e5, dE[i,:], col, ls='--') for i,col in enumerate(colors)]
[ax_110_T1.plot(eps*1e5, T1[i,:]/1e6, col, ls='--') for i,col in enumerate(colors)]
[ax_110_T2.plot(eps*1e5, T2[i,:]/1e6, col, ls='--', label='Orientation {}, Misaligned'.format(i+1)) for i,col in enumerate(colors)]

# Aligned 100
dE, T1, T2 = find_coherence(B_aligned, alpha_100, beta_100, np.array([[0,1],[2,3]]))
[ax_100_dE.plot(eps*1e5, dE[i,:], col, ls='-') for i,col in enumerate(colors)]
[ax_100_T1.plot(eps*1e5, T1[i,:]/1e6, col, ls='-') for i,col in enumerate(colors)]
[ax_100_T2.plot(eps*1e5, T2[i,:]/1e6, col, ls='-') for i,col in enumerate(colors)]

# Misaligned 100
dE, T1, T2 = find_coherence(B_misaligned, alpha_100, beta_100)
[ax_100_dE.plot(eps*1e5, dE[i,:], col, ls='--') for i,col in enumerate(colors)]
[ax_100_T1.plot(eps*1e5, T1[i,:]/1e6, col, ls='--') for i,col in enumerate(colors)]
[ax_100_T2.plot(eps*1e5, T2[i,:]/1e6, col, ls='--') for i,col in enumerate(colors)]

ax_100_T2.set_xlabel(r'$\epsilon \times 10^5$')
ax_110_T2.set_xlabel(r'$\epsilon \times 10^5$')
ax_100_dE.set_title(r'100 Oriented Cantilever')
ax_110_dE.set_title(r'110 Oriented Cantilever')
ax_110_dE.set_ylabel(r'Qubit Frequency (GHz)')
ax_110_T1.set_ylabel(r'$T_1^{-1}/10^6$ (arb.)')
ax_110_T2.set_ylabel(r'$T_2^{-1}/10^6$ (arb.)')
ax_110_T2.legend(bbox_to_anchor=(0,-.45,2,.1), loc='center', ncol=4)
plt.show()