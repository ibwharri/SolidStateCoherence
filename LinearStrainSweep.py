import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import numpy as np
from Simulation import *

g = lambda omega: np.sqrt(omega)
h = lambda omega: np.sqrt(omega)
Delta = lambda omega: omega**2
kBT = 83.5 # GHz
lambda_soc = 45/2 # GHz
q_orb = .1

# Sweep radial field conditions
# =============================
# Define magnetic field components
theta = 0
B_magnitude = 1 # GHz
B = B_magnitude*np.array( [np.cos(theta), np.sin(theta)] )

# Define strain components alpha, beta
max_strain = 20 # GHz
Npts = 200
beta = np.linspace(-max_strain, max_strain, Npts)
Npts = 5
alpha = np.linspace(-max_strain, max_strain, Npts)
alpha, beta = np.meshgrid( alpha, beta)

H = h_total(lambda_soc, q_orb, 0, B[0], B[1], alpha, beta)
E, V = np.linalg.eigh(H)
dE = (E[...,1]-E[...,0])

# Decoherence rate with degeneracy correction
Lcorr = getLindbladian(H, g, h, Delta, kBT, np.array([[0,1],[2,3]]))
#T1_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'z' )
T2_x_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'x')

D = getDephasingOperator(Lcorr, 'x')
Mu = getDephasingComponents(D)

plt.subplot(311)
cmap = get_cmap('viridis')
norm = Normalize()
norm.autoscale(alpha)
for i,strain in enumerate(alpha[0,:]):
    plt.plot(beta[:,0], T2_x_corr[:,i], label='{0}={1:.1f}'.format(r'$\alpha$', strain), color=cmap(norm(strain)))
plt.legend()
plt.ylabel(r'$T_2^{-1}$ (arb.)')

plt.subplot(312)
S = np.kron(z, I) + np.kron(x, I)
S = np.einsum('...ji,...jk,...kl->...il', np.conjugate(V), S, V)
Mu = np.real(getDephasingComponents(S))

for i,strain in enumerate(alpha[0,:]):
    plt.plot(beta[:,0], Mu[:,i,0,1], label='{0}={1:.1f}'.format(r'$\alpha$', strain), color=cmap(norm(strain)))
plt.ylabel(r'$Tr(\sigma_0\otimes\sigma_xS)$ (GHz)')
plt.legend()

plt.subplot(313)
for i,strain in enumerate(alpha[0,:]):
    plt.plot(beta[:,0], np.sqrt(beta[:,0]**2+alpha[:,i]**2), label='{0}={1:.1f}'.format(r'$\alpha$', strain), color=cmap(norm(strain)))
    #plt.plot(beta[:,0], E[:,i,1], color=cmap(norm(strain)))
    #plt.plot(beta[:,0], E[:,i,2], color=cmap(norm(strain)))
    #plt.plot(beta[:,0], E[:,i,3], color=cmap(norm(strain)))
plt.xlabel(r'$\beta$ (GHz)')
plt.ylabel(r'E (GHz)')
plt.legend()
plt.show()