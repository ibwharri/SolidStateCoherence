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
Npts = 11
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

plt.subplot(211)
for i,strain in enumerate(alpha[0,:]):
    plt.plot(beta[:,0], T2_x_corr[:,i], label='{0:.1f}'.format(strain))
plt.legend()
for i,strain in enumerate(alpha[0,:]):
    plt.subplot(int(np.ceiling(Npts/5)), 5)
    plt.imshow(Mu[i,])
plt.legend()
plt.show()