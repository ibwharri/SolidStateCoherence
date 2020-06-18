import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from Simulation import *

g = lambda omega: np.sqrt(omega)
h = lambda omega: np.sqrt(omega)
Delta = lambda omega: omega**2
kBT = 83.5 # GHz
lambda_soc = 45 # GHz
q_orb = .1

# Sweep z field
# =============
Npts = 300
Bz = np.linspace(0, 1, Npts)
H = h_total(lambda_soc, q_orb, Bz, 0, 0, 0, 0 )
L0 = getLindbladian(H, g, h, Delta, kBT )

T1 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'z' )
T2_x = thermalisedOrbitalDephasingRate(H, L0, kBT, 'x')
T2_y = thermalisedOrbitalDephasingRate(H, L0, kBT, 'y')

plt.subplot(1,2,1)
plt.plot( Bz, T1 )
plt.xlabel(r'$B_z$ (GHz)')
plt.ylabel(r'Axial Decoherence rate (arb.)')
plt.legend(['$T_1^{-1}$'])
plt.subplot(1,2,2)
plt.plot( Bz, T2_x, Bz, T2_y )
plt.xlabel(r'$B_z$ (GHz)')
plt.ylabel(r'Radial Decoherence rate (arb.)')
plt.legend(['$T_{2,x}^{-1}$', '$T_{2,y}^{-1}$'])
plt.show()