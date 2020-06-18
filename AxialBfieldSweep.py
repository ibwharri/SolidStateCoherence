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

# Sweep radial field conditions
# =============================
# Define magnetic field components
Nangle = 5
theta = np.linspace(0, np.pi/2, Nangle)
B_magnitude = 1 # GHz
B = B_magnitude*np.array( [np.cos(theta), np.sin(theta)] )

# Define strain components alpha, beta
Npts = 50
max_strain = 20 # GHz
alpha = np.linspace(-max_strain, max_strain, Npts)
beta = np.linspace(-max_strain, max_strain, Npts)
alpha, beta = np.meshgrid( alpha, beta)

# Ensure B, alpha, beta are of shape (Nangle, Npts, Npts)
B = np.repeat(B[:,:,np.newaxis], Npts, axis=-1)
B = np.repeat(B[:,:,:,np.newaxis], Npts, axis=-1)
alpha = np.repeat(alpha[np.newaxis,:,:], Nangle, axis=0)
beta = np.repeat(beta[np.newaxis,:,:], Nangle, axis=0)

H = h_total(lambda_soc, q_orb, 0, B[0,:,:,:], B[1,:,:,:], alpha, beta)
E, _ = np.linalg.eigh(H)
dE = E[...,1]-E[...,0]

# Decoherence rate without degeneracy correction
L0 = getLindbladian(H, g, h, Delta, kBT)
T1_0 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'z' )
T2_x_0 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'x')
T2_y_0 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'y')
r0 = np.array([T1_0, T2_x_0, T2_y_0])

# Decoherence rate with degeneracy correction
Lcorr = getLindbladian(H, g, h, Delta, kBT, np.array([[0,1],[2,3]]))
T1_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'z' )
T2_x_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'x')
T2_y_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'y')
rcorr = np.array([T1_corr, T2_x_corr, T2_y_corr])

rmax = np.max( np.array([r0, rcorr]) )
rmin = np.min( np.array([r0, rcorr]) )
Nlevels = 100

fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=Nangle+1, nrows=5, figure=fig)

# Plot uncorrected decoherence rates
for j in range(2):
    for i in range(Nangle):
        ax = fig.add_subplot( spec[j,i] )
        plt.imshow( r0[j, i, :, :], interpolation='bilinear', vmin=rmin, vmax=rmax )
        if j == 0:
            plt.title(r'$\theta=$' + '{0:.1f}'.format(theta[i]*180/np.pi) + r'$\deg$')
        if i == 0:
            if j == 0:
                plt.ylabel('$T_1^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 1:
                plt.ylabel('$T_{2,x}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 2:
                plt.ylabel('$T_{2,y}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
        else:
            plt.tick_params(left='off', labelleft='off')
        plt.tick_params(bottom='off', labelbottom='off')

for i in range(Nangle):
    ax = fig.add_subplot( spec[2,i] )
    plt.imshow( dE[i, :, :], interpolation='bilinear', cmap='spring')
    if i ==0:
        plt.ylabel('$E_1-E_0$\n'+r'$\alpha$ (GHz)')
    else:
        plt.tick_params(left='off', labelleft='off')
    plt.tick_params(bottom='off', labelbottom='off')

# Plot corrected decoherence rates
for j in range(2):
    for i in range(Nangle):
        ax = fig.add_subplot( spec[j+3, i] )
        plt.imshow( rcorr[j, i, :, :], interpolation='bilinear', vmin=rmin, vmax=rmax )
        if i == 0:
            if j == 0:
                plt.ylabel('$T_1^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 1:
                plt.ylabel('$T_{2,x}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 2:
                plt.ylabel('$T_{2,y}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
        else:
            plt.tick_params(left='off', labelleft='off')
        if j == 1:
            plt.xlabel(r'$\beta$ (GHz)')
        else:
            plt.tick_params(bottom='off', labelbottom='off')

# Plot T2 vs T1
# Uncorrected
ax = fig.add_subplot( spec[ 0:2 , Nangle] )
plt.scatter( T1_0.flatten(), T2_x_0.flatten(), label='$T_{2,x}^{-1}$' )
xlim = np.array(ax.get_xlim())
plt.plot( xlim, xlim/2, 'k--', label='$T_2^{-1}=$(2T_1)^{-1}$' )
plt.xlabel('$T_1^{-1}$')
plt.ylabel('$T_2^{-1}$')

# Corrected
ax = fig.add_subplot( spec[ 3:5 , Nangle] )
plt.scatter( T1_corr.flatten(), T2_x_corr.flatten(), label='$T_{2,x}^{-1}$' )
xlim = np.array(ax.get_xlim())
plt.plot( xlim, xlim/2, 'k--', label='$T_2^{-1}=$(2T_1)^{-1}$' )
plt.xlabel('$T_1^{-1}$')
plt.ylabel('$T_2^{-1}$')

plt.show()