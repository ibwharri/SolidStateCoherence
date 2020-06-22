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
dE_target = 1 # GHz

# Sweep radial field conditions
# =============================
# Define magnetic field components
Nangle = 5
Nmag = 30
theta = np.linspace(0, np.pi/2, Nangle)
B_magnitude = 3 # GHz
B_magnitude = np.linspace(0, B_magnitude, Nmag)
theta, B_magnitude = np.meshgrid( theta, B_magnitude)
B = np.array( [B_magnitude*np.cos(theta), B_magnitude*np.sin(theta)] )

# Define strain components alpha, beta
Npts = 50
max_strain = 20 # GHz
alpha = np.linspace(-max_strain, max_strain, Npts)
beta = np.linspace(-max_strain, max_strain, Npts)
alpha, beta = np.meshgrid( alpha, beta)

# Ensure B, alpha, beta are of shape (Nangle, Npts, Npts)
B = np.repeat(B[:,:,:,np.newaxis], Npts, axis=-1)
B = np.repeat(B[:,:,:,:,np.newaxis], Npts, axis=-1)
alpha = np.repeat(alpha[np.newaxis,:,:], Nangle, axis=0)
alpha = np.repeat(alpha[np.newaxis,:,:,:], Nmag, axis=0)
beta = np.repeat(beta[np.newaxis,:,:], Nangle, axis=0)
beta = np.repeat(beta[np.newaxis,:,:,:], Nmag, axis=0)

H = h_total(lambda_soc, q_orb, 0, B[0,:,:,:,:], B[1,:,:,:,:], alpha, beta)
E, _ = np.linalg.eigh(H)
dE = (E[...,1]-E[...,0])

# Find closest field to target
i_target = np.argmin( np.abs(dE-dE_target), axis=0 )
i_target = i_target[np.newaxis,np.newaxis,...]
B = np.take_along_axis(B, i_target, axis=1).squeeze()
alpha = np.take_along_axis(alpha, i_target[0,...], axis=0).squeeze()
beta = np.take_along_axis(beta, i_target[0,...], axis=0).squeeze()

H = h_total(lambda_soc, q_orb, 0, B[0,...], B[1,...], alpha, beta)
E, _ = np.linalg.eigh(H)
dE = (E[...,1]-E[...,0])

# Decoherence rate with degeneracy correction
Lcorr = getLindbladian(H, g, h, Delta, kBT, np.array([[0,1],[2,3]]))
T1_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'z' )
T2_x_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'x')
T2_y_corr = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'y')
rcorr = np.array([T1_corr, T2_x_corr, T2_y_corr])

rmax = np.max( rcorr )
rmin = np.min( rcorr )

fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=Nangle, nrows=4, figure=fig)

# Plot uncorrected decoherence rates
for j in range(2):
    for i in range(Nangle):
        ax = fig.add_subplot( spec[j,i] )
        plt.imshow( rcorr[j, i, :, :], interpolation='bilinear', vmin=rmin, vmax=rmax, extent=[-max_strain, max_strain, -max_strain, max_strain] )
        if j == 0:
            plt.title(r'$\theta=$' + '{0:.1f}'.format(theta[0,i]*180/np.pi) + r'$\deg$')
        if i == 0:
            if j == 0:
                plt.ylabel('$T_1^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 1:
                plt.ylabel('$T_{2}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 2:
                plt.ylabel('$T_{2,y}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
        else:
            plt.tick_params(left=False, labelleft=False)
        if i==(Nangle-1):
            plt.colorbar()
        plt.tick_params(bottom=False, labelbottom=False)

# Plot qubit energy splitting
dEmin = np.min(dE)
dEmax = np.max(dE)
for i in range(Nangle):
    ax = fig.add_subplot( spec[2,i] )
    plt.imshow( dE[i, :, :], interpolation='bilinear', cmap='spring', vmin=dEmin, vmax=dEmax, extent=[-max_strain, max_strain, -max_strain, max_strain])
    if i ==0:
        plt.ylabel('$E_1-E_0$\n'+r'$\alpha$ (GHz)')
    else:
        plt.tick_params(left=False, labelleft=False)
    if i==(Nangle-1):
        plt.colorbar()
    plt.tick_params(bottom=False, labelbottom=False)

# Plot required magnetic field
Bmag = np.sqrt(B[0,...]**2+B[1,...]**2)
Bmax = np.max(Bmag)
Bmin = np.min(Bmag)
for i in range(Nangle):
    ax = fig.add_subplot( spec[3, i] )
    plt.imshow( Bmag[i,...], interpolation='bilinear', vmin=Bmin, vmax=Bmax, cmap='summer', extent=[-max_strain, max_strain, -max_strain, max_strain] )
    if i == 0:
        plt.ylabel('$B$ (GHz)\n'+r'$\alpha$ (GHz)')
    else:
        plt.tick_params(left=False, labelleft=False)
    if i==(Nangle-1):
        plt.colorbar()
    plt.xlabel(r'$\beta$ (GHz)')

# Plot T2 vs T1
plt.figure()
ax = plt.subplot(111)
for i in range(Nangle):
    plt.scatter( T1_corr[i,:,:].flatten(), T2_x_corr[i,:,:].flatten(), s=1, label='$T_{2,x}^{-1}$' + r'$\theta=$' + '{0:.1f}'.format(theta[0,i]*180/np.pi) )
xlim = np.array(ax.get_xlim())
plt.plot( xlim, xlim/2, 'k--', label='$T_2^{-1}=(2T_1)^{-1}$' )
plt.title('Corrected')
plt.legend()
plt.xlabel('$T_1^{-1}$')
plt.ylabel('$T_2^{-1}$')

plt.show()