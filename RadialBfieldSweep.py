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
dE = (E[...,1]-E[...,0])

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

fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=Nangle, nrows=5, figure=fig)

# Plot uncorrected decoherence rates
for j in range(2):
    for i in range(Nangle):
        ax = fig.add_subplot( spec[j,i] )
        plt.imshow( r0[j, i, :, :], interpolation='bilinear', vmin=rmin, vmax=rmax, extent=[-max_strain, max_strain, -max_strain, max_strain] )
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
            plt.tick_params(left=False, labelleft=False)
        if i==(Nangle-1):
            plt.colorbar()
        plt.tick_params(bottom=False, labelbottom=False)

dEmin = np.min(dE)
dEmax = np.max(dE)

# Plot qubit energy splitting
for i in range(Nangle):
    ax = fig.add_subplot( spec[2,i] )
    plt.imshow( dE[i, :, :], interpolation='bilinear', cmap='spring', vmax=dEmax, vmin=dEmin, extent=[-max_strain, max_strain, -max_strain, max_strain])
    if i ==0:
        plt.ylabel('$E_1-E_0$\n'+r'$\alpha$ (GHz)')
    else:
        plt.tick_params(left=False, labelleft=False)
    if i==(Nangle-1):
            plt.colorbar()
    plt.tick_params(bottom=False, labelbottom=False)

# Plot corrected decoherence rates
for j in range(2):
    for i in range(Nangle):
        ax = fig.add_subplot( spec[j+3, i] )
        plt.imshow( rcorr[j, i, :, :], interpolation='bilinear', vmin=rmin, vmax=rmax, extent=[-max_strain, max_strain, -max_strain, max_strain] )
        if i == 0:
            if j == 0:
                plt.ylabel('$T_1^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 1:
                plt.ylabel('$T_{2,x}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
            if j == 2:
                plt.ylabel('$T_{2,y}^{-1}$ (arb.)\n'+r'$\alpha$ (GHz)')
        else:
            plt.tick_params(left=False, labelleft=False)
        if j == 1:
            plt.xlabel(r'$\beta$ (GHz)')
        else:
            plt.tick_params(bottom=False, labelbottom=False)
        if i==(Nangle-1):
            plt.colorbar()

# Plot T2 vs T1
# Uncorrected
plt.figure()
ax = plt.subplot(121)
for i in range(Nangle):
    plt.scatter( T1_0[i,:,:].flatten(), T2_x_0[i,:,:].flatten(), s=1, label='$T_{2,x}^{-1}$' + r'$\theta=$' + '{0:.1f}'.format(theta[i]*180/np.pi) )
xlim = np.array(ax.get_xlim())
plt.plot( xlim, xlim/2, 'k--', label='$T_2^{-1}=(2T_1)^{-1}$', zorder=0 )
plt.title('Uncorrected')
plt.legend()
plt.xlabel('$T_1^{-1}$')
plt.ylabel('$T_2^{-1}$')

# Corrected
ax = plt.subplot(122)
for i in range(Nangle):
    plt.scatter( T1_corr[i,:,:].flatten(), T2_x_corr[i,:,:].flatten(), s=1, label='$T_{2,x}^{-1}$' + r'$\theta=$' + '{0:.1f}'.format(theta[i]*180/np.pi) )
xlim = np.array(ax.get_xlim())
plt.plot( xlim, xlim/2, 'k--', label='$T_2^{-1}=(2T_1)^{-1}$', zorder=0  )
plt.title('Corrected')
plt.legend()
plt.xlabel('$T_1^{-1}$')
plt.ylabel('$T_2^{-1}$')

# Plot energy gap*T2 ~ coherence time/rabi time
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=Nangle, nrows=1, figure=fig)

# Plot uncorrected decoherence rates

for i in range(Nangle):
    ax = fig.add_subplot( spec[0,i] )
    plt.imshow( dE[i, :, :]/T2_x_corr[i, :, :], interpolation='bilinear' )
    
    plt.title(r'$\theta=$' + '{0:.1f}'.format(theta[i]*180/np.pi) + r'$\deg$')
    if i == 0:
        plt.ylabel(r'$\Delta ET_{2,y}^{-1}$' + ' (arb.)\n'+r'$\alpha$ (GHz)')
    else:
        plt.tick_params(left=False, labelleft=False)

# Plot T2 vs dE
plt.figure()
plt.scatter( dE, T2_x_corr, s=1, label='Data' )
plt.axvline(B_magnitude, color='k', linestyle='--', label='B-field Frequency')
plt.legend()
plt.xlabel(r'$\Delta E$ (GHz)')
plt.ylabel(r'$1/T_2$ (arb.)')

# Plot decoherence components for a given angle
angle = 0
for direction in ['x','z']:
    D = getDephasingOperator(Lcorr, direction)
    Mu = getDephasingComponents(D)
    Mu_min = np.real(np.min(Mu))
    Mu_max = np.real(np.max(Mu))

    fig = plt.figure()
    fig.suptitle('{0}={1:.1f}, {2}-direction'.format(r'$\theta$', theta[angle]*180/np.pi, direction))
    spec = gridspec.GridSpec(nrows = 4, ncols=4, figure=fig )

    for i in range(4):
        for j in range(4):
            ax = plt.subplot(spec[i,j])
            plt.imshow( np.real(Mu[angle,:,:,i,j]), vmin = Mu_min, vmax = Mu_max )


plt.show()