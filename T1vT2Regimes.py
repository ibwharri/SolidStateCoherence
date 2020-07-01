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

# Sweep z field
# =============
Npts = 300
Bz = np.linspace(1e-3, 10, Npts)
H = h_total(lambda_soc, q_orb, Bz, 0, 0, 0, 0 )
E = np.linalg.eigvalsh(H)
L0 = getLindbladian(H, g, h, Delta, kBT )

T2 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'x')

fit = np.polyfit(Bz, T2, 2)
print(fit)
T2_poleffect = fit[-1]

plt.figure()
ax=plt.subplot(111)
plt.plot( Bz, T2, label=r'$T_2^{-1}$' )
plt.plot( Bz, np.polyval(fit, Bz), color='navy', linestyle='--', label= 'Quadratic Fit')
plt.axhline( T2_poleffect, color='crimson', linestyle='--', zorder=1, label=r'$T_2(B_z=0)$')
plt.xlabel(r'$B_z$ (GHz)')
plt.ylabel(r'$T_2^{-1}$ (arb.)')
plt.legend()

# plt.subplot(212)
# plt.plot( Bz, E )

# Sweep radial field conditions
# =============================
# Define magnetic field components
Nangle = 5
theta = np.linspace(0, np.pi/2, Nangle)
B_magnitude = 1 # GHz
B = B_magnitude*np.array( [np.cos(theta), np.sin(theta)] )

# Define strain components alpha, beta
Npts = 20
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
E = np.linalg.eigvalsh(H)
dE = (E[...,1]-E[...,0])

# Decoherence rate with degeneracy correction
Lcorr = getLindbladian(H, g, h, Delta, kBT, np.array([[0,1],[2,3]]))
T1rad = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'z' )
T2rad = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'x')

# Plot T2 vs T1
# Uncorrected
plt.figure()
ax = plt.subplot(111)
plt.scatter( T1rad[:,:,:].flatten(), T2rad[:,:,:].flatten(), s=3, label='$T_2^{-1}$' )
xlim = np.array(ax.get_xlim())
plt.plot( xlim, xlim/2, 'k--', label='$T_2^{-1}=(2T_1)^{-1}$', zorder=0 )
plt.axhline( T2_poleffect, color='crimson', linestyle='--', label=r'Axial Field $T_2(B_z=0)$', zorder=0)
plt.title('Radial')
plt.legend()
plt.xlabel('$T_1^{-1}$')
plt.ylabel('$T_2^{-1}$')


# Plot while varying temperature
N_temp = 5
kBTs = np.linspace(83.5, 1, N_temp)

S = np.kron(z,I) + np.kron(x,I)
_, U = np.linalg.eigh(S)

figT1T2 = plt.figure()
spec = gridspec.GridSpec(nrows=1, ncols=N_temp, figure=figT1T2)
figDistT1 = plt.figure()
specDistT1 = gridspec.GridSpec(nrows=1, ncols=N_temp, figure=figDistT1)
figRT2 = plt.figure()
specRT2 = gridspec.GridSpec(nrows=1, ncols=N_temp, figure=figRT2)

for i,kBT in enumerate(kBTs):
    Bz = np.linspace(1e-3, 10, 10)
    H = h_total(lambda_soc, q_orb, Bz, 0, 0, 0, 0 )
    E = np.linalg.eigvalsh(H)
    L0 = getLindbladian(H, g, h, Delta, kBT )

    T2 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'x')

    fit = np.polyfit(Bz, T2, 2)
    T2_poleffect = fit[-1]

    # Radial component
    H = h_total(lambda_soc, q_orb, 0, B[0,:,:,:], B[1,:,:,:], alpha, beta)
    E, V = np.linalg.eigh(H)
    dE = (E[...,1]-E[...,0])

    dist = basisSimilarity(U, V)
    r = np.log10(phononPolarisationRatio(np.kron(z,I),np.kron(x,I),V))

    # Decoherence rate with degeneracy correction
    Lcorr = getLindbladian(H, g, h, Delta, kBT, np.array([[0,1],[2,3]]))
    T1rad = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'z' )
    T2rad = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'x')

    # Plot T1 vs T2
    plt.figure(3)
    ax = plt.subplot(spec[0,i])
    sc = ax.scatter( T1rad.flatten(), T2rad.flatten()/1e6, s=3, c=dist.flatten(), label='$T_2^{-1}$' )
    ax.plot( xlim, xlim/2e6, 'k--', label='$T_2^{-1}=(2T_1)^{-1}$', zorder=0 )
    ax.axhline( T2_poleffect/1e6, color='crimson', linestyle='--', label=r'Axial Field $T_2(B_z=0)$', zorder=0)
    ax.set_title('$k_BT=${0:.1f} GHz'.format(kBT))
    if i==0:
        ax.set_ylabel(r'$T_2^{-1}x10^{-6}$ (arb.)')
        ax.legend()
    if i==N_temp-1:
        plt.colorbar(sc, ax=ax)
    
    ax.set_xlabel('$T_1^{-1}$')

    # Plot dist vs T1
    plt.figure(4)
    ax = plt.subplot(specDistT1[0,i])
    ax.scatter( dist.flatten(), T1rad.flatten()/1e6, s=3 )
    ax.set_title('$k_BT=${0:.1f} GHz'.format(kBT))
    if i==0:
        ax.set_ylabel(r'$T_1^{-1}x10^{-6}$ (arb.)')
    
    ax.set_xlabel('$||UV^\dag||_1$')

    # Plot r vs T2
    plt.figure(5)
    ax = plt.subplot(specRT2[0,i])
    sc = ax.scatter( dist.flatten(), T2rad.flatten()/1e6, s=3, c=dist.flatten(), label='$T_2^{-1}$' )
    ax.axhline( T2_poleffect/1e6, color='crimson', linestyle='--', label=r'Axial Field $T_2(B_z=0)$', zorder=0)
    ax.set_title('$k_BT=${0:.1f} GHz'.format(kBT))
    if i==0:
        ax.set_ylabel(r'$T_2^{-1}x10^{-6}$ (arb.)')
        ax.legend()
    #else:
        #plt.tick_params(left=False, labelleft=False)
    if i==N_temp-1:
        plt.colorbar(sc, ax=ax)
    
    ax.set_xlabel('$log(r)$')
    

plt.figure()
plt.imshow(dist[0,:,:])

plt.show()