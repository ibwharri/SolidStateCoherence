import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from Simulation import *
from time import perf_counter

g = lambda omega: np.sqrt(omega)
zero = lambda omega: 0*omega
Delta = lambda omega: omega**2
kBT = 83.5 # GHz
lambda_soc = 45/2 # GHz
q_orb = .1
dgs = 1.3e6 # GHz
fgs = -1.7e6 # GHz

# Plot qubit energy, T1, T2 as a function of cantilever angle for 111 cut cantilever, with B along z to test symmetry 
# ===================================================================================================================
eps_max = 10e-5
Npts = 50
Nangle = 300
eps = np.linspace(-eps_max, eps_max, Npts)
theta = np.linspace(-np.pi, np.pi, Nangle)
eps_cant = np.zeros((3,3))
eps_cant[0,0] = 1


# Find strain for in defect coordinate system
theta_111 = np.arccos(2/np.sqrt(6))

#eps_110 = rotate_tensor(eps_cant, np.array([0,1,0]), theta_111, 2)
#axis = rotate_tensor(np.array([0,0,1]), np.array([0,1,0]), theta_111)
eps_110 = eps_cant.copy()
axis = np.array([0,0,1])
eps_cant = np.array([rotate_tensor(eps_110, axis, -angle, 2) for angle in theta])
eps_cant = np.einsum('i,jkl->ijkl', eps, eps_cant)

# Find alpha/beta
alpha = find_alpha(eps_cant, dgs, fgs)
beta = find_beta(eps_cant, dgs, fgs)

def find_coherence(B, alpha, beta, g_phonon, h_phonon, transition_pair=None):
    H = h_total(lambda_soc, q_orb, B[...,2], B[...,0], B[...,1], alpha, beta)
    E = np.linalg.eigvalsh(H)
    dE = (E[...,1]-E[...,0])

    # Decoherence rate without degeneracy correction
    L = getLindbladian( H, g_phonon, h_phonon, Delta, kBT, transition_pair)

    #T1_0 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'z' )
    T2 = thermalisedOrbitalDephasingRate(H, L, kBT, 'y')

    return dE, np.log10(T2)

plt.figure()
ax = plt.subplot(111)
plt.plot(theta,alpha[-1,:],theta,beta[-1,:])
ax.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
ax.set_xticklabels([
    r'$-\pi$',
    r'$-\frac{3\pi}{4}$',
    r'$-\frac{\pi}{2}$',
    r'$-\frac{\pi}{4}$',
    '0',
    r'$\frac{\pi}{4}$',
    r'$\frac{\pi}{2}$',
    r'$\frac{3\pi}{4}$',
    r'$\pi$'])
plt.legend([r'$\alpha$',r'$\beta$'])

# Loop through values of h relative g
# ===================================

Nh = 9
rhs = np.linspace(-2,2,Nh)
B_z = np.array([0,0,1])

fig = plt.figure()
plt.tight_layout()
spec = gridspec.GridSpec(nrows=2, ncols=Nh, figure=fig)

for i,rh in enumerate(rhs):
    print(rh)
    t0 = perf_counter()

    h = lambda omega: rh*np.sqrt(omega)

    ax_z_dE = plt.subplot(spec[0,i])
    ax_z_dE.set_title('$r_h$={0:.1f}'.format(rh), pad=36)

    ax_z_T2 = plt.subplot(spec[1,i])
    ax_z_T2.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax_z_T2.set_xticklabels([
        r'$-\pi$',
        r'$-\frac{3\pi}{4}$',
        r'$-\frac{\pi}{2}$',
        r'$-\frac{\pi}{4}$',
        '0',
        r'$\frac{\pi}{4}$',
        r'$\frac{\pi}{2}$',
        r'$\frac{3\pi}{4}$',
        r'$\pi$'])

    if i==0:
        ax_z_dE.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    else:
        ax_z_dE.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False, top=True, labeltop=True)
        ax_z_T2.tick_params(left=False, labelleft=False)

    # Aligned 110
    dE_z, T2_z = find_coherence( B_z, alpha, beta, g, h, None)

    dE_max = np.max( dE_z )
    T_max = np.max( T2_z )
    dE_min = np.min( np.array([dE_z]) )
    T_min = np.min( T2_z )

    # Aligned plot
    h_subplot = .225*2/3
    v_wspace = .047

    if i==Nh-1:
        im = ax_z_dE.imshow( dE_z, vmin=dE_min, vmax=dE_max, cmap='spring', aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
        fig.colorbar(im, cax=fig.add_axes((.92, .88-h_subplot, 0.01, h_subplot)))
        im = ax_z_T2.imshow( T2_z, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
        fig.colorbar(im, cax=fig.add_axes((.92, .88-v_wspace-2*h_subplot, 0.01, h_subplot)))
    else:
        ax_z_dE.imshow( dE_z, vmin=dE_min, vmax=dE_max, cmap='spring', aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
        ax_z_T2.imshow( T2_z, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])

    print('Time: {0:.1f}'.format(perf_counter() - t0))


def find_coherence(B, alpha, beta, g_phonon, h_phonon, transition_pair=None):
    H = h_total(lambda_soc, q_orb, B[...,2], B[...,0], B[...,1], alpha, beta)
    E = np.linalg.eigvalsh(H)
    dE = (E[...,1]-E[...,0])

    # Decoherence rate without degeneracy correction
    LEgx = getLindbladian( H, g_phonon, zero, Delta, kBT, transition_pair)
    LEgy = getLindbladian( H, zero, h_phonon, Delta, kBT, transition_pair)
    L = [LEgx, LEgy]

    #T1_0 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'z' )
    T2 = thermalisedOrbitalDephasingRate(H, L, kBT, 'y')

    return dE, np.log10(T2)

plt.figure()
ax = plt.subplot(111)
plt.plot(theta,alpha[-1,:],theta,beta[-1,:])
ax.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
ax.set_xticklabels([
    r'$-\pi$',
    r'$-\frac{3\pi}{4}$',
    r'$-\frac{\pi}{2}$',
    r'$-\frac{\pi}{4}$',
    '0',
    r'$\frac{\pi}{4}$',
    r'$\frac{\pi}{2}$',
    r'$\frac{3\pi}{4}$',
    r'$\pi$'])
plt.legend([r'$\alpha$',r'$\beta$'])

# Loop through values of h relative g
# ===================================

Nh = 9
rhs = np.linspace(-2,2,Nh)
B_z = np.array([0,0,1])

fig = plt.figure()
plt.tight_layout()
spec = gridspec.GridSpec(nrows=2, ncols=Nh, figure=fig)

for i,rh in enumerate(rhs):
    print(rh)
    t0 = perf_counter()

    h = lambda omega: rh*np.sqrt(omega)

    ax_z_dE = plt.subplot(spec[0,i])
    ax_z_dE.set_title('$r_h$={0:.1f}'.format(rh), pad=36)

    ax_z_T2 = plt.subplot(spec[1,i])
    ax_z_T2.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax_z_T2.set_xticklabels([
        r'$-\pi$',
        r'$-\frac{3\pi}{4}$',
        r'$-\frac{\pi}{2}$',
        r'$-\frac{\pi}{4}$',
        '0',
        r'$\frac{\pi}{4}$',
        r'$\frac{\pi}{2}$',
        r'$\frac{3\pi}{4}$',
        r'$\pi$'])

    if i==0:
        ax_z_dE.tick_params(bottom=False, labelbottom=False)
    else:
        ax_z_dE.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax_z_T2.tick_params(left=False, labelleft=False)

    # Aligned 110
    dE_z, T2_z = find_coherence( B_z, alpha, beta, g, h, None)

    dE_max = np.max( dE_z )
    T_max = np.max( T2_z )
    dE_min = np.min( np.array([dE_z]) )
    T_min = np.min( T2_z )

    # Aligned plot
    h_subplot = .225*2/3
    v_wspace = .047

    if i==Nh-1:
        im = ax_z_dE.imshow( dE_z, vmin=dE_min, vmax=dE_max, cmap='spring', aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
        fig.colorbar(im, cax=fig.add_axes((.92, .88-h_subplot, 0.01, h_subplot)))
        im = ax_z_T2.imshow( T2_z, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
        fig.colorbar(im, cax=fig.add_axes((.92, .88-v_wspace-2*h_subplot, 0.01, h_subplot)))
    else:
        ax_z_dE.imshow( dE_z, vmin=dE_min, vmax=dE_max, cmap='spring', aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
        ax_z_T2.imshow( T2_z, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])

    print('Time: {0:.1f}'.format(perf_counter() - t0))


plt.show()