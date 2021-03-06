import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from Simulation import *

g = lambda omega: np.sqrt(omega)
h = lambda omega: np.sqrt(omega)
zero = lambda omega: 0*omega
Delta = lambda omega: omega**2
kBT = 83.5 # GHz
lambda_soc = 45/2 # GHz
q_orb = .1
dgs = 1.3e6 # GHz
fgs = -1.7e6 # GHz

# Plot qubit energy, T1, T2 as a function of cantilever angle
# ===========================================================
eps_max = 10e-5
Npts = 300
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

# Plot
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

# Find T2
# =============================
# Define magnetic field components
B_aligned = np.array([0,1,0])
B_aligned = np.array([rotate_tensor(B_aligned, axis, angle, 1) for angle in theta]) # Keep B along the same direction relative the strain
B_aligned = np.einsum('ij,k->kij', B_aligned, np.ones_like(theta))
B_misaligned = rotate_tensor(np.array([0,0,1]), np.array([0,1,0]), theta_111)
B_z = rotate_tensor(np.array([0,0,1]), np.array([0,1,0]), np.arccos(1/np.sqrt(3)))

def find_coherence(B, alpha, beta, transition_pair=None):
    H = h_total(lambda_soc, q_orb, B[...,2], B[...,0], B[...,1], alpha, beta)
    E = np.linalg.eigvalsh(H)
    dE = (E[...,1]-E[...,0])

    # Decoherence rate without degeneracy correction
    L = getLindbladian( H, g, h, Delta, kBT, transition_pair)
    #T1_0 = thermalisedOrbitalDephasingRate(H, L0, kBT, 'z' )
    T1 = thermalisedOrbitalDephasingRate(H, L, kBT, 'z')
    T2 = thermalisedOrbitalDephasingRate(H, L, kBT, 'y')

    return dE, np.log(T1)/np.log(10), np.log(T2)/np.log(10)

# Plot
fig = plt.figure()
plt.tight_layout()
spec = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

ax_aligned_dE = plt.subplot(spec[0,0])
ax_aligned_dE.set_title('B Aligned', pad=36)
ax_aligned_dE.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
ax_aligned_dE.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
ax_aligned_dE.set_xticklabels([
    r'$\langle\overline{1}\overline{1}0\rangle$',
    r'$\langle\overline{1}00\rangle$',
    r'$\langle\overline{1}10\rangle$',
    r'$\langle 010\rangle$',
    r'$\langle 110\rangle$',
    r'$\langle 100\rangle$',
    r'$\langle 1\overline{1}0\rangle$',
    r'$\langle 0\overline{1}0\rangle$',
    r'$\langle\overline{1}\overline{1}0\rangle$'])
ax_aligned_dE.set_ylabel('$E_1-E_0$\n'+'\n'+r'$\epsilon\times 10^5$')

ax_misaligned_dE = plt.subplot(spec[0,1])
ax_misaligned_dE.set_title(r'B 90$^\circ$ Misaligned', pad=36)
ax_misaligned_dE.tick_params(bottom=False, labelbottom=False, 
left=False, labelleft=False, top=True, labeltop=True)
ax_misaligned_dE.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
ax_misaligned_dE.set_xticklabels([
    r'$\langle\overline{1}\overline{1}0\rangle$',
    r'$\langle\overline{1}00\rangle$',
    r'$\langle\overline{1}10\rangle$',
    r'$\langle 010\rangle$',
    r'$\langle 110\rangle$',
    r'$\langle 100\rangle$',
    r'$\langle 1\overline{1}0\rangle$',
    r'$\langle 0\overline{1}0\rangle$',
    r'$\langle\overline{1}\overline{1}0\rangle$'])

ax_z_dE = plt.subplot(spec[0,2])
ax_z_dE.set_title('B 001', pad=36)
ax_z_dE.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False, top=True, labeltop=True)
ax_z_dE.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
ax_z_dE.set_xticklabels([
    r'$\langle\overline{1}\overline{1}0\rangle$',
    r'$\langle\overline{1}00\rangle$',
    r'$\langle\overline{1}10\rangle$',
    r'$\langle 010\rangle$',
    r'$\langle 110\rangle$',
    r'$\langle 100\rangle$',
    r'$\langle 1\overline{1}0\rangle$',
    r'$\langle 0\overline{1}0\rangle$',
    r'$\langle\overline{1}\overline{1}0\rangle$'])

ax_aligned_T1 = plt.subplot(spec[1,0])
ax_aligned_T1.tick_params(bottom=False, labelbottom=False)
ax_aligned_T1.set_ylabel(r'$log(T_1^{-1})$'+'\n'+r'$\epsilon\times 10^5$')
ax_misaligned_T1 = plt.subplot(spec[1,1])
ax_misaligned_T1.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
ax_z_T1 = plt.subplot(spec[1,2])
ax_z_T1.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

ax_aligned_T2 = plt.subplot(spec[2,0])
ax_aligned_T2.set_ylabel(r'$log(T_2^{-1})$'+'\n'+r'$\epsilon\times 10^5$')
ax_aligned_T2.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
ax_aligned_T2.set_xticklabels([
    r'$-\pi$',
    r'$-\frac{3\pi}{4}$',
    r'$-\frac{\pi}{2}$',
    r'$-\frac{\pi}{4}$',
    '0',
    r'$\frac{\pi}{4}$',
    r'$\frac{\pi}{2}$',
    r'$\frac{3\pi}{4}$',
    r'$\pi$'])

ax_misaligned_T2 = plt.subplot(spec[2,1])
ax_misaligned_T2.tick_params(left=False, labelleft=False)
ax_misaligned_T2.set_xticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
ax_misaligned_T2.set_xticklabels([
    r'$-\pi$',
    r'$-\frac{3\pi}{4}$',
    r'$-\frac{\pi}{2}$',
    r'$-\frac{\pi}{4}$',
    '0',
    r'$\frac{\pi}{4}$',
    r'$\frac{\pi}{2}$',
    r'$\frac{3\pi}{4}$',
    r'$\pi$'])

ax_z_T2 = plt.subplot(spec[2,2])
ax_z_T2.tick_params(left=False, labelleft=False)
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

# Aligned 110
dE_aligned, T1_aligned, T2_aligned = find_coherence(B_aligned, alpha, beta, np.array([[0,1],[2,3]]))
dE_misaligned, T1_misaligned, T2_misaligned = find_coherence(B_misaligned, alpha, beta, None)
dE_z, T1_z, T2_z = find_coherence(B_z, alpha, beta, None)

dE_max = np.max( np.array([dE_aligned, dE_misaligned, dE_z]) )
T_max = np.max( np.array([T1_aligned, T1_misaligned, T1_z, T2_aligned, T2_misaligned, T2_z]) )
dE_min = np.min( np.array([dE_aligned, dE_misaligned, dE_z]) )
T_min = np.min( np.array([T1_aligned, T1_misaligned, T1_z, T2_aligned, T2_misaligned, T2_z]) )

# Aligned plot
ax_aligned_dE.imshow( dE_aligned, vmin=dE_min, vmax=dE_max, cmap='spring', aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
ax_aligned_T1.imshow( T1_aligned, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
ax_aligned_T2.imshow( T2_aligned, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])

ax_misaligned_dE.imshow( dE_misaligned, vmin=dE_min, vmax=dE_max, cmap='spring', aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
ax_misaligned_T1.imshow( T1_misaligned, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
ax_misaligned_T2.imshow( T2_misaligned, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])

h_subplot = .225
v_wspace = .047
im = ax_z_dE.imshow( dE_z, vmin=dE_min, vmax=dE_max, cmap='spring', aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
fig.colorbar(im, cax=fig.add_axes((.92, .88-h_subplot, 0.01, h_subplot)))
im = ax_z_T1.imshow( T1_z, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
fig.colorbar(im, cax=fig.add_axes((.92, .88-v_wspace-2*h_subplot, 0.01, h_subplot)))
im = ax_z_T2.imshow( T2_z, vmin=T_min, vmax=T_max, aspect='auto', extent=[-np.pi,np.pi,-eps_max*1e5,eps_max*1e5])
fig.colorbar(im, cax=fig.add_axes((.92, .88-2*v_wspace-3*h_subplot, 0.01, h_subplot)))

plt.show()