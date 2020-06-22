import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
Nangle = 30
theta = np.linspace(0, 2*np.pi, Nangle)
B_magnitude = 1 # GHz
B = B_magnitude*np.array( [np.cos(theta), np.sin(theta)] )

# Define strain components alpha, beta
Npts = 50
max_strain = 50 # GHz
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

Lcorr = getLindbladian(H, g, h, Delta, kBT, np.array([[0,1],[2,3]]))
T2 = thermalisedOrbitalDephasingRate(H, Lcorr, kBT, 'x')

rmax = np.max( np.array(T2) )
rmin = np.min( np.array(T2) )

i =0
fig = plt.figure()
text = plt.text(5,5,'{0:.1f}'.format(theta[i]*180/np.pi))
im = plt.imshow(T2[i,:,:], interpolation='bicubic', vmin=rmin, vmax=rmax,  animated=True)


def updatefig(*args):
    global i, Nangle, theta
    i += 1
    i = i%Nangle
    im.set_array(T2[i,:,:])
    text = plt.text(5,5,'{0:.1f}'.format(theta[i]*180/np.pi))
    return im, text

ani = FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()