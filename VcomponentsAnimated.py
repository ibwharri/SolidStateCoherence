import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from Simulation import *
from matplotlib.animation import FuncAnimation

g = lambda omega: np.sqrt(omega)
h = lambda omega: np.sqrt(omega)
Delta = lambda omega: omega**2
kBT = 83.5 # GHz
lambda_soc = 45/2 # GHz
q_orb = .1

# Sweep radial field conditions
# =============================
# Define magnetic field components
Nangle = 50
theta = np.linspace(0, np.pi*2, Nangle)
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
E, V = np.linalg.eigh(H)
dE = (E[...,1]-E[...,0])

S = np.kron(z, I) + np.kron(x, I)
Mu = getDephasingComponents(S)
plt.figure()
plt.imshow(Mu)
S = S*np.ones_like(H)
S = np.einsum('...ji,...jk,...kl->...il', np.conjugate(V), S, V)


# Components of S
Mu = getDephasingComponents(S)
Mu_min = np.real(np.min(Mu))
Mu_max = np.real(np.max(Mu))

fig = plt.figure()
def init(*args):
    global imgs, angle, title
    angle = 0
    imgs = [1]*16
    for angle in range(Nangle):
        title = fig.suptitle('{0}={1:.1f}'.format(r'$\theta$', theta[angle]*180/np.pi))
        spec = gridspec.GridSpec(nrows = 4, ncols=4, figure=fig )

        for i in range(4):
            for j in range(4):
                ax = plt.subplot(spec[i,j])
                imgs[i+4*j] = plt.imshow( np.real(Mu[angle,:,:,i,j]), animated=True, vmin = Mu_min, vmax = Mu_max )
    
    return tuple(imgs) + tuple([title])

def updatefig(*args):
    global Nangle, theta, angle, imgs, title
    angle += 1
    angle = angle%Nangle

    
    title.set_text('{0}={1:.1f}'.format(r'$\theta$', theta[angle]*180/np.pi))
    

    for i in range(4):
        for j in range(4):
            imgs[i+4*j].set_array( np.real(Mu[angle,:,:,i,j]) )
    
    return tuple(imgs) + tuple([title])

ani = FuncAnimation(fig, updatefig, init_func=init, interval=50)
plt.show()