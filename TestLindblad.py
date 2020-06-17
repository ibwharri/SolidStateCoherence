import matplotlib.pyplot as plt
import numpy as np
from Simulation import *

g = lambda omega: 1+0*omega
h = lambda omega: 1+0*omega
Delta = lambda omega: 1+0*omega

H = h_total(45, .1, 1, 0, 0, 0, 0 )
L0 = getLindbladian(H, g, h, Delta, 83.5 )
# plot_dephasing(L0, 'x')
# plot_dephasing(L0, 'y')
# plot_dephasing(L0, 'z')
# plt.show()

g = lambda omega: np.sqrt(omega)
h = lambda omega: np.sqrt(omega)
Delta = lambda omega: omega**2

#Lcorr = getLindbladian(H, g, h, Delta, 83.5, np.array([[0,2],[1,3]]) )

N = 3
H = h_total(45, .1, 10*np.random.rand(N,N), 0, 0, 0, 0 )

L0 = getLindbladian(H, g, h, Delta, 83.5 )

def plot_dephasing(L, direction):
    plt.figure()
    plt.title(direction)
    D = getDephasingOperator(L, direction)
    Mu = np.zeros_like(D)
    for i in range(4):
        for j in range(4):
            mu = np.kron(sigma[i,:,:], sigma[j,:,:] )
            prod = np.einsum('...im,mj->...ij', D, mu)
            Mu[...,i,j] = np.einsum('...mm->...', prod)

    for i in range(N**2):
        plt.subplot(N,N,i+1)
        if i%3==int(np.floor(N/2)) and np.floor(i/3)==0:
            plt.title(direction)
        plt.imshow(np.real(Mu[i%3, int(np.floor(i/3)),:,:]))
plot_dephasing(L0, 'x')
plot_dephasing(L0, 'y')
plot_dephasing(L0, 'z')
plt.show()