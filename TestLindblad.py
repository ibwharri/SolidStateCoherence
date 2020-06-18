import matplotlib.pyplot as plt
import numpy as np
from Simulation import *

g = lambda omega: 1+0*omega
h = lambda omega: 1+0*omega
Delta = lambda omega: 1+0*omega
kBT = 83.5

H = h_total(45, .1, 1, 0, 0, 0, 0 )
L0 = getLindbladian(H, g, h, Delta, kBT )
# plot_dephasing(L0, 'x')
# plot_dephasing(L0, 'y')
# plot_dephasing(L0, 'z')
# plt.show()

g = lambda omega: np.sqrt(omega)
h = lambda omega: np.sqrt(omega)
Delta = lambda omega: omega**2

#Lcorr = getLindbladian(H, g, h, Delta, 83.5, np.array([[0,2],[1,3]]) )

N = 5
Bz = 50*np.random.rand(N,N)
H = h_total(45, .1, Bz, 0, 0, 0, 0 )

L0 = getLindbladian(H, g, h, Delta, kBT )



def plot_dephasing(L, direction):
    plt.figure()
    D = getDephasingOperator(L, direction)
    Mu = getDephasingComponents(D)

    for i in range(N**2):
        plt.subplot(N,N,i+1)
        plt.title('Bz={0:.2f}'.format(Bz[i%N, int(np.floor(i/N))]) )
        if i%3==int(np.floor(N/2)) and np.floor(i/3)==0:
            plt.title('{0}\nBz={1:.1f}'.format(direction,Bz[i%N, int(np.floor(i/N))]) )
        plt.imshow(np.real(Mu[i%N, int(np.floor(i/N)),:,:]))
plot_dephasing(L0, 'x')
plot_dephasing(L0, 'y')
plot_dephasing(L0, 'z')
plt.show()