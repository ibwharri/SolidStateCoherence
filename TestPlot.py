import matplotlib.pyplot as plt
import numpy as np
from Simulation import *

# Plot eigenvalues as a function of magnetic field/strain
for i in range(4):
    # Compare manual calculation to h_total function
    alpha = np.linspace(0,2,100)
    beta = .5*i
    h = np.repeat(hsoc[np.newaxis,:,:],100,axis=0) + beta*np.repeat(hbx[np.newaxis,:,:],100,axis=0) + alpha[:,np.newaxis,np.newaxis]*np.repeat(hbz[np.newaxis,:,:],100,axis=0)
    E = np.linalg.eigvals(h)
    E = np.sort(E)
    plt.subplot(3,4,i+1)
    plt.plot(alpha,E)
    plt.title( r'$\beta={}$'.format(beta))

    h = h_total(1,0,alpha,beta,0,0,0)
    E = np.linalg.eigvals(h)
    E = np.sort(E)
    plt.subplot(3,4,i+5)
    plt.plot(alpha,E)

    # Plot diagonal components of transformed H against calculated eigenvalues (should be same)
    _, V = np.linalg.eigh(h)
    Ediag = np.einsum('nji,njk,nki->ni', np.conjugate(V), h, V)
    plt.subplot(3,4,i+9)
    plt.plot(E, Ediag)
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.plot(xlim,xlim,'k:') # Plot y=x line
plt.show()