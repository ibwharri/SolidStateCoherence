import matplotlib.pyplot as plt
import numpy as np

x = np.array([[0,1],[1,0]])
y = np.array([[0,-1j],[1j,0]])
z = np.array([[1,0],[0,-1]])
I = np.eye(2)

hbz = -np.kron(I,z)
hbx = -np.kron(I,x)
hby = -np.kron(I,y)
hsoc = -np.kron(y,z)
hl = np.kron(y,I)
hegx = np.kron(z,I)
hegy = np.kron(x,I)

def h_total(l, q, bz, bx, by, alpha, beta):
    # Takes Hamiltonian parameters
    # l: spin-orbit splitting
    # q: ratio between orbital and magnetic component of magnetic susceptibility
    # bz, bx, by: magnetic field along z, x direction
    # alpha, beta: Egx/Egy strain field magnitude
    # Parameters are all either integers, or of identical shape (m,n,...)
    # Returns array of Hamiltonians for system of shape (m,n,..., 4, 4)

    param_list = [np.array(param) for param in [l, q, bz, bx, by, alpha, beta] ]
    shapes = [ param.shape for param in param_list]
    shape = shapes[ np.argmax(np.array( [len(shape) for shape in shapes] )) ]

    # Reshape Hamiltonian components and parameters
    h_list = [ h.reshape( (1,)*len(shape) + (4,4) ) for h in [hbz, hbx, hby, hsoc, hl, hegx, hegy] ]

    for i,dim in enumerate(shape):
        h_bz, h_bx, h_by, h_soc, h_l, h_egx, h_egy = tuple(np.repeat(h, dim, axis=i) for h in h_list)
        
    L, Q, Bz, Bx, By, Alpha, Beta = tuple( param.reshape( (1,)*(len(shape)+2) ) if len(shapes[i])==0 else param.reshape( shape + (1,1)) for i,param in enumerate(param_list) )

    h = L*h_soc + Q*Bz*h_l + Bz*h_bz + Bx*h_bx + + By*h_by + Alpha*h_egx + Beta*h_egy

    return h


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
    