import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations

x = np.array([[0,1],[1,0]])
y = np.array([[0,-1j],[1j,0]])
z = np.array([[1,0],[0,-1]])
I = np.eye(2)
sigma = np.array([I,x,y,z])

hbz = -np.kron(I,z)
hbx = -np.kron(I,x)
hby = -np.kron(I,y)
hsoc = np.kron(y,z)
hl = np.kron(y,I)
hegx = -np.kron(z,I)
hegy = np.kron(x,I)

def h_total(l, q, bz, bx, by, alpha, beta):
    # Takes Hamiltonian parameters
    # l: spin-orbit splitting
    # q: ratio between orbital and magnetic component of magnetic susceptibility
    # bz, bx, by: magnetic field along z, x direction
    # alpha, beta: Egx/Egy strain field magnitude
    # Parameters are all either floats, or of identical shape (m,n,...)
    # Returns array of Hamiltonians for system of shape (m,n,..., 4, 4)

    param_list = [np.array(param) for param in [l, q, bz, bx, by, alpha, beta] ]
    shapes = [ param.shape for param in param_list]
    shape = shapes[ np.argmax(np.array( [len(shape) for shape in shapes] )) ]
    if not shape:
        shape = (1,)

    # Reshape Hamiltonian components and parameters
    h_list = [ h.reshape( (1,)*len(shape) + (4,4) ) for h in [hbz, hbx, hby, hsoc, hl, hegx, hegy] ]

    for i,dim in enumerate(shape):
        h_bz, h_bx, h_by, h_soc, h_l, h_egx, h_egy = tuple(np.repeat(h, dim, axis=i) for h in h_list)
        
    L, Q, Bz, Bx, By, Alpha, Beta = tuple( param.reshape( (1,)*(len(shape)+2) ) if len(shapes[i])==0 else param.reshape( shape + (1,1)) for i,param in enumerate(param_list) )

    h = L*h_soc + Q*Bz*h_l + Bz*h_bz + Bx*h_bx + + By*h_by + Alpha*h_egx + Beta*h_egy

    return h

def getLindbladian(H, g, h, Delta, kBT, degenerate_pair = None):
    # Takes a Hamiltonian, H of shape (m,n,...,4,4), functions g(omega), h(omega), Delta(omega) denoting strength of phonon-orbital coupling for the Egx, Egy modes, and the density of states at frequency omega, and thermal energy kBT. Returns Lindbladians in the shape (m,n,...,i,j,4,4) as (m,n,...) list of Lindbladians L_ij. degenerate_spin is a boolean indicating whether transition 12 is degenerate with transition 34

    E, V = np.linalg.eigh(H)


    Sx = -np.kron(z, I)*np.ones_like(H)
    Sx = np.einsum('...ji,...jk,...kl->...il', np.conjugate(V), Sx, V)
    Sy = np.kron(x, I)*np.ones_like(H)
    Sy = np.einsum('...ji,...jk,...kl->...il', np.conjugate(V), Sy, V)

    L = np.zeros_like(H)
    L = L[..., np.newaxis, np.newaxis,:,:]
    L = np.repeat(L, 4, axis=-3)
    L = np.repeat(L, 4, axis=-4)

    for i in range(4):
        nu_i = E[...,i]
        for j in range(4):
            if i==j:
                pass
            else:
                nu_j = E[...,j]
                gij = g(np.abs(nu_i-nu_j))
                hij = h(np.abs(nu_i-nu_j))
                Deltaij = Delta(np.abs(nu_i-nu_j))
                gammaij = n_th(np.abs(nu_i-nu_j), kBT)
                gammaij = gammaij + 1 if i<j else gammaij

                complementary_pair = test_degenerate(i,j,degenerate_pair)
                #print((i,j), 'None' if complementary_pair is None else tuple(complementary_pair))
                if complementary_pair is None:
                    L[...,i,j,i,j] = np.sqrt(2*np.pi*Deltaij*gammaij)*( gij*Sx[...,i,j] + hij*Sy[...,i,j] )
                else:
                    L[...,i,j,i,j] = np.sqrt(np.pi*Deltaij*gammaij)*( gij*Sx[...,i,j] + hij*Sy[...,i,j] )
                    L[...,i,j, complementary_pair[0], complementary_pair[1] ] = np.sqrt(np.pi*Deltaij*gammaij)*( gij*Sx[...,complementary_pair[0], complementary_pair[1]] + hij*Sy[...,complementary_pair[0], complementary_pair[1]] )
            
    
    return L

def getDephasingOperator(L, direction):
    # Takes Lindbladians output from getLindbladian, and a direction (1,2,3) or 'x','y','z' and outputs the decoherence operator
    if direction == 1 or direction == 'x':
        mu = np.kron(I,x)
    elif direction == 2 or direction == 'y':
        mu = np.kron(I,y)
    elif direction == 3 or direction == 'z':
        mu = np.kron(I,z)
    
    D = np.einsum('...ijlk,lm,...ijmn->...kn', np.conjugate(L), mu, L)
    D -= 1/2*np.einsum('...ijlk,...ijlm,mn->...kn', np.conjugate(L), L, mu)
    D -= 1/2*np.einsum('kl,...ijml,...ijmn->...kn', mu, np.conjugate(L), L)

    return D

def getDephasingComponents(D):
    Mu = np.zeros_like(D)
    for i in range(4):
        for j in range(4):
            mu = np.kron(sigma[i,:,:], sigma[j,:,:] )
            #prod = np.einsum('...im,mj->...ij', D, mu)
            Mu[...,i,j] = np.einsum('...im,mi->...', D, mu)/4

    return Mu

def thermalisedOrbitalDephasingRate(H, L, kBT, direction):
    # Evalutes the relaxation rate along a direction given by direction for a list of Hamiltonians of shape ...xNxN, at a thermal energy given by kBT, and Lindbladians of shape ...xNxNxNxNgiven by L under assumption that the orbital degree of freedom. L may be a list of arrays of shape ...xNxNxNxN, in which case the contributions of all operators in the last will be added together.

    direction = 1 if direction=='x' else( 2 if direction=='y' else 3) # Convert directino to int

    E, _ = np.linalg.eigh(H)
    if isinstance(L, list):
        L = np.array(L)
        D = getDephasingOperator(L, direction)
        D = np.einsum('i...kn->...kn', D)
    else:
        D = getDephasingOperator(L, direction)
    Mu = getDephasingComponents(D)
    shape = E.shape[0:-1]

    # Get array of thermal matrices
    rhoth = np.repeat(E[...,np.newaxis], 4, axis=-1)
    rhoth = np.exp(-rhoth/kBT)
    
    # Only non-zero diagonal elements
    ii = np.arange(0,4,dtype=np.int64)
    ii = ii.reshape( (1,)*len(shape)+(4,1))
    ii = np.tile(ii, shape+(1,4) )
    jj = np.arange(0,4,dtype=np.int64)
    jj = jj.reshape( (1,)*len(shape)+(1,4))
    jj = np.tile(jj, shape+(4,1) )
    rhoth[ ii != jj ] = 0

    # Normalise
    rhoth = rhoth/np.einsum('...ii->...',rhoth)[...,np.newaxis,np.newaxis]
    
    # Calculate orbital polarisation
    muZI_th = np.einsum('...im,mj->...', rhoth, np.kron(z, I)/4 )

    # Spin dephasing rate under thermalise orbital approximation
    r = -np.real(Mu[...,0,direction] + Mu[...,3,direction]*muZI_th)
    
    return r

def basisSimilarity(U, V):
    # Given two lists of unitaries U and V of size ...xmxm, returns maximum ||U.V^T|| for every possible permutation of the columns of U
    prod = np.einsum( '...ji,...jk->...ik', np.conjugate(U), V )
    sigma = np.einsum( '...ij->...', np.abs(prod))/prod.shape[-1]
    return sigma

def phononPolarisationRatio(Sx, Sy, V):
    # Takes two polarised scattering matrices, and returns ratio of relative scattering rates of polarised phonons off 1 and 0 qubit states

    rx = np.einsum('...ji,jk,...kl->...il', np.conjugate(V), Sx, V)
    ry = np.einsum('...ji,jk,...kl->...il', np.conjugate(V), Sy, V)

    r0x = rx[...,0,2]
    r1x = rx[...,1,3]
    r0y = ry[...,0,2]
    r1y = ry[...,1,3]

    r = np.abs(r0x*r1y/(r0y*r1x))**2

    return r

def test_degenerate(i,j,pairs):
    if pairs is None:
        return None
    elif i in pairs[0,:] and j in pairs[0,:]:
        return order_pair(i,j, pairs[1,:] )
    elif i in pairs[1,:] and j in pairs[1,:]:
        return order_pair(i,j, pairs[0,:] )
    elif i in pairs[:,0] and j in pairs[:,0]:
        return order_pair(i,j, pairs[:,1])
    elif i in pairs[:,1] and j in pairs[:,1]:
        return order_pair(i,j, pairs[:,0] )
    else:
        return None

def order_pair(i,j,pair):
    # Order piar like i, j
    pair.sort()
    if j>i:
        return pair
    else:
        return pair[::-1]

def n_th(E,kBT):
    return 1/(np.exp(E/kBT)-1)

def Pi(m,n):
    Pi = np.zeros((4,4))
    Pi[m,n] = 1

    return Pi

def rotate_tensor(T, a, theta=None, order=1):
    # Takes ...x3x... list of 3D tensors, T of rank order, and rotates them around axis defined by a. If rotation angle theta is not defined, angle of rotation is magnitude of a.

    assert order<14, "Order of tensor is too high for numpy.einsum to handle"
    
    # Find rotation matrix
    if theta is None:
        theta = np.linalg.norm(a)

    a = a/np.linalg.norm(a)

    Cpm = np.array([[    0,   -a[2],  a[1]],
                    [  a[2],     0  ,-a[0]],
                    [ -a[1],   a[0],    0 ]])

    Rn = np.cos(theta)*np.eye(3) + np.sin(theta)*Cpm + (1-np.cos(theta))*np.einsum('i,j->ij',a,a)

    # Get subscript
    letters = 'abcdefghijklmnopqrstuvwxyz'
    subscript = [ letters[2*i:2*i+2] + ',' for i in range(int(order/2+1)) ]
    subscript.append( '...' + letters[1:2*order+1:2]+'->' )
    subscript.append( '...' + letters[:2*order:2] )
    subscript = "".join(subscript)

    # Get operands
    op = [Rn]*order
    op.append(T)

    # Perform transformation
    T_transformed = np.einsum(subscript, *op)

    return T_transformed

find_alpha = lambda epsilon, d, f: d*(epsilon[...,0,0]-epsilon[...,1,1]) + f*epsilon[...,2,0]
find_beta = lambda epsilon, d, f: -2*d*(epsilon[...,0,1])  + f*epsilon[...,1,2]