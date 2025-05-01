import torch
import itertools
import numpy as np
import qiskit.quantum_info as qi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def single_sys_partial_trace(X, d_local, sys2btraced):

    dN = int(X.shape[0]) # Size of X. dN = d_local^N
    N = np.round(np.log(dN)/np.log(d_local), decimals=0) # N: Number of qudits
    N = int(N)

    d1 = d_local**(sys2btraced)
    I1 = np.identity(d1)
    d2 = d_local**(N-sys2btraced-1)
    I2 = np.identity(d2)

    v = np.identity(d_local)
    bra = np.kron(I1, np.kron(v[0],I2))
    Y = bra @ X @ bra.T

    for i in range(1,d_local):
        bra = np.kron(I1, np.kron(v[i],I2))
        Y += bra @ X @ bra.T

    return Y


def partialTrace(X, d_local, complement):
    """
    X: nxn matrix to be traced
    sub_systems: sub systems to trace
    d: local dimension
    """
    complement = sorted(complement)

    for n,label in enumerate(complement):
        if n > 0:
            X = single_sys_partial_trace(X, d_local, label-n)
        else:
            X = single_sys_partial_trace(X, d_local, label)
    return X


def swapper(d):

    p = 0
    Id = np.identity(d)
    for i in range(d):
        for j in range(d):
            v = np.outer(Id[:,i],Id[:,j])
            u = np.transpose(v)
            p += np.kron(v,u)

    return p


def Pj( in_label, marginal, dl, num_of_qudits, swapper_d ):
    """
    dl: local dimension
    marginal: reduced system with labels given in the tuple "in_label"
    """

    label = in_label
    n = num_of_qudits - int(np.log(marginal.shape[0])/np.log(dl))
    # n = num_of_qudits - len( marginal.dims() ) 
    dims = tuple( [ dl for i in range( num_of_qudits ) ] )
    swapped_matrix = kron( marginal.data, np.identity( dl**n ) )
    
    all_labels = [ i for i in range( num_of_qudits ) ]
    right_labels = [ i for i in range( list( label )[-1] + 1, num_of_qudits ) ]
    left_labels = [ i for i in range( list( label )[0] ) ]
    
    if left_labels + list( label ) + right_labels == all_labels:
        nl  = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = kron( Il, marginal.data, Ir )
        return swapped_matrix/np.trace(swapped_matrix)
    else:
        nl = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = kron( Il, marginal.data, Ir )
        label = tuple( left_labels + list( label ) )
        
    length = len( label )
    remaining = tuple( [ i for i in range( length ) ] )
    
    while length > 0 and label != remaining:
        
        last = label[-1]
        numOfswapps = np.abs( last  - length ) 
        l1, l2 = length - 1, num_of_qudits - ( length + 1 )
        I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
        gate = kron( I1, swapper_d, I2 ) 
        swapped_matrix = gate @ swapped_matrix @ gate
        
        for i in range( numOfswapps ):
            l1, l2 = l1 + 1, l2 - 1 
            I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
            gate = kron( I1, swapper_d, I2 )
            swapped_matrix = gate @ swapped_matrix @ gate

        label = tuple( list( label[:-1] ) )
        length = len( label )
        remaining = tuple( [ i for i in range( length ) ] ) 
    
    return swapped_matrix/np.trace(swapped_matrix)


def kron(*matrices):
    
    m1, m2, *ms = matrices
    m3 = np.kron(m1, m2)
    
    for m in ms:
        m3 = np.kron(m3, m)
    
    return m3


def compute_marginals_distance(rho0, prescribed_marginals,d,num_of_qudits):
    
    all_systems = set( list( range(num_of_qudits)) )
    marginal_hsd = 0
    projected_marginals = {}
    
    for l in list( prescribed_marginals.keys() ):
        antisys = tuple( all_systems - set(l) )
        projected_marginals[l] = partialTrace(rho0 , d, list( antisys ) )
        marginal_hsd += np.linalg.norm( projected_marginals[l] - prescribed_marginals[l])**2
    
    norm = len( list( prescribed_marginals.keys() ) )
    marginal_hsd = np.sqrt(marginal_hsd/norm)
    
    return projected_marginals, marginal_hsd


def get_marginals(rho_in, d, num_of_qudits, labels_marginals):

    dn = d**num_of_qudits

    marginals = {}
    all_systems = set( list( range(num_of_qudits)) )

    for s in labels_marginals:
        tracedSystems = tuple( all_systems - set( s ) )
        if len(tracedSystems) > 0:
            marginals[s] = partialTrace(rho_in,d,list(tracedSystems))
        else:
            marginals[s] = partialTrace(rho_in,d,list(s))
            
    return marginals


def physical_imposition_operator(d: int, 
                                 num_of_qudits: int, 
                                 x_0: np.array, 
                                 prescribed_marginals: dict, 
                                 swapper_d: np.array) -> np.array:
    all_systems = set( list( range(num_of_qudits)) )
    for l in list( prescribed_marginals.keys() ):
        antisys = tuple( all_systems - set( l ) )
        tr_rho0_I = partialTrace( x_0 , d, list( antisys ) )
        x_0 = (x_0  + Pj( l, prescribed_marginals[l], d, num_of_qudits, swapper_d ) -
               Pj( l, tr_rho0_I, d, num_of_qudits, swapper_d )  )

    return x_0


def tensor_single_sys_ptrace(X, d_local, sys2btraced):

    dN = X.shape[-1] # Size of X. dN = d_local^N
    N = np.round(np.log(dN)/np.log(d_local), decimals=0) # N: Number of qudits
    N = int(N)

    d1 = d_local**(sys2btraced)
    I1 = torch.eye(d1).to(device)

    d2 = d_local**(N-sys2btraced-1)
    I2 = torch.eye(d2).to(device)

    v = torch.eye(d_local).to(device)
    bra = torch.kron(I1, torch.kron(v[0],I2))
    Y = bra @ X @ bra.T

    for i in range(1,d_local):
        bra = torch.kron(I1, torch.kron(v[i],I2))
        Y += bra @ X @ bra.T

    return Y


def tensor_ptrace(X, d_local, complement):
    """
    X: torch tensor
    complement: subsystems to trace
    d: local dimension
    """
    complement = sorted(complement)

    for n,label in enumerate(complement):
        if n > 0:
            X = tensor_single_sys_ptrace(X, d_local, label-n)
        else:
            X = tensor_single_sys_ptrace(X, d_local, label)
    return X


def get_state_of_rank(rank,d,num_of_qudits):

    if rank == 1:
        return qi.random_statevector(d**num_of_qudits).to_operator().data

    rho_gen = qi.random_density_matrix(rank).data

    U = qi.random_unitary(rank).data

    prob_dist = np.diag(U @ rho_gen @ np.conj(U.T) )

    rho = prob_dist[0]*qi.random_statevector(d**num_of_qudits).to_operator()

    for pi in prob_dist[1:]:
        rho += pi*qi.random_statevector(d**num_of_qudits).to_operator()

    rho = rho.data

    return rho/np.trace(rho)


def tensor_to_state(tensor):
    
    rho = tensor[0] + 1j*tensor[1]
    rho = (rho + torch.conj(rho.T))/2
    rho = rho.detach().numpy()

    return rho/np.trace(rho)


def random_testing(number_of_samples,d,N,k,mdl, device):
    
    fidelities = []
    labels_marginals = list(itertools.combinations( range( N) , r = k))  
    swapper_d = swapper(d)
    success_rate = 0

    for _ in range(number_of_samples):

        rho_noisless = get_state_of_rank( d**N, d, N)
        target_marginals = get_marginals( rho_noisless, d, N, labels_marginals )

        initial_seed = qi.random_density_matrix(d**N).data
        rho_noisy = physical_imposition_operator(d,N,
                    initial_seed ,
                    target_marginals,
                    swapper_d
                    )
        tensor_rho_noisy = torch.Tensor( np.stack((rho_noisy.real, rho_noisy.imag)) ).to(device)
        output = mdl( tensor_rho_noisy ).cpu()
        predicted_state_np = tensor_to_state(output)

        predicted_marginals = get_marginals(predicted_state_np, 2, N, labels_marginals)
        eigenvals = np.linalg.eigvalsh(predicted_state_np)

        eigenvals[abs(eigenvals) < 1e-10] = 0
        
        if np.all(eigenvals >= 0):
            success_rate += 1
            
            for label in target_marginals.keys():
                fidelities.append(qi.state_fidelity(target_marginals[label], predicted_marginals[label], validate=True))

    avg_fidelity = np.array(fidelities).mean()

    return avg_fidelity, success_rate/number_of_samples



