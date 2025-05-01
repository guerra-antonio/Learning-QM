import torch
import numpy as np
import cvxpy as cp

import qiskit.quantum_info as qi
from torch import nn
import matplotlib.pyplot as plt


class ConvDenoiser(nn.Module):

    def __init__(self, num_of_channels):
        super(ConvDenoiser, self).__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels = num_of_channels.get('encoder_in_channels_0'),
                          out_channels = num_of_channels.get('encoder_out_channels_0'),
                          kernel_size = 3, 
                          padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(in_channels = num_of_channels.get('encoder_out_channels_0'),
                          out_channels = num_of_channels.get('encoder_out_channels_1'),
                          kernel_size = 3,
                          padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(in_channels = num_of_channels.get('encoder_out_channels_1'),
                          out_channels = num_of_channels.get('encoder_out_channels_2'),
                          kernel_size = 3,
                          padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2)
            )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels = num_of_channels.get('encoder_out_channels_2'),
                                   out_channels = num_of_channels.get('decoder_out_channels_0'),
                                   kernel_size = 3,
                                   padding=1,
                                   stride=2),
                nn.Tanh(),

                nn.ConvTranspose2d(in_channels = num_of_channels.get('decoder_out_channels_0'),
                                   out_channels = num_of_channels.get('decoder_out_channels_1'),
                                   kernel_size = 5,
                                   padding=1,
                                   stride=2),
                nn.Tanh(),


                nn.ConvTranspose2d(in_channels = num_of_channels.get('decoder_out_channels_1'),
                                   out_channels = num_of_channels.get('decoder_out_channels_2'),
                                   kernel_size = 6,
                                   stride=2),
                nn.Tanh(),
                nn.Conv2d(in_channels = num_of_channels.get('decoder_out_channels_2'),
                          out_channels = num_of_channels.get('encoder_in_channels_0'),
                          kernel_size = 3)
            )

    def forward(self,x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output



def partial_trace_cvxpy(rho, d_local, N, complement):
    
    total_num_of_parts = N
    complement = sorted(tuple(complement))
    rho_copy = rho.copy()
    for n, label in enumerate(complement):
        if n > 0:
            rho_copy = cp.partial_trace(rho_copy, dims=tuple([d_local]*total_num_of_parts), axis=label - n) 
        else:
            rho_copy = cp.partial_trace(rho_copy, dims=tuple([d_local]*total_num_of_parts), axis=label) 
        total_num_of_parts -= 1
    
    return rho_copy


def sdp_solver(marginals, N, solver = cp.SCS, max_iters = 2000):

    d = 2

    X = cp.Variable((d**N, d**N), complex=True)

    L = set(range(N))
    constraints = [X >> 0, X.H == X]
    constraints += [ partial_trace_cvxpy( X, d, N, L - set(l) ) == marginal for l, marginal in marginals.items()]
    prob = cp.Problem(cp.Maximize(1), constraints)
    prob.solve(solver = solver, max_iters=max_iters)    
    
    return X.value



def single_sys_partial_trace(X, d_local, sys2btraced):
    """
    Computes the partial trace over a single subsystem of a multi-qudit quantum system.

    Parameters:
    X (np.ndarray): The input density matrix of the full system.
    d_local (int): The local dimension of each qudit (e.g., 2 for qubits).
    sys2btraced (int): The index of the subsystem to trace out (0-based index).

    Returns:
    np.ndarray: The resulting density matrix after the partial trace over the specified subsystem.
    """

    # Calculate the number of qudits in the system (N)
    dN = int(X.shape[0])  # Dimension of the full system: dN = d_local^N
    N = int(np.log(dN) / np.log(d_local))  # Number of qudits N
    
    # Define the identity matrices for the dimensions before and after the subsystem to trace out
    d1 = d_local**(sys2btraced)  # Dimension of the subsystems before sys2btraced
    d2 = d_local**(N - sys2btraced - 1)  # Dimension of the subsystems after sys2btraced
    I1 = np.identity(d1)
    I2 = np.identity(d2)

    # Initialize the result matrix Y as zero with the proper shape
    Y = np.zeros((d1 * d2, d1 * d2), dtype=complex)

    # Create the identity matrix for the local subsystem
    v = np.identity(d_local)
    
    # Perform the partial trace by summing over the basis states of the traced subsystem
    for i in range(d_local):
        bra = np.kron(I1, np.kron(v[i], I2))
        Y += bra @ X @ bra.T

    return Y

        
def partialTrace(X, d_local, complement):
    """
    Computes the partial trace of a multi-qudit quantum state over specific subsystems.

    Parameters:
    X (np.ndarray): The input density matrix of the full system to be traced out.
    d_local (int): The local dimension of each qudit (e.g., 2 for qubits).
    complement (list): List of subsystems to trace out (0-based indexing).

    Returns:
    np.ndarray: The reduced density matrix after tracing out the specified subsystems.
    """
    
    # Sort the list of subsystems to trace in ascending order
    complement = sorted(complement)
    
    # Perform the partial trace over each specified subsystem
    for n, label in enumerate(complement):
        # For each subsystem, call single_sys_partial_trace
        # Adjust the label of the subsystem due to the reduced dimension after each trace
        if n > 0:
            X = single_sys_partial_trace(X, d_local, label - n)
        else:
            X = single_sys_partial_trace(X, d_local, label)
    
    return X


def swapper(d):
    """
    Creates the swap operator P for a system of dimension d.
    
    The swap operator acts on two qudits, and it exchanges the states of both qudits.
    
    Parameters:
    d (int): The dimension of the qudits.
    
    Returns:
    np.ndarray: The swap operator matrix of dimension (d^2, d^2).
    """
    
    # Initialize the swap operator as a zero matrix of size d^2 x d^2
    P = np.zeros((d**2, d**2))
    
    # Identity matrix of dimension d
    Id = np.identity(d)
    
    # Construct the swap operator using outer products and Kronecker products
    for i in range(d):
        for j in range(d):
            # Outer product of basis vectors |i⟩⟨j|
            v = np.outer(Id[:, i], Id[:, j])
            
            # Transpose of the outer product, i.e., |j⟩⟨i|
            u = v.T
            
            # Add the contribution of v ⊗ u to the swap operator P
            P += np.kron(v, u)
    
    return P


def Pj(in_label, marginal, dl, num_of_qudits, swapper_d):
    """
    Constructs a swapped matrix based on the input label and marginal, 
    applying the swap operator as needed and normalizing the result.
    
    Parameters:
    in_label (tuple): Labels of the subsystems in the reduced system.
    marginal (np.ndarray): Reduced density matrix of the subsystem with the given labels.
    dl (int): Local dimension of each qudit.
    num_of_qudits (int): Total number of qudits in the system.
    swapper_d (np.ndarray): Swap operator for dimension 'dl'.

    Returns:
    np.ndarray: The swapped and normalized density matrix.
    """
    
    label = in_label
    # Calculate how many qudits are being traced out
    n = num_of_qudits - int(np.log(marginal.shape[0]) / np.log(dl))

    # Initialize the full system dimensions
    dims = tuple([dl for _ in range(num_of_qudits)])
    
    # Create the initial swapped matrix (adding identity for traced out qudits)
    swapped_matrix = kron(marginal.data, np.identity(dl**n))
    
    # Define labels on the left and right of the target qudits
    all_labels = [i for i in range(num_of_qudits)]
    right_labels = [i for i in range(list(label)[-1] + 1, num_of_qudits)]
    left_labels = [i for i in range(list(label)[0])]
    
    # If the labels are already ordered correctly, no swap is needed
    if left_labels + list(label) + right_labels == all_labels:
        nl = list(label)[0]
        nr = num_of_qudits - nl - len(label)
        Il, Ir = np.identity(dl**nl), np.identity(dl**nr)
        swapped_matrix = kron(Il, marginal.data, Ir)
        return swapped_matrix / np.trace(swapped_matrix)
    
    # Adjust labels and begin swapping
    nl = list(label)[0]
    nr = num_of_qudits - nl - len(label)
    Il, Ir = np.identity(dl**nl), np.identity(dl**nr)
    swapped_matrix = kron(Il, marginal.data, Ir)
    label = tuple(left_labels + list(label))
    
    # Keep track of the remaining number of labels to be swapped
    length = len(label)
    remaining = tuple([i for i in range(length)])
    
    # Swap the labels until they are in the correct position
    while length > 0 and label != remaining:
        
        last = label[-1]
        numOfswaps = abs(last - length) 
        l1, l2 = length - 1, num_of_qudits - (length + 1)
        I1, I2 = np.identity(dl**l1), np.identity(dl**l2)
        gate = kron(I1, swapper_d, I2)
        swapped_matrix = gate @ swapped_matrix @ gate
        
        # Apply the necessary swaps
        for i in range(numOfswaps):
            l1 += 1
            l2 -= 1
            I1, I2 = np.identity(dl**l1), np.identity(dl**l2)
            gate = kron(I1, swapper_d, I2)
            swapped_matrix = gate @ swapped_matrix @ gate

        # Update label and length for the next iteration
        label = tuple(list(label[:-1]))
        length = len(label)
        remaining = tuple([i for i in range(length)])
    
    # Normalize the swapped matrix
    return swapped_matrix / np.trace(swapped_matrix)


def kron(*matrices):
    """
    Computes the Kronecker product of multiple matrices.

    Parameters:
    *matrices: A variable number of matrices to compute the Kronecker product.
    
    Returns:
    np.ndarray: The Kronecker product of the provided matrices.
    """
    
    # Unpack the first two matrices from the input
    m1, m2, *ms = matrices
    
    # Compute the Kronecker product of the first two matrices
    result = np.kron(m1, m2)
    
    # Iteratively compute the Kronecker product with the remaining matrices
    for m in ms:
        result = np.kron(result, m)
    
    return result


def compute_marginals_distance(rho0, prescribed_marginals, d, num_of_qudits):
    """
    Computes the Hilbert-Schmidt distance between the projected marginals of a quantum state
    and the prescribed marginals.

    Parameters:
    rho0 (np.ndarray): The density matrix of the full quantum system.
    prescribed_marginals (dict): A dictionary where the keys are tuples representing subsystems 
                                 and the values are the prescribed marginals for those subsystems.
    d (int): The local dimension of each qudit (e.g., 2 for qubits).
    num_of_qudits (int): The total number of qudits in the system.
    
    Returns:
    projected_marginals (dict): A dictionary where the keys are tuples representing subsystems
                                and the values are the projected marginals computed from rho0.
    marginal_hsd (float): The normalized Hilbert-Schmidt distance between the projected marginals 
                          and the prescribed marginals.
    """
    
    all_systems = set(range(num_of_qudits))  # Set of all qudits in the system
    marginal_hsd = 0  # To accumulate the Hilbert-Schmidt distance
    projected_marginals = {}  # To store the computed projected marginals
    
    # Loop through each key (subsystem) in the prescribed marginals
    for l in prescribed_marginals.keys():
        # Identify the subsystems to trace out (the complement of the current subsystem)
        antisys = tuple(all_systems - set(l))
        
        # Compute the marginal by tracing out the complementary subsystems
        projected_marginals[l] = partialTrace(rho0, d, list(antisys))
        
        # Accumulate the Hilbert-Schmidt distance (squared difference of matrices)
        marginal_hsd += np.linalg.norm(projected_marginals[l] - prescribed_marginals[l]) ** 2
    
    # Normalize the distance by the number of prescribed marginals
    norm = len(prescribed_marginals.keys())
    marginal_hsd = np.sqrt(marginal_hsd / norm)  # Final Hilbert-Schmidt distance
    
    return projected_marginals, marginal_hsd


def get_marginals(rho_in, d, num_of_qudits, labels_marginals):
    """
    Computes the marginal density matrices for a given quantum state.

    Parameters:
    rho_in (np.ndarray): The input density matrix of the full quantum system.
    d (int): The local dimension of each qudit (e.g., 2 for qubits).
    num_of_qudits (int): The total number of qudits in the system.
    labels_marginals (list of tuples): List of tuples where each tuple specifies 
                                       the subsystems for which to compute the marginal.

    Returns:
    dict: A dictionary where the keys are tuples (subsystems) and the values are the 
          corresponding marginal density matrices.
    """
    
    # Initialize the dictionary to store the marginal density matrices
    marginals = {}

    # Set of all qudit indices in the system
    all_systems = set(range(num_of_qudits))

    # Iterate through each subsystem label in labels_marginals
    for s in labels_marginals:
        # Determine which systems to trace out (those not in the current label)
        tracedSystems = tuple(all_systems - set(s))
        
        # If there are systems to trace out, compute the marginal by tracing them out
        if len(tracedSystems) > 0:
            marginals[s] = partialTrace(rho_in, d, list(tracedSystems))
        else:
            # If no systems are traced out, the marginal is just the full density matrix
            marginals[s] = rho_in
            
    return marginals


def physical_imposition_operator(d: int, 
                                 num_of_qudits: int, 
                                 x_0: np.array, 
                                 prescribed_marginals: dict, 
                                 swapper_d: np.array):
    """
    Adjusts the quantum state x_0 to impose the prescribed marginals on it.
    
    Parameters:
    d (int): The local dimension of each qudit (e.g., 2 for qubits).
    num_of_qudits (int): The total number of qudits in the system.
    x_0 (np.ndarray): The input density matrix of the quantum system.
    prescribed_marginals (dict): A dictionary where the keys are tuples representing subsystems 
                                 and the values are the prescribed marginals for those subsystems.
    swapper_d (np.ndarray): The swap operator for dimension d.
    
    Returns:
    np.ndarray: The updated density matrix that incorporates the prescribed marginals.
    """
    
    # Get the set of all qudits (qubit or qudit labels)
    all_systems = set(range(num_of_qudits))
    
    # Iterate over each prescribed marginal
    for l in prescribed_marginals.keys():
        # Find the complementary subsystems (those to trace out)
        antisys = tuple(all_systems - set(l))
        
        # Compute the partial trace of x_0 over the complementary subsystems
        tr_rho0_I = partialTrace(x_0, d, list(antisys))
        
        # Update x_0 by imposing the prescribed marginal and subtracting the current marginal
        x_0 = (x_0 + Pj(l, prescribed_marginals[l], d, num_of_qudits, swapper_d) -
               Pj(l, tr_rho0_I, d, num_of_qudits, swapper_d))
    
    return x_0


def tensor_single_sys_ptrace(X, d_local, sys2btraced, device='cpu'):
    """
    Computes the partial trace over a single subsystem of a multi-qudit quantum state 
    using PyTorch tensors.

    Parameters:
    X (torch.Tensor): The input density matrix of the full quantum system (d_local^N x d_local^N).
    d_local (int): The local dimension of each qudit (e.g., 2 for qubits).
    sys2btraced (int): The index of the subsystem to trace out (0-based index).
    device (str): Device where the tensor calculations are performed ('cpu' or 'cuda').

    Returns:
    torch.Tensor: The reduced density matrix after tracing out the specified subsystem.
    """

    # Compute the number of qudits (N) from the size of X
    dN = X.shape[-1]  # dN = d_local^N
    N = int(np.round(np.log(dN) / np.log(d_local)))  # Number of qudits
    
    # Define identity matrices for dimensions before and after the subsystem to be traced out
    d1 = d_local ** sys2btraced
    I1 = torch.eye(d1).to(device)  # Identity for systems before the traced subsystem

    d2 = d_local ** (N - sys2btraced - 1)
    I2 = torch.eye(d2).to(device)  # Identity for systems after the traced subsystem

    # Initialize Y as the contribution from the first basis vector of the subsystem
    v = torch.eye(d_local).to(device)  # Local subsystem identity (d_local x d_local)
    bra = torch.kron(I1, torch.kron(v[0], I2))  # |0⟩⟨0| with surrounding systems
    Y = bra @ X @ bra.T  # Initial contribution to the partial trace

    # Sum over the remaining basis vectors for the traced subsystem
    for i in range(1, d_local):
        bra = torch.kron(I1, torch.kron(v[i], I2))  # |i⟩⟨i| with surrounding systems
        Y += bra @ X @ bra.T  # Add the contribution to Y

    return Y


def tensor_ptrace(X, d_local, complement):
    """
    Computes the partial trace over multiple subsystems of a quantum state.

    Parameters:
    X (torch.Tensor): The input density matrix as a PyTorch tensor.
    d_local (int): The local dimension of each qudit (e.g., 2 for qubits).
    complement (list): List of subsystems to trace out (0-based indices).

    Returns:
    torch.Tensor: The reduced density matrix after tracing out the specified subsystems.
    """
    
    # Sort the subsystems to trace for proper ordering
    complement = sorted(complement)

    # Apply the single system partial trace iteratively
    for n, label in enumerate(complement):
        if n > 0:
            X = tensor_single_sys_ptrace(X, d_local, label - n)  # Adjust for reduced dimensions
        else:
            X = tensor_single_sys_ptrace(X, d_local, label)
    
    return X


def get_state_of_rank(rank, d, num_of_qudits):
    """
    Generates a quantum state of a given rank for a system with a specified number of qudits.

    Parameters:
    rank (int): The rank of the quantum state (1 for pure states, >1 for mixed states).
    d (int): The local dimension of each qudit (e.g., 2 for qubits).
    num_of_qudits (int): The total number of qudits in the system.

    Returns:
    np.ndarray: The generated quantum state as a density matrix (d^N x d^N).
    """
    
    # If the rank is 1, generate a pure state (state vector)
    if rank == 1:
        return qi.random_statevector(d**num_of_qudits).to_operator().data

    # Generate a density matrix with the specified rank
    rho_gen = qi.random_density_matrix(rank).data

    # Generate a random unitary transformation
    U = qi.random_unitary(rank).data

    # Compute the probability distribution from the eigenvalues of the transformed density matrix
    prob_dist = np.diag(U @ rho_gen @ np.conj(U.T))

    # Generate the mixed state by summing over the weighted pure states
    rho = prob_dist[0] * qi.random_statevector(d**num_of_qudits).to_operator()

    for pi in prob_dist[1:]:
        rho += pi * qi.random_statevector(d**num_of_qudits).to_operator()

    # Normalize the density matrix
    rho = rho.data

    return rho / np.trace(rho)
    

def mio_cdae(d,num_of_qubits, input_marginals, trained_model, device):

    initial_seed = qi.random_density_matrix(d**num_of_qubits).data

    swapper_d = swapper(d)

    rho_noisy = physical_imposition_operator(d,num_of_qubits,
                initial_seed ,
                input_marginals,
                swapper_d
                )
    
    tensor_rho_noisy = torch.Tensor( np.stack((rho_noisy.real, rho_noisy.imag)) ).to(device)
    tensor_rho_noisy = tensor_rho_noisy.to(torch.float32)
    output = trained_model( tensor_rho_noisy )
    output = output.cpu()
    
    predicted_state = output[0] + 1j*output[1]
    predicted_state = (predicted_state + torch.conj(predicted_state.T))/2

    predicted_state_np = predicted_state.detach().numpy()
    
    return predicted_state_np/np.trace(predicted_state_np)



def mio_cdae_mio(d,num_of_qubits, input_marginals, trained_model, device):

    predicted_state_np = mio_cdae(d,num_of_qubits, input_marginals, trained_model, device)
    swapper_d = swapper(d)

    predicted_by_second_mio = physical_imposition_operator(d,num_of_qubits,
                predicted_state_np ,
                input_marginals,
                swapper_d
                )
    
    return predicted_by_second_mio/np.trace(predicted_by_second_mio)



class PlotData:

    def __init__(self, data, name, color, marker='o', linestyle = 'solid'):
        self.data = data
        self.name = name
        self.color = color
        self.marker = marker
        self.ls = linestyle


def plot_std(args,ylims=[0.9,1.002],xlims=[1, 10],pathplot="",legen_loc='center right',xlabel='rank', transparent = True):

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7,2.6))
    for k in args:
        plt.plot(np.array(k.data.columns), k.data.describe().loc['mean'],
                marker=k.marker,
                label=k.name,
                color=k.color,
                linestyle = k.ls,
                linewidth = 1.,
                markersize = 5
                )
    
        plt.fill_between(np.array(k.data.columns),
                        (k.data.describe().loc['mean'] - k.data.describe().loc['std']),
                        (k.data.describe().loc['mean'] + k.data.describe().loc['std']),
                        color=k.color, alpha=0.15)
    
    if len(xlabel)>0:
        plt.xlabel(xlabel, fontsize=13)
    plt.ylabel("Fidelity", fontsize=13)
    plt.legend(loc=legen_loc, fontsize=12)
    plt.ylim(ylims[0],ylims[1])
    plt.xlim(xlims[0],xlims[1])
    plt.xticks(args[0].data.columns,args[0].data.columns,fontsize=13)
    plt.yticks(fontsize=13)
    if len(pathplot)>0:
        plt.savefig(pathplot, bbox_inches='tight', transparent=transparent)
    plt.show()


def plot_medians(data_sets,ylims=[0.9,1.002],xlims=[1, 10],title="",legen_loc='center right',xlabel='rank'):

    plt.style.use('seaborn-v0_8-whitegrid')

    for data in data_sets:

        median = data.data.describe().loc['50%']
        upper_wing = data.data.describe().loc['75%']
        lower_wing = data.data.describe().loc['25%']

        plt.fill_between(list(data.data.columns),
                        lower_wing,
                        upper_wing,
                        color=data.color,
                        alpha=0.15)

        plt.plot(np.array(list(data.data.columns)), median,
                marker=data.marker,
                label=data.name,
                color=data.color,
                linestyle = data.ls,
                linewidth = 1.,
                markersize = 5
        )

    plt.ylabel("Fidelity", fontsize=13)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.legend(loc=legen_loc, fontsize=12)
    plt.ylim(ylims[0],ylims[1])
    plt.xlim(xlims[0],xlims[1])
    plt.xticks(data_sets[0].data.columns,data_sets[0].data.columns,fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
