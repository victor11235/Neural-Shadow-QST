import os
import sys
path = '/Users/victorwei/Research projects/VQE_2022_summer/NEM_SU2/code'
sys.path.append(path)

from typing import Tuple, Optional
import torch
import utils
import numpy as np

import qiskit


# assert that num_shadows divisible by K
def median_of_mean(data_list, K):
    assert len(data_list) % K == 0
    
    groups = np.split(np.array(data_list), K)
    median_list = []
    for group_i in groups:
        mean_i = np.mean(group_i)
        median_list.append(mean_i)

    return np.median(median_list)

def overlap(state, edState, N):
    overlap = np.dot(state.conj().reshape(2**N, 1).T, edState.reshape(2**N, 1))
    fidelity = (np.linalg.norm(overlap))**2
    return fidelity

def torch_delete(tensor, indices):
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


# this function takes unique basis samples and output dense Clifford state log_amplitudes 
# needs to be changed for larger system size, switching to stabilizer framework
def samples_to_log_clifford_amplitudes(clifford_dense, samples, weights):
    samples_array = samples.numpy()
    
    index_list = []
    clifford_log_amplitudes = []
    for i in range(len(samples_array)):
        sample_i = samples_array[i]
        index = int("".join(str(x) for x in sample_i), 2)
        
        # we have to be careful of whether the amplitude is zero or not
        if clifford_dense[index] == 0.0:
            index_list.append(i)
        else:
            clifford_log_amplitudes.append( np.log(clifford_dense[index]) )
    # new weights with zero-valued basis excluded
    weights_new = torch_delete(weights, index_list)
    unique_samples_new = torch_delete(samples, index_list)
    
    # we have to be careful when new weights and unique_samples are empty
    if len(weights_new) == 0:
        return np.ones(1), torch.tensor([0]), samples[[0]]
    
    return np.array(clifford_log_amplitudes), weights_new, unique_samples_new

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# now we write function that will measure in random Clifford circuits
def random_Clifford_shadow(circuit, num_shadow):
    
    N = circuit.num_qubits
    
    #rng = np.random.default_rng(1717)
    cliffords = [qiskit.quantum_info.random_clifford(N) for _ in range(num_shadow)]

    results = []
    for cliff in cliffords:
        qc_c  = circuit.compose(cliff.to_circuit())

        counts = qiskit.quantum_info.Statevector(qc_c).sample_counts(1)
        results.append(counts)
        
    return cliffords, results
