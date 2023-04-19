from typing import Tuple, Optional
import torch
import utils
import numpy as np

import qiskit
from qiskit.quantum_info import Statevector, DensityMatrix
# for noisy clifford sampling and measurement
import qiskit.providers.aer.noise as noise
from qiskit import QuantumCircuit, execute, Aer
from math import pi


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

'''
This function takes unique basis samples and output dense Clifford state log_amplitudes.
In the future, we need to modify this and incorporate stabilizer framework, to achieve O(n^3) time complexity. 
'''
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
        state = Statevector.from_int(0,2**N)
        state_dense = state.evolve(qc_c)
        counts = state_dense.sample_counts(1)
        results.append(counts)
        
    return cliffords, results

# now we write function that will measure in random Clifford circuits, for the noisy case
def random_Clifford_shadow_noisy(circuit, num_shadow, noise_model):
    
    N = circuit.num_qubits
    
    #rng = np.random.default_rng(1717)
    cliffords = [qiskit.quantum_info.random_clifford(N) for _ in range(num_shadow)]

    results = []
    for cliff in cliffords:
        qc_c = circuit.compose(qiskit.compiler.transpile(cliff.to_circuit(), basis_gates = ['rx', 'ry', 'rz', 'cx']) )
        qc_c.measure_all()
        result = execute(qc_c, Aer.get_backend('qasm_simulator'),
                 basis_gates=noise_model.basis_gates,
                 noise_model=noise_model,
                 shots=1,
                 memory=False).result()
        counts = result.get_counts(0)
        results.append(counts)
    return cliffords, results

# now we write function that will sample a density matrix after going through n-qubit depolarizing channel
def random_Clifford_shadow_dep_n(circuit, num_shadow, lamb):
    f = 1 - lamb  
    N = circuit.num_qubits
    
    #rng = np.random.default_rng(1717)
    cliffords = [qiskit.quantum_info.random_clifford(N) for _ in range(num_shadow)]

    results = []
    for cliff in cliffords:
        qc_c = circuit.compose( cliff.to_circuit() )
        state = Statevector.from_int(0,2**N)
        state_dense = state.evolve(qc_c)
        DM = f * DensityMatrix(state_dense) + (1-f) * DensityMatrix( (np.identity(2**N) / (2**N)) )
        counts = DM.sample_counts(1)
        results.append(counts)
        
    return cliffords, results

def GHZ_circuit():
    
    circuit = qiskit.circuit.QuantumCircuit(6)
    
    # note: qubit 0 corresponds to the rightmost element / LSB of the statvetcor index
    # flip qubits
    circuit.h(0)
    circuit.rz(np.pi/2,0)
    circuit.cx(0,1, label='GHZ')
    circuit.cx(1,2, label='GHZ')
    circuit.cx(2,3, label='GHZ')
    circuit.cx(3,4, label='GHZ')
    circuit.cx(4,5, label='GHZ')
    
    circuit.barrier()
    
    return circuit
