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

import cirq
from qusetta import Qiskit
from qusetta import Cirq, Quasar
from bitstring import BitArray


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

def samples_to_clifford_amplitudes(phi, samples, weights, N):
    samples_array = samples.numpy()
    clifford_amplitudes = []
        
    for i in range(len(samples_array)):
        sample_i = samples_array[i]
        index = int("".join(str(x) for x in sample_i), 2)        
        clifford_amplitudes.append(  phi.inner_product_of_state_and_x(index) ) 
    
    return np.array(clifford_amplitudes, dtype = np.complex128), weights, samples


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# now we write function that will measure in random Clifford circuits
def random_Clifford_shadow(circuit, num_shadow):
    
    N = circuit.num_qubits
    
    #rng = np.random.default_rng(1717)
    cliffords = [qiskit.quantum_info.random_clifford(N) for _ in range(num_shadow)]

    phi_list = []
    for cliff in cliffords:
        qiskit_circuit  = circuit.compose(cliff.to_circuit())
        cirq_circuit = Cirq.from_qiskit(qiskit_circuit)
        
        qubits = cirq_circuit.all_qubits()
        cirq_circuit.append(cirq.M(qubits) )

        state = cirq.StabilizerStateChForm(num_qubits=N)
        classical_data = cirq.ClassicalDataDictionaryStore()
        for op in cirq_circuit.all_operations():
            args = cirq.StabilizerChFormSimulationState(
                qubits=list(cirq_circuit.all_qubits()),
                prng=np.random.RandomState(),
                classical_data=classical_data,
                initial_state=state,
            )
            cirq.act_on(op, args)
        measurements = {str(k): list(v[-1]) for k, v in classical_data.records.items()}
        
        for reg, bit in measurements.items():
            state = cirq.StabilizerStateChForm(num_qubits = N, initial_state = BitArray(bit).uint)
            cirq_circuit = Cirq.from_qiskit(cliff.adjoint().to_circuit())

            for op in cirq_circuit.all_operations():
                args = cirq.StabilizerChFormSimulationState(
                    qubits=list(cirq_circuit.all_qubits()),
                    initial_state=state,
                )
                cirq.act_on(op, args)

            phi = args.state
            phi_list.append(phi)
        
    return phi_list


def GHZ_circuit(N):
    
    circuit = qiskit.circuit.QuantumCircuit(N)
    
    # note: qubit 0 corresponds to the rightmost element / LSB of the statvetcor index
    circuit.h(0)
    circuit.s(0)
    for i in range(N-1):
        circuit.cx(i,i+1)
    
    return circuit
