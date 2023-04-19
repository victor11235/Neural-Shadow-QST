# Neural-Shadow-QST
We demonstrate the numerical implementation of neural-shadow quantum state tomography (NSQST), and apply it on a 6-qubit phase-shifted GHZ state.

## NEM Code
A large portion of the code is directly adopted from neural error mitigation (NEM) repository https://github.com/1QB-Information-Technologies/NEM

`nqs_model`: Neural network quantum states. From NEM.

`exact_solvers`: Contains useful tools for exact diagonlization and transform neural network quantum state into dense Numpy 1-D statevector. We need to use the GenericExactState module to calculate the exact infidelity between our neural network quantum state and target state. From NEM.

`utils`: Contains utilities for complex-valued tensors in PyTorch. From NEM. 

## NSQST Code

`Helper_fun.py`: Contains useful helper functions, including random Clifford generator, constructing GHZ state circuit, basis state samples to corresponding stabilizer state amplitude etc. The stabilizer state amplitude calculation is currently done without the efficient stabilizer formalism, which means that the current implementation will only work for small system size (reproducing 6-qubit numerical results). Stabilizer formalism will be included in future updates. 

`NSQST_Trainer`: Implementation of the NSQST algorithm.

## Demo Notebook

`NSQST_GHZ_demo.ipynb`: Contains a demo of applying NSQST to a phase-shifted 6-qubit GHZ state, in the noiseless case.

## Dependencies
This code works on Python (3.10.9) with Torch (2.0.0), NumPy (1.23.5), and Qiskit (0.42.1).
