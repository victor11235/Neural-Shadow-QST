# Neural-Shadow-QST
We demonstrate the numerical implementation of neural-shadow quantum state tomography (NSQST), and apply it on a 6-qubit phase-shifted GHZ state.

## NEM Code
A large portion of the code is directly adopted from neural error mitigation (NEM) repository https://github.com/1QB-Information-Technologies/NEM

`nqs_model`: Neural network quantum states. From NEM.

`exact_solvers`: Contains useful tools for exact diagonlization and transform neural network quantum state into dense Numpy 1-D statevector. We need to use the GenericExactState module to calculate the exact infidelity between our neural network quantum state and target state. From NEM.

`utils`: Contains utilities for complex-valued tensors in PyTorch. From NEM. 

## NSQST Code
`data`: Contains useful data for NSQST with pre-training, including the 40-qubit pre-trained amplitude neural network parameters and 200 Clifford shadows of the 40-qubit phase-shifted GHZ state.

`Helper_fun.py`: Contains useful helper functions, including random Clifford generator, constructing GHZ state circuit, basis state samples to corresponding stabilizer state amplitude etc. The stabilizer state amplitude calculation is currently done without the efficient stabilizer formalism, which means that the current implementation will only work for small system size (reproducing 6-qubit numerical results). Stabilizer formalism will be included in future updates. Other modifications may also be needed to make NSQST scalable, see Appendix D of the paper.

`Helper_fun_new.py`: Contains useful helper functions for reproducing NSQST with pre-training beyond 6 qubits, where stabilizer formalism is implemented using Cirq.

`NSQST_Trainer.py`: Implementation of the NSQST algorithm.

`NSQST_Pre_Trainer_GHZ.py`: Implementation of the NSQST with pre-training algorithm with efficient stabilizer formalism included. Exact infidelity calculated in every iteration for the phase-shifted GHZ state.

## Demo Notebook

`NSQST_GHZ_demo.ipynb`: Contains a demo of applying NSQST to a phase-shifted 6-qubit GHZ state, in the noiseless case.

`NSQST_with_pre_training_demo.ipynb`: Contains a demo of applying NSQST with pre-training to a phase-shifted 40-qubit GHZ state, in the noiseless case.


## Dependencies
The code works on Python (3.10.9) with Torch (2.0.0), NumPy (1.23.5), Qiskit (0.42.1), and Cirq (1.2.0).


