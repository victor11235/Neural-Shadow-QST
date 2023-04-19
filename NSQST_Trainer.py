from typing import List, Union, Optional
import torch
import utils
from exact_solvers import GenericExactState
from nqs_models import NeuralQuantumState
from torch.optim import Optimizer
import qiskit
import random

# import helper functions
import Helper_fun as helper


class NSQST_Trainer():
    '''
    The neural-shadow quantum state tomography (NSQST) algorithm
    '''
    
    def __init__(self,
                 nqs_model: NeuralQuantumState,
                 circuit,
                 optimizer: Optimizer,
                 batch_size:int,
                 target_state,
                 max_iters:int=100,
                 num_samples:int=10000,
                 state_file_name:str=None,
                 K:int=1):
        '''

        '''
        self.nqs = nqs_model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.target_state = target_state
        self.max_iters = int(max_iters)
        self.num_qubits = circuit.num_qubits
        self.num_samples = num_samples
        self.K = K
        self.circuit = circuit
        self.state_file_name = state_file_name


    def _step(self, i=0):
        '''
        Training update for one iteration of NSQST. Updates the
        neural quantum state parameters based on the shadow-tomography based infidelity estimate.

        ''' 
        with torch.no_grad():
            # first we collect samples from neural network state
            NN_samples = self.nqs.sample(self.num_samples)
            unique_samples, counts = torch.unique(NN_samples, dim=0, return_counts=True)
            weights = counts.type(torch.double)/self.num_samples
            # create the list for calculating the fidelity later
            prediction_list = []
            
        random.seed()
        cliffords_batch, results_batch = helper.random_Clifford_shadow(self.circuit, num_shadow=self.batch_size)
        
        loss_nsqst = 0
        for cliff, res in zip(cliffords_batch, results_batch):
            
            with torch.no_grad():
                mat  = cliff.adjoint().to_matrix()
                for bit,count in res.items():
                    Ub = mat[:,int(bit,2)] # this is Udag|b>
                    Clifford_sample_log_amplitudes, weights_new, unique_samples_new = \
                        helper.samples_to_log_clifford_amplitudes(Ub, unique_samples, weights)                         
                    NN_sample_log_amplitudes_no_grad = self.nqs.amplitudes(unique_samples_new, return_polar = True)
                    amp_fraction = (utils.Complex(Clifford_sample_log_amplitudes) - NN_sample_log_amplitudes_no_grad).exp()
                    
                    # get the conjugate version of amp_fraction
                    amp_fraction_conj = amp_fraction.conjugate()
                    mean = (utils.Complex(weights_new) * amp_fraction).sum(dim=0)
                    overlap = utils.Complex.abs(mean)**2
                    
                    # append to the list for fidelity estimate
                    overlap_scalar = overlap.detach().numpy()
                    prediction_i = (2**self.num_qubits+1) * overlap_scalar - 1 
                    prediction_list.append(prediction_i)
            # calculate the gradient 
            
            # get real and imgaginary parts of term 1
            log_amplitudes = self.nqs.amplitudes(unique_samples_new, return_polar=True)
            term_1_log_moduli = amp_fraction_conj.real * log_amplitudes.real
            term_1_phases = amp_fraction_conj.imag * log_amplitudes.imag
            term_1_real = (weights_new*(term_1_log_moduli - term_1_phases)).sum(dim=0)       
            term_1_imag = (weights_new*(amp_fraction_conj.real * log_amplitudes.imag + amp_fraction_conj.imag * 
                                        log_amplitudes.real)).sum(dim=0) 
            
            # now since loss = term_1 * term_2, get the real part of the product
            term_2_real = mean.real
            term_2_imag = mean.imag
            loss_real = term_1_real * term_2_real - term_1_imag * term_2_imag
            # propagate the gradient for every shadow, here we accumulate the gradient for every shadow
            loss_nsqst_i = - 2 * (2**self.num_qubits+1) * loss_real / self.batch_size
            loss_nsqst += loss_nsqst_i
        
        # end of the step, we update the model parameters
        torch.nn.utils.clip_grad_norm_(self.nqs.parameters(), 5)
        loss_nsqst.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            cost_fun = 1 - helper.median_of_mean(prediction_list,self.K)
            print('Iteration ' + str(i) )
            print('Cost function is ' + str(cost_fun) )


        return cost_fun

            
    def train(self):
        infidelity_list = []
        cost_fun_list = []
        for i in range(self.max_iters):
            cost_fun = self._step(i=i)           
            infidelity = 1-self.target_state.fidelity_to(self.nqs.full_state())
            
            infidelity_list.append(infidelity.detach().numpy())
            cost_fun_list.append(cost_fun)
            print('Exact infidelity is ' + str(infidelity.detach().numpy() ) )

        if self.state_file_name != None:
            torch.save(self.nqs.state_dict(), self.state_file_name)
        return infidelity_list, cost_fun_list