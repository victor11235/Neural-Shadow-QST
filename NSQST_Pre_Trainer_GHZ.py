from typing import List, Union, Optional
import torch
import utils
from exact_solvers import GenericExactState
from nqs_models import NeuralQuantumState
from torch.optim import Optimizer
import qiskit
import random
import numpy as np

# import helper functions
import Helper_fun_new as helper


class ShadowTomographyTrainer():
    '''
    The neural quantum state tomography algorithm using classical-shadow based cost function
    '''
    
    def __init__(self,
                 N,
                 nqs_model_amp: NeuralQuantumState,
                 nqs_model_phase:NeuralQuantumState,
                 phi_list,
                 optimizer: Optimizer,
                 max_iters:int=100,
                 num_samples:int=10000,
                 batch_size = 200,
                 state_file_name_amp:str=None,
                 state_file_name_phase:str=None,
                 K:int=1
                ):
        '''

        '''
        self.N = N
        self.phi_list = phi_list
        self.nqs_amp = nqs_model_amp
        self.nqs_phase = nqs_model_phase
        self.optimizer = optimizer
        self.max_iters = int(max_iters)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.K = K
        self.state_file_name_amp = state_file_name_amp
        self.state_file_name_phase = state_file_name_phase
        


    def _step(self, i=0):
        '''
        Training update for one epoch of neural quantum state tomography. Updates the
        neural quantum state parameters based on the shadow-tomography based infidelity estimate.

        ''' 
        with torch.no_grad():
            # first we collect samples from neural network state
            NN_samples = self.nqs_amp.sample(self.num_samples)
            unique_samples, counts = torch.unique(NN_samples, dim=0, return_counts=True)
            weights = counts.type(torch.double)/self.num_samples
            # create the list for calculating the fidelity later
            prediction_list = []

        
        loss_nsqst = 0
        for phi in self.phi_list:           
            with torch.no_grad():
                Clifford_sample_amplitudes, weights, unique_samples = \
                    helper.samples_to_clifford_amplitudes(phi, unique_samples, weights, self.N) 
                                                
                # only the phase term
                NN_sample_amplitudes_no_grad = self.nqs_phase.amplitudes(unique_samples, return_polar = False)

                overlap_ = utils.Complex(Clifford_sample_amplitudes) * NN_sample_amplitudes_no_grad.conjugate()

                # get the conjugate version of amp_fraction
                mean = (utils.Complex(np.sqrt(weights)) * overlap_).sum(dim=0)
                overlap = utils.Complex.abs(mean)**2

                # append to the list for fidelity estimate
                overlap_scalar = overlap.detach().numpy()
                prediction_i = (2**self.N+1) * overlap_scalar - 1 
                prediction_list.append(prediction_i)
                    
            # calculate the gradient 
            
            # get real and imgaginary parts of term 1
            NN_phase_exp = self.nqs_phase.amplitudes(unique_samples, return_polar=False)
            term_1_ = utils.Complex(Clifford_sample_amplitudes).conjugate() * NN_phase_exp
            term_1 = (utils.Complex(np.sqrt(weights)) * term_1_).sum(dim=0)   
            term_1_real = term_1.real    
            term_1_imag = term_1.imag
            
            # now since loss = term_1 * term_2, get the real part of the product
            term_2_real = mean.real
            term_2_imag = mean.imag
            loss_real = term_1_real * term_2_real - term_1_imag * term_2_imag
            # propagate the gradient for every shadow, here we accumulate the gradient for every shadow
            loss_nsqst_i = - 2 * (2**self.N+1) * loss_real / self.batch_size
            loss_nsqst += loss_nsqst_i
        
        # end of the step, we update the model parameters
        torch.nn.utils.clip_grad_norm_(self.nqs_phase.parameters(), 5)
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
            # we will have to calculate infidelity using dense vectors
            infid = 0

            amp_1_ = self.nqs_amp.amplitudes(torch.tensor([1] * self.N) ).real.detach().numpy() 
            amp_0_ = self.nqs_amp.amplitudes(torch.tensor([0] * self.N) ).real.detach().numpy() 
            
            amp_1 = amp_1_ * self.nqs_phase.amplitudes(torch.tensor([1] * self.N) ).detach().numpy() 
            amp_0 = amp_0_ * self.nqs_phase.amplitudes(torch.tensor([0] * self.N) ).detach().numpy() 
            
            infid = 1 - np.abs(amp_1 * (1/np.sqrt(2)) + amp_0 * (1/np.sqrt(2)) * 1j  )**2
            
            infidelity_list.append(infid)
            cost_fun_list.append(cost_fun)
            print('Exact infidelity is ' + str(infid) ) 

        if self.state_file_name_amp != None:
            torch.save(self.nqs_amp.state_dict(), self.state_file_name_amp)
            torch.save(self.nqs_phase.state_dict(), self.state_file_name_phase)
        return infidelity_list, cost_fun_list