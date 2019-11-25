"""
Created on May 05, 2017

Grid world tests


"""
__author__ = 'amm'
__date__  = "May 05, 2017"
__version__ = 0.0

import numpy as np
import pylab as plt

np.set_printoptions(linewidth = 100, edgeitems = 'all', suppress = True, 
                 precision = 2)

class GridWorld(object):
    """
    4 x 3 grid world from UC CS 188. 
    
    Final rewards
    ------------------------
    |     |     |     | +1 | 
    ------------------------
    |     |  X  |     | -1 | 
    ------------------------
    |     |     |     |    | 
    ------------------------
    
    Cell numbering
    ------------------------
    |  1  |  2  |  3  |  4 | 
    ------------------------
    |  5  |  X  |  6  |  7 | 
    ------------------------
    |  8  |  9  |  10 | 11 | 
    ------------------------
    
    The transition probability matrix is 11 x 11 x 4 (stacking four 11 x 11 
    matrices, one each for actions (1-up, 2-down, 3-left, 4-right). 
    Example: P[:, :, 0] is the 11 x 11 matrix describing transitions for "up".
    
    Parameters
    ----------
    In    : num_states, reward, gamma, P (transition probability matrix)
    Out   : self.policies, self.values, self.Q (value iteration)
            self.policies, self.vpi, self.Qpol (policy iteration)

    Examples
    --------
    gw = GridWorld(num_states, reward, gamma, P, verbose=False)
    gw = GridWorld(11, -0.01, 1.0, P)    
    """
    def __init__(self, num_states, reward, gamma, P, verbose=False):
        self.verbose = verbose
        self.num_states = num_states
        self.reward = reward * np.ones(num_states)
        self.gamma = gamma
        
        # transition probabilities (3D matrix, states x states x actions)
        self.P = P
        self.num_actions = self.P.shape[2]

    def value_iteration(self, num_iter):
        """Value iteration:
        
        Vk+1 (s) = max ( sum Pss'(a)*( Rss'(a) + gamma * Vk (s') ) )
                    a     s'
        
        Parameters
        ----------
        In    : num_iter
        Out   : self.policies, self.values, self.Q 
        """
        self.num_iter = num_iter
        
        self.Q = []
        # Q for i = 0:
        self.Q.append(np.zeros((self.num_actions, self.num_states)))
        self.Q[0][:, 3] = 1.0; self.Q[0][:, 6] = -1.0

        self.values = []
        # Values for i = 0:
        self.values.append(np.zeros(self.num_states))
        self.values[0][3] = 1.0; self.values[0][6] = -1.0

        self.policies = []
        # Zero policy for i = 0:
        self.policies.append(np.zeros(self.num_states, 'int'))
        for i in range(1, self.num_iter):
            qup = np.inner(self.P[:, :, 0], self.reward + self.gamma*self.values[i-1])
            qdown = np.inner(self.P[:, :, 1], self.reward + self.gamma*self.values[i-1])
            qleft = np.inner(self.P[:, :, 2], self.reward + self.gamma*self.values[i-1])
            qright = np.inner(self.P[:, :, 3], self.reward + self.gamma*self.values[i-1])

            # Q values: up (1) down (2) left (3) right (4)
            Q = np.vstack([qup, qdown, qleft, qright])
            Q[:, 3] = 1.0; Q[:, 6] = -1.0
            self.Q.append(Q)
            self.values.append(np.max(Q, axis=0))
            self.policies.append(np.argmax(Q, axis=0)+1)
            if self.verbose: 
                print( "Moving (1) up / (2) down / (3) left / (4) right")
                print( Q)
        self.policies = np.array(self.policies).T
        self.values = np.array(self.values).T
        # 3d transpose (axis = (1, 2, 0))
        self.Q = np.array(self.Q).transpose((1, 2, 0))
        if self.policies.shape[0] != self.P.shape[0]: 
            raise ValueError("Please check policies vector and transition probability matrix.")
        if self.values.shape[0] != self.P.shape[0]: 
            raise ValueError("Please check values vector and transition probability matrix.")

    def policy_iteration(self, num_pol_improv, pol_pi_init):
        """Policy iteration: choose initial policy a = pi_i(s)
        
        Increase k until convergence:
        Vpik+1(s) =  sum Pss'(pi_i(s))*( Rss'(pi_i(s)) + gamma*Vpik (s') ) 
                         s'
        
        Policy improvement:
        pi_i+1(s) =  argmax sum Pss'(pi(s))*( Rss'(pi(s)) + gamma*Vpi (s') ) 
                             s'
        Parameters
        ----------
        In    : num_pol_improv, pol_pi_init
        Out   : self.policies, self.vpi, self.Qpol 

        Example policy: 
        pol_pi_init = np.array([1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3])
        
        Note 1: pol_pi_init values      1.,2.,3.,4. (up, down, left, right)
                map to indices          0.,1.,2.,3.
                
        Note 2: pol_probs slices through the 3d matrix self.P. For the example policy
                the first five rows of pol_probs are given by: 
                gw.P[0,:,0]
                gw.P[1,:,1]
                gw.P[2,:,2]
                gw.P[3,:,3]
                gw.P[4,:,2]
                ...
        """
        self.num_pol_improv = num_pol_improv
        
        self.Qpol = []
        # Qpol for i = 0:
        self.Qpol.append(np.zeros((self.num_actions, self.num_states)))
        self.Qpol[0][:, 3] = 1.0; self.Qpol[0][:, 6] = -1.0
        
        self.vpi = []
        # Vpi for i = 0:
        self.vpi.append(np.zeros(self.num_states))
        self.vpi[0][3] = 1.0; self.vpi[0][6] = -1.0
        
        self.policies = []
        pol_pi = pol_pi_init
        # Policy for i = 0:
        self.policies.append(pol_pi)

        self.avg_num_pol_eval = 0
        self.num_pol_improv = num_pol_improv
        for i in range(1, num_pol_improv):
            # Vpi for num_pol_eval = 0
            if self.verbose: print( "i = ", i)
            this_vpi = np.zeros(num_states)
            this_vpi[3] = 1.0; this_vpi[6] = -1.0
            # Select the transition probabilities based on policy
            pol_probs = self.P[range(self.num_states), :, pol_pi-1]
            
            eps_stop = 1e-3; eps = 1.0; num_pol_eval = 0
            while (eps > eps_stop) and (num_pol_eval < 100):
                num_pol_eval += 1
                vpi_old = this_vpi
                this_vpi = np.inner(pol_probs, self.reward + self.gamma*vpi_old)
                this_vpi[3] = 1.0
                this_vpi[6] = -1.0
                delta_vpi_new = this_vpi - vpi_old
                eps = np.sqrt(np.inner(delta_vpi_new, delta_vpi_new))
            
            if self.verbose: 
                print( "I needed %d iterations during policy evaluation."%num_pol_eval)
            self.avg_num_pol_eval = (self.avg_num_pol_eval*(i-1)+num_pol_eval)/float(i)
            
            # Update pi_i+1 with converged Vpi: up (1) down (2) left (3) right (4)
            qup = np.inner(self.P[:, :, 0], self.reward + self.gamma*this_vpi)
            qdown = np.inner(self.P[:, :, 1], self.reward + self.gamma*this_vpi)
            qleft = np.inner(self.P[:, :, 2], self.reward + self.gamma*this_vpi)
            qright = np.inner(self.P[:, :, 3], self.reward + self.gamma*this_vpi)
        
            Qpol = np.vstack([qup, qdown, qleft, qright])
            Qpol[:, 3] = 1.0; Qpol[:, 6] = -1.0
            self.Qpol.append(Qpol)
            vpi = np.max(Qpol, axis=0)
            self.vpi.append(vpi)
            if self.verbose: print( "Qpol = \n", Qpol)
            if self.verbose: print( "vpi = \n", vpi)
            # Need pol_pi to compute pol_probs in next iteration
            pol_pi = np.argmax(Qpol, axis=0)+1
            self.policies.append(pol_pi)
        self.policies = np.array(self.policies).T
        self.vpi = np.array(self.vpi).T
        # 3d transpose (axis = (1, 2, 0))
        self.Qpol = np.array(self.Qpol).transpose((1, 2, 0))
        if self.policies.shape[0] != self.P.shape[0]: 
            raise ValueError("Please check policies vector and transition probability matrix.")
        if self.vpi.shape[0] != self.P.shape[0]: 
            raise ValueError("Please check Vpi vector and transition probability matrix.")

if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\grid_world_repository\\grid_world_project\\src\\grid_world_module.py')
    """
    
    print( "\n")
    print( 60 * '-')
    print( 18 * ' ' + " Grid World Exercises ")
    print( 60 * '-')
    print( "(1) Grid World 4 x 3 from UC CS 188.")
    print( 60 * '-')

    invalid_input = True
    while invalid_input:
        try:
            user_in = int(input("Make selection (1): "))
            invalid_input = False
        except ValueError as e:
            print( "%s is not a valid selection. Please try again. "\
            %e.args[0].split(':')[1])

    if user_in == 1:
        print( "(1) Grid World 4 x 3 from UC CS 188...\n")
        
        # Transition probability UP
        pup = \
        np.array([[ 0.9,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0.1,  0.8,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0.1,  0.8,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0.8,  0. ,  0. ,  0. ,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0.8,  0. ,  0. ,  0.1,  0.1,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0.8,  0. ,  0. ,  0.1,  0.1,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0.8,  0.1,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.8,  0. ,  0. ,  0.1,  0. ,  0.1],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.8,  0. ,  0. ,  0.1,  0.1]])
        
        # Transition probability DOWN
        pdown = \
        np.array([[ 0.1,  0.1,  0. ,  0. ,  0.8,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0.1,  0.8,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0.1,  0. ,  0.1,  0. ,  0.8,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0.2,  0. ,  0. ,  0.8,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0.1,  0. ,  0. ,  0.8,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.9,  0.1,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0.8,  0.1,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0.8,  0.1],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0.9]])

        # Transition probability LEFT
        pleft = np.array([[ 0.9,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0.8,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0.8,  0.1,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0.1,  0. ,  0. ,  0. ,  0.8,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0.1,  0. ,  0. ,  0.8,  0. ,  0. ,  0. ,  0.1,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0.9,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.8,  0.2,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0.8,  0.1,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0.8,  0.1]])

        # Transition probability RIGHT
        pright = \
        np.array([[ 0.1,  0.8,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0.2,  0.8,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0.1,  0.8,  0. ,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0.1,  0. ,  0. ,  0. ,  0.8,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0.1,  0. ,  0. ,  0. ,  0.8,  0. ,  0. ,  0.1,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0.1,  0.8,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.2,  0.8,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ,  0.1,  0.8],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ,  0.9]])
        
        # Overall transition probabilities P_ss' (a)
        P = np.stack([pup, pdown, pleft, pright], axis=2)
        num_states = 11
        gamma = 1.00

        # Value iteration
        num_iter = 100
        verbose=False
        reward = [-0.01, -0.03, -0.4, -2.]
        for rew_idx in reward: 
            print( "\nValue iteration: reward = %0.2f, gamma = %0.2f"%(rew_idx, gamma))
            print( "=============================================")
            gw = GridWorld(num_states, rew_idx, gamma, P, verbose)
            gw.value_iteration(num_iter)
            print( "Optimal values after %d iterations with reward R = %0.2f"%(num_iter, rew_idx))
            print( gw.values[:, num_iter-1])
            print( "Optimal policies after %d iterations with reward R = %0.2f"%(num_iter, rew_idx))
            print( gw.policies[:, num_iter-1])
            print( "Optimal Q values after %d iterations with reward R = %0.2f"%(num_iter, rew_idx))
            print( gw.Q[:, :, num_iter-1])
            print( "Testing sum(Q): %4.2f"%gw.Q[:, :, num_iter-1].sum())

        ########################################################################
        #            
        #  Checking values for policies where we know the optimal result
        #  
        #  optimal for reward = -0.01, gamma = 1.00
        #  pol_pi_init = np.array([ 4,  4,  4,  1,  1,  3,  1,  1,  3,  3,  2])
        #  optimal for reward = -0.03, gamma = 1.00
        #  pol_pi_init = np.array([ 4,  4,  4,  1,  1,  1,  1,  1,  3,  3,  3])
        #  optimal for reward = -0.40, gamma = 1.00
        #  pol_pi_init = np.array([ 4,  4,  4,  1,  1,  1,  1,  1,  4,  1,  3])
        #  optimal for reward = -2.00, gamma = 1.00
        #  pol_pi_init = np.array([ 4,  4,  4,  1,  1,  4,  1,  4,  4,  4,  1])
        #            
        ########################################################################

        # Policy iteration
        pol_pi_init = np.array([1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3])
        num_pol_improv = 20
        verbose=False
        reward = [-0.01, -0.03, -0.4, -2.]
        for rew_idx in reward:
            print( "\nPolicy iteration: reward = %0.2f, gamma = %0.2f"%(rew_idx, gamma))
            print( "==============================================")
            gw = GridWorld(num_states, rew_idx, gamma, P, verbose)
            gw.policy_iteration(num_pol_improv, pol_pi_init)
            print( "Optimal values with %d policy improvements, %3.1f average policy evaluations, and reward R = %0.2f"\
                %(num_pol_improv, gw.avg_num_pol_eval, rew_idx))
            print( gw.vpi[:, num_pol_improv-1])
            print( "Optimal policies with %d policy improvements, %3.1f average policy evaluations, and reward R = %0.2f"\
                %(num_pol_improv, gw.avg_num_pol_eval, rew_idx))
            print( gw.policies[:, num_pol_improv-1])
            print( "Optimal Qpol with %d policy improvements, %3.1f average policy evaluations, and reward R = %0.2f"\
                %(num_pol_improv, gw.avg_num_pol_eval, rew_idx))
            print( gw.Qpol[:, :, num_pol_improv-1])
            print( "Testing sum(Qpol): %4.2f"%gw.Qpol[:, :, num_pol_improv-1].sum())

    else:
        print( "Invalid selection. Program terminating. ")
    print( "Finished.")
