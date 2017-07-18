"""
Created on May 05, 2017

Grid world using two arrays

Reinforcement Learning, Chapter 4
(Sutton, Barto, 1998)

"""
__author__ = 'amm'
__date__  = "May 05, 2017"
__version__ = 0.0

import numpy as np
import pylab as plt
from copy import deepcopy

np.set_printoptions(linewidth = 100, edgeitems = 'all', suppress = True, 
                 precision = 4)

class GridWorld_Barto(object):
    """
    Grid world using two arrays
     
    """
    def __init__(self, policy, num_rows, num_cols, reward, verbatim=False):
        """
        Grid world parameters. 
         
        Parameters
        ----------
        In    : policy (tuple, sum=1) , num_rows, num_cols, reward, verbatim
         
        Examples
        --------
        grid_world = GridWorld_Barto(policy, num_rows, num_cols, reward)
        grid_world = GridWorld_Barto(0.25, 0.25, 0.25, 0.25), 4, 4, -1, True)
        """
        # User control
        self.name = "GridWorld_Barto"
        # tuple with policy probabilities (up, down, right, left), sum(policy) = 1
        self.policy = policy
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.reward = reward
        self.verbatim = verbatim
         
        # Internal parameters
        self.gamma = 1.0
        self.MAXITER = 10
        self.MAXDELTA = 1e-6
        self.counter = 0
        self.current_delta = self.MAXDELTA + 1
         
    def iterative_policy_evaluation(self):
        """
        This method does most of the work. For all states compute the value 
        function contribution for the possible moves up, down, right, left.  
         
        There are two possible stopping criteria.  Either number of iterations
        (self.MAXITER) or the maximum change in the value function for a 
        single state (self.MAXDELTA).  
         
        Parameters
        ----------
        Out   : self.value, self.counter, self.current_delta
         
        Examples
        --------
        grid_world.iterative_policy_evaluation()
        """
        self.value_old = np.zeros((self.num_rows, self.num_cols))
        self.value_new = np.zeros((self.num_rows, self.num_cols))
        counter = self.counter
        current_delta = self.current_delta
         
        if self.verbatim: 
            print "self.value_new = \n", self.value_new
            print "counter = ", counter
            print "current_delta = ", current_delta, "\n"
         
        while (current_delta > self.MAXDELTA and 
               counter < grid_world.MAXITER):
            current_delta = 0.0
            # visit all states
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if (r==0 and c==0) or (r==num_rows-1 and c==num_cols-1):
                        # disregard the two terminal states
                        continue
                     
                    # v to keep track of progress
                    v = self.value_old[r, c]
                    # Compute value contribution for each action
                    mup = move_up(r, c, self.num_rows, self.num_cols, 
                                  self.policy[0], reward, self.gamma, self.value_old)
                    mdown = move_down(r, c, self.num_rows, self.num_cols, 
                                      self.policy[1], reward, self.gamma, self.value_old)
                    mright = move_right(r, c, self.num_rows, self.num_cols, 
                                        self.policy[2], reward, self.gamma, self.value_old)
                    mleft = move_left(r, c, self.num_rows, self.num_cols, 
                                      self.policy[3], reward, self.gamma, self.value_old)
                     
                    self.value_new[r, c] = mup + mdown + mright + mleft
                    current_delta = max(current_delta, abs(v - self.value_new[r, c]))
             
            self.value_old = deepcopy(self.value_new)
            counter += 1
             
            if self.verbatim: 
                print "self.value_new = \n", self.value_new
                print "counter = ", counter
                print "current_delta = ", current_delta, "\n"
         
        self.value = self.value_new
        self.counter = counter
        self.current_delta = current_delta

def move_up(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from going up.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : mup
    
    Examples
    --------
    mup = move_up(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if r==0:
        # Top row 
        mup = policy * (reward + gamma*value_old[r,c])
    elif (r==1 and c==0):
        # Just below the terminal state
        mup = policy * (reward + gamma*value_old[r-1,c])
    else:
        # Default contribution up
        mup = policy * (reward + gamma*value_old[r-1,c])
    return mup

def move_down(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from going down.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : mdown
    
    Examples
    --------
    mdown = move_down(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if r==num_rows-1:
        # Bottom row 
        mdown = policy * (reward + gamma*value_old[r,c])
    elif (r==num_rows-2 and c==num_cols-1):
        # Just above the terminal state
        mdown = policy * (reward + gamma*value_old[r+1,c])
    else:
        # Default contribution down
        mdown = policy * (reward + gamma*value_old[r+1,c])
    return mdown
    
def move_right(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from stepping right.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : mright
    
    Examples
    --------
    mright = move_right(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if c==num_cols-1:
        # Right most column
        mright = policy * (reward + gamma*value_old[r,c])
    elif (r==num_rows-1 and c==num_cols-2):
        # Just to the left of the terminal state
        mright = policy * (reward + gamma*value_old[r,c+1])
    else:
        # Default contribution right
        mright = policy * (reward + gamma*value_old[r,c+1])
    return mright

def move_left(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from stepping left.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : mleft
    
    Examples
    --------
    mleft = move_left(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if c==0:
        # Left most column
        mleft = policy * (reward + gamma*value_old[r,c])
    elif (r==0 and c==1):
        # Just to the right of the terminal state
        mleft = policy * (reward + gamma*value_old[r,c-1])
    else:
        # Default contribution left
        mleft = policy * (reward + gamma*value_old[r,c-1])
    return mleft
    
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
    In    : num_states, reward (per state), gamma, num_iter, P (transition probability matrix)
    Out   : gw.values, gw.policies (list of values and policies per state)

    Examples
    --------
    gw = GridWorld(num_states, rew_idx, gamma, num_iter, P, verbose=False)
    gw = GridWorld(11, -0.01, 1.0, 50, P)    
    """
    def __init__(self, num_states, reward, gamma, num_iter, P, verbose=False):
        self.num_states = num_states
        # transition probabilities (3D matrix, states x states x actions)
        self.P = P
        self.num_actions = self.P.shape[2]
        # per stages reward
        self.reward = reward * np.ones(num_states)
        self.gamma = gamma
        self.num_iter = num_iter
        self.values = np.zeros((num_states, num_iter))
        self.policies = np.zeros((num_states, num_iter))
        self.Q = np.zeros((self.num_actions, num_states, num_iter))
        self.Qpol = np.zeros((self.num_actions, num_states, 1))
        # Exit from cell 4
        self.values[3, 0] = 1.0
        # Exit from cell 7
        self.values[6, 0] = -1.0
        self.verbose = verbose
        if self.values.shape[0] != self.P.shape[0]: 
            raise ValueError("Please check values vector and transition probability matrix.")
        if self.policies.shape[0] != self.P.shape[0]: 
            raise ValueError("Please check policies vector and transition probability matrix.")
        
    def value_iteration(self):
        """Value iteration
        
        Vk+1 (s) = max ( sum Pss'(a)*( Rss'(a) + gamma * Vk (s') ) )
                    a     s'
        
        Values are written into 2d array currently to observe progress. Can 
        be made into 1d in the future to save memory. 
        """
        for i in range(1, self.num_iter):
            # moving up (1)
            qup = np.inner(self.P[:, :, 0], \
                           self.reward + self.gamma*self.values[:, i-1])
            qup[3] = 1.0
            qup[6] = -1.0
            # moving down (2)
            qdown = np.inner(self.P[:, :, 1], \
                             self.reward + self.gamma*self.values[:, i-1])
            qdown[3] = 1.0
            qdown[6] = -1.0
            # moving left (3)
            qleft = np.inner(self.P[:, :, 2], \
                             self.reward + self.gamma*self.values[:, i-1])
            qleft[3] = 1.0
            qleft[6] = -1.0
            # moving right (4)
            qright = np.inner(self.P[:, :, 3], \
                              self.reward + self.gamma*self.values[:, i-1])
            qright[3] = 1.0
            qright[6] = -1.0
            
            # Q values 
            Q = np.vstack([qup, qdown, qleft, qright])
            self.Q[:, :, i] = Q
            self.values[:, i] = np.max(Q, axis=0)
            self.policies[:, i] = np.argmax(Q, axis=0)+1.
            if self.verbose: 
                print "Moving (1) up / (2) down / (3) left / (4) right"
                print Q

    def policy_iteration(self, iter_pol, pol_pi_init):
        """Policy iteration: choose initial policy a = pi_i(s)
        
        Increase k until convergence:
        Vpik+1(s) =  sum Pss'(pi_i(s))*( Rss'(pi_i(s)) + gamma*Vpik (s') ) 
                         s'
        
        Policy improvement:
        pi_i+1(s) =  argmax sum Pss'(pi(s))*( Rss'(pi(s)) + gamma*Vpi (s') ) 
                             s'
        Parameters
        ----------
        In    : iter_pol, number of iterations for policy convergence
                pol_pi_init, array mapping states to actions
        
        Note  : pol_pi_init values      1.,2.,3.,4. (up, down, left, right)
                map to indices     0.,1.,2.,3.
                
        Example policy: 
        pol_pi_init = np.array([1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3])
        
        pol_probs slices through the 3d matrix self.P. For the example policy
        the first five rows of pol_probs are given by: 
        gw.P[0,:,0]
        gw.P[1,:,1]
        gw.P[2,:,2]
        gw.P[3,:,3]
        gw.P[4,:,2]
        ...
        """
        pol_pi = pol_pi_init
        self.policies_new = []
        self.policies_new.append(pol_pi)
#         for i in range(1, self.num_iter):
        for i in range(1, 5):
            self.vpi_new = np.zeros((num_states, 1))
            self.vpi_new[3, 0] = 1.0
            self.vpi_new[6, 0] = -1.0
            # Select the transition probabilities based on policy
            pol_probs = self.P[range(self.num_states), :, pol_pi-1]
            # Record all Vpi as iter_pol_count increases (TODO: only last Vpi needed)
            eps_stop = 1e-3; eps = 1.0; iter_pol_count = 0
            while (eps > eps_stop) and (iter_pol_count < 100):
                iter_pol_count += 1
                this_vpi = np.inner(pol_probs, self.reward + self.gamma*self.vpi_new[:, iter_pol_count-1])
                # keep track of vpi, can be removed to save memory
                self.vpi_new = np.hstack([self.vpi_new, this_vpi.reshape((num_states, 1))])
                self.vpi_new[3, iter_pol_count] = 1.0
                self.vpi_new[6, iter_pol_count] = -1.0
                delta_vpi = self.vpi_new[:, -1] - self.vpi_new[:, -2]
                eps = np.sqrt(np.inner(delta_vpi, delta_vpi))
            print "self.vpi_new = \n", self.vpi_new[:, -10:]
            print "iter_pol_count = ", iter_pol_count
            
            # Update pi_i+1 with converged Vpi: up (1) down (2) left (3) right (4)
            qup = np.inner(self.P[:, :, 0], \
                           self.reward + self.gamma*self.vpi_new[:, -1])
            qdown = np.inner(self.P[:, :, 1], \
                             self.reward + self.gamma*self.vpi_new[:, -1])
            qleft = np.inner(self.P[:, :, 2], \
                             self.reward + self.gamma*self.vpi_new[:, -1])
            qright = np.inner(self.P[:, :, 3], \
                              self.reward + self.gamma*self.vpi_new[:, -1])
        
            Qpol = np.vstack([qup, qdown, qleft, qright])
            Qpol[:, 3] = 1.0; Qpol[:, 6] = -1.0
            print "Qpol = \n", Qpol
            self.Qpol[:, :, 0] = Qpol
            vpi_new = np.max(Qpol, axis=0); print "vpi_new = ", vpi_new
            pol_pi = np.argmax(Qpol, axis=0)+1
            self.policies_new.append(pol_pi)
            print "np.array(self.policies_new) = \n", np.array(self.policies_new)
        
        
#         # for loop
#         self.vpi = np.zeros((num_states, iter_pol))
#         self.policies = np.zeros((num_states, iter_pol))
#         self.vpi[3, 0] = 1.0
#         self.vpi[6, 0] = -1.0
#         for i in range(1, iter_pol):
#             pol_val = np.inner(pol_probs, self.reward + self.gamma*self.vpi[:, i-1])
#             # keep track of vpi, can be removed to save memory
#             self.vpi[:, i] = pol_val
#             self.vpi[3, i] = 1.0
#             self.vpi[6, i] = -1.0
#         vpi = self.vpi[:, -10:]
#         print "\nself.vpi = \n", self.vpi[:, -10:]
#         print "iter_pol = ", iter_pol, "\n"

if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\grid_world_repository\\grid_world_project\\src\\grid_world_module.py')
    """
    
    print "\n"
    print 60 * '-'
    print 18 * ' ' + " Grid World Exercises "
    print 60 * '-'
    print "(1) Grid World by Sutton/ Barto."
    print "(2) Grid World 4 x 3 from UC CS 188."
    print 60 * '-'

#     invalid_input = True
#     while invalid_input:
#         try:
#             user_in = int(raw_input("Make selection (1)-(2): "))
#             invalid_input = False
#         except ValueError as e:
#             print "%s is not a valid selection. Please try again. "\
#             %e.args[0].split(':')[1]
    user_in = 2

    if user_in == 1:
        print "(1) Grid World by Sutton/ Barto..."
        policy = (0.25, 0.25, 0.25, 0.25)
    #     policy = (0.7, 0.1, 0.1, 0.1)
    #     policy = (1.0, 0.0, 0.0, 0.0)
        num_rows = 4
        num_cols = 4
        reward = -1
        grid_world = GridWorld_Barto(policy, num_rows, num_cols, reward)
        print "grid_world.num_rows = ", grid_world.num_rows
        print "grid_world.num_cols = ", grid_world.num_cols
        print "grid_world.reward = ", grid_world.reward
        print "grid_world.gamma = ", grid_world.gamma
        print "grid_world.MAXDELTA = ", grid_world.MAXDELTA
        print "grid_world.MAXITER = ", grid_world.MAXITER
        grid_world.iterative_policy_evaluation()
        print "\ngrid_world.value = \n", grid_world.value
        print "grid_world.counter = ", grid_world.counter
        print "grid_world.current_delta = ", grid_world.current_delta
    elif user_in == 2:
        # UC CS 188 Experiment
        print "(2) Grid World 4 x 3 from UC CS 188...\n"
        
        # Transition probability UP
        pup = np.array([[ 0.9,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
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
        pdown = np.array([[ 0.1,  0.1,  0. ,  0. ,  0.8,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
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
        pright = np.array([[ 0.1,  0.8,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
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
        num_iter = 100
        gamma = 1.00
        verbose=False
#         reward = [-0.01, -0.03, -0.4, -2.]
        reward = [-0.01, -0.03]
        # Value iteration
        for rew_idx in reward: 
            print "\nValue iteration: reward = %0.2f, gamma = %0.2f"%(rew_idx, gamma)
            print "============================================="
            gw = GridWorld(num_states, rew_idx, gamma, num_iter, P, verbose=False)
            gw.value_iteration()
            print "Optimal values after %d iterations with reward R = %0.2f"%(num_iter, rew_idx)
            print gw.values[:, num_iter-1]
            print "Optimal policies after %d iterations with reward R = %0.2f"%(num_iter, rew_idx)
            print gw.policies[:, num_iter-1]
            print "Optimal Q values after %d iterations with reward R = %0.2f"%(num_iter, rew_idx)
            print gw.Q[:, :, num_iter-1]

        # Policy iteration
        pol_pi_init = np.array([1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3])
        
        # Checking values for policies where we know the optimal result
#         # optimal for reward = -0.01, gamma = 1.00
#         pol_pi_init = np.array([ 4,  4,  4,  1,  1,  3,  1,  1,  3,  3,  2])
        # optimal for reward = -0.03, gamma = 1.00
#         pol_pi_init = np.array([ 4,  4,  4,  1,  1,  1,  1,  1,  3,  3,  3])
#         # optimal for reward = -0.40, gamma = 1.00
#         pol_pi_init = np.array([ 4,  4,  4,  1,  1,  1,  1,  1,  4,  1,  3])
#         # optimal for reward = -2.00, gamma = 1.00
#         pol_pi_init = np.array([ 4,  4,  4,  1,  1,  4,  1,  4,  4,  4,  1])
        
        iter_pol = 140
        reward = [-0.03]
        for rew_idx in reward:
            print "\nPolicy iteration: reward = %0.2f, gamma = %0.2f"%(rew_idx, gamma)
            print "=============================================="
            gw = GridWorld(num_states, rew_idx, gamma, num_iter, P, verbose=False)
            gw.policy_iteration(iter_pol, pol_pi_init)
            
        
    else:
        print "Invalid selection. Program terminating. "
    print "Finished."


    
    
