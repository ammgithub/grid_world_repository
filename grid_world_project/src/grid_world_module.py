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

class GridWorld_TwoArrays(object):
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
        grid_world = GridWorld_TwoArrays(policy, num_rows, num_cols, reward)
        grid_world = GridWorld_TwoArrays(0.25, 0.25, 0.25, 0.25), 4, 4, -1, True)
        """
        # User control
        self.name = "GridWorld_TwoArrays"
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
    
if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\grid_world_repository\\grid_world_project\\src\\grid_world_module.py')
    """
    policy = (0.25, 0.25, 0.25, 0.25)
#     policy = (0.7, 0.1, 0.1, 0.1)
#     policy = (1.0, 0.0, 0.0, 0.0)
    num_rows = 4
    num_cols = 4
    reward = -1
    grid_world = GridWorld_TwoArrays(policy, num_rows, num_cols, reward)
    print "\nRunning Grid World with two arrays..."
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

    
    
