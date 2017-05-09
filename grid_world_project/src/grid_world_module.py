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
    def __init__(self, policy, num_rows, num_cols, reward):
        """
        Grid world parameters. 
        
        Parameters
        ----------
        In    : policy (scalar <= 1) , num_rows, num_cols, reward)
        Out   : 
        
        Examples
        --------
        grid_world = GridWorld_TwoArrays(policy, num_rows, num_cols, reward)
        grid_world = GridWorld_TwoArrays(0.25, 4, 4, -1)
        """
        # User control
        self.policy = policy
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.reward = reward
        
        # Internal parameters
        self.MAXITER = 10
        self.gamma = 1.0
        self.MAX_DELTA = 1e-2
        
    def iterative_policy_evaluation(self):
        """
        This method does most of the work. Save old values, then compute new 
        values (per state) and calculate difference (current_delta) between 
        old and new values.  
        
        Parameters
        ----------
        In    : 
        Out   : 
        
        Examples
        --------
        grid_world.iterative_policy_evaluation()
        """
        self.value_old = np.zeros((self.num_rows, self.num_cols))
        self.value_new = np.zeros((self.num_rows, self.num_cols))
        counter = 0
        current_delta = self.MAX_DELTA + 1
        
        print "self.value_new = \n", self.value_new
        print "counter = ", counter
        print "current_delta = ", current_delta, "\n"
        
        while (current_delta > self.MAX_DELTA and 
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
                    m_up = move_up(r, c, self.num_rows, self.num_cols, policy, \
                                   reward, self.gamma, self.value_old)
                    m_down = move_down(r, c, self.num_rows, self.num_cols, \
                                              policy, reward, self.gamma, self.value_old)
                    m_right = move_right(r, c, self.num_rows, self.num_cols, \
                                               policy, reward, self.gamma, self.value_old)
                    m_left = move_left(r, c, self.num_rows, self.num_cols, \
                                              policy, reward, self.gamma, self.value_old)
                    
                    self.value_new[r, c] = m_up + m_down + m_right + m_left
                    current_delta = max(current_delta, abs(v - self.value_new[r, c]))
            
            self.value_old = deepcopy(self.value_new)
            counter += 1
            print "self.value_new = \n", self.value_new
            print "counter = ", counter
            print "current_delta = ", current_delta, "\n"
        
        self.value = self.value_new

def move_up(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from going up.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : m_up
    
    Examples
    --------
    m_up = move_up(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if r==0:
        # Top row 
        m_up = policy * (reward + gamma*value_old[r,c])
    elif (r==1 and c==0):
        # Just below the terminal state
        m_up = policy * (reward + gamma*value_old[r-1,c])
    else:
        # Default contribution up
        m_up = policy * (reward + gamma*value_old[r-1,c])
    return m_up

def move_down(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from going down.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : m_down
    
    Examples
    --------
    m_down = move_down(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if r==num_rows-1:
        # Bottom row 
        m_down = policy * (reward + gamma*value_old[r,c])
    elif (r==num_rows-2 and c==num_cols-1):
        # Just above the terminal state
        m_down = policy * (reward + gamma*value_old[r+1,c])
    else:
        # Default contribution down
        m_down = policy * (reward + gamma*value_old[r+1,c])
    return m_down
    
def move_right(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from stepping right.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : m_right
    
    Examples
    --------
    m_right = move_right(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if c==num_cols-1:
        # Right most column
        m_right = policy * (reward + gamma*value_old[r,c])
    elif (r==num_rows-1 and c==num_cols-2):
        # Just to the left of the terminal state
        m_right = policy * (reward + gamma*value_old[r,c+1])
    else:
        # Default contribution right
        m_right = policy * (reward + gamma*value_old[r,c+1])
    return m_right

def move_left(r, c, num_rows, num_cols, policy, reward, gamma, value_old):
    """
    Compute the temporary value resulting from stepping left.  
    
    Parameters
    ----------
    In    : r, c, num_rows, num_cols, policy, reward, gamma, value_old
    Out   : m_left
    
    Examples
    --------
    m_left = move_left(r, c, num_rows, num_cols, policy, reward, gamma, value_old)
    """
    if c==0:
        # Left most column
        m_left = policy * (reward + gamma*value_old[r,c])
    elif (r==0 and c==1):
        # Just to the right of the terminal state
        m_left = policy * (reward + gamma*value_old[r,c-1])
    else:
        # Default contribution left
        m_left = policy * (reward + gamma*value_old[r,c-1])
    return m_left
    
if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\grid_world_repository\\grid_world_project\\src\\grid_world_module.py')
    """
    policy = 0.25
    num_rows = 4
    num_cols = 4
    reward = -1
    grid_world = GridWorld_TwoArrays(policy, num_rows, num_cols, reward)
    print "grid_world.policy = ", grid_world.policy
    print "grid_world.num_rows = ", grid_world.num_rows
    print "grid_world.num_cols = ", grid_world.num_cols
    print "grid_world.reward = ", grid_world.reward
    grid_world.iterative_policy_evaluation()
    print "grid_world.value = \n", grid_world.value

    
    
