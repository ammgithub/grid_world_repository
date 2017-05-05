"""
Created on May 05, 2017

Grid world example 

Reinforcement Learning, Chapter 4
(Sutton, Barto, 1998)

"""
__author__ = 'amm'
__date__  = "May 05, 2017"
__version__ = 0.0

import numpy as np
import pylab as plt

np.set_printoptions(linewidth = 100, edgeitems = 'all', suppress = True, 
                 precision = 4)

class GridWorld(object):
    """
    Grid world object
    """
    def __init__(self, policy, num_rows, num_cols, current_delta):
        """
        Grid world parameters. 
        
        Parameters
        ----------
        In    : 
        Out   : 
        
        Examples
        --------
        grid_world = GridWorld(policy, num_rows, num_cols, current_delta)
        grid_world = GridWorld(0.25, 4, 4, 1000000)
        
        """
        # User control
        self.policy = policy
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.current_delta = current_delta
        
        # Internal parameters
        self.MAXITER = 10
        self.gamma = 1.0
        self.MAX_DELTA = 0.001
        
        self.value_old = np.zeros((num_rows, num_cols))
        self.value_new = np.zeros((num_rows, num_cols))
        
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
        grid_world = GridWorld(policy, num_rows, num_cols, current_delta)
        grid_world = GridWorld(0.25, 4, 4, 1000000)
        
        """
        
        

    
if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\grid_world_repository\\grid_world_project\\src\\grid_world_module.py')
    """
    policy = 0.25
    num_rows = 4
    num_cols = 4
    current_delta = 1000000
    grid_world = GridWorld(policy, num_rows, num_cols, current_delta)
    print "grid_world.policy = ", grid_world.policy
    print "grid_world.num_rows = ", grid_world.num_rows
    print "grid_world.num_cols = ", grid_world.num_cols
    print "grid_world.current_delta = ", grid_world.current_delta
