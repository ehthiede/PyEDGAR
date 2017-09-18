# -*- coding: utf-8 -*-
"""
Class definitions for the dynamical dataset object used to store and work with dynamical data.
@author: Erik
"""
from __future__ import absolute_import, division, print_function, unicode_literals

class dynamical_dataset(object):
    """
    This object aims to provide a handy framework for dealing with dynamical data.
    """
    def __init__(self,data):
        """ Initiates the dynamical dataset.

        Parameters
        ----------
        traj_data : list of arrays OR tuple of two arrays OR single numpy array 
            Dynamical data on which to perform the delay embedding. This dataset can be of multiple types. If a list of arrays is provided, the data is interpreted as a list of trajectories, where each array is a single trajectory. If a tuple is provided, the first element in the tuple is an array containing all of the trajectory information stacked vertically. The first N_1 elements are the datapoints for the first trajectory, the next N_2 the datapoints for the second trajectory, and so on. The second element in the tuple is the edges of the trajectory: an array of [0,N_1,N_2,...]. Finally, if only a single numpy array is provided, the data is taken to come from a single numpy trajectory. 

        """
        pass

    def compute_generator(lag=1):
        """ Computes a Galerkin approximation of the generator.

        Parameters
        ----------
        lag : int
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.

        Returns
        -------
        L : 2d numpy array
            Matrix giving the Galerkin approximation of the generator.

        """
