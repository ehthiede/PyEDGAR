# -*- coding: utf-8 -*-
"""Class definitions for the dynamical dataset object used to store and work
with dynamical data.

@author: Erik

"""
from __future__ import absolute_import, division, print_function, unicode_literals


class DynamicalDataset(object):
    """This object aims to provide a handy framework for dealing with dynamical
    data."""

    def __init__(self, data, lag=1):
        """Initiates the dynamical dataset.

        Parameters
        ----------
        traj_data : list of arrays OR tuple of two arrays OR single numpy array
            Dynamical data on which to perform the delay embedding. This dataset can be of multiple types. If a list of arrays is provided, the data is interpreted as a list of trajectories, where each array is a single trajectory. If a tuple is provided, the first element in the tuple is an array containing all of the trajectory information stacked vertically. The first N_1 elements are the datapoints for the first trajectory, the next N_2 the datapoints for the second trajectory, and so on. The second element in the tuple is the edges of the trajectory: an array of [0,N_1,N_2,...]. Finally, if only a single numpy array is provided, the data is taken to come from a single numpy trajectory.
        lag : int
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.

        """
        pass

    def compute_generator(self, lag=None):
        """Computes a Galerkin approximation of the generator.

        Returns
        -------
        L : 2d numpy array
            Matrix giving the Galerkin approximation of the generator.

        """
        return

    def compute_transop(self, lag=None):
        """Computes a Galerkin approximation of the transfer operator.

        Returns
        -------
        L : 2d numpy array
            Matrix giving the Galerkin approximation of the generator.

        """
        return

    def initial_inner_product(self, dynamical_data):
        """Calculates the inner product of a function against the given
        dynamical dataset.

        Parameters
        ----------
        dynamical_data : dynamical dataset object
            Other dynamical dataset object with which to perform the dot product.

        Returns
        -------
        product : numpy array
            Output of the estimate of the product of the two datasets against the initial density.

        """
        return

    def get_tlist(self):
        """Returns the trajectory data in the trajectory list format.

        Returns
        -------
        tlist : list of numpy arrays
            List, where each element is a trajectory of size N_n by d, where N_n is the length of the trajectory and d is the dimensionality of the system.

        """
        return

    def get_flat_data(self):
        """Returns the trajectory data in the flat format.

        Returns
        -------
        traj2D : 2D numpy array
            Numpy array containing the flattened trajectory information.
        traj_edges : 1D numpy array
            Numpy array where each element is the start of each trajectory: the n'th trajectory runs from traj_edges[n] to traj_edges[n+1]

        """
        return
