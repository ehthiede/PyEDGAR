# -*- coding: utf-8 -*-
"""Class definitions for the dynamical dataset object used to store and work
with dynamical data.

@author: Erik

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pyedgar import data_manipulation as manip


class DynamicalDataset(object):
    """This object aims to provide a handy framework for dealing with dynamical
    data."""

    def __init__(self, data, lag=1, timestep=1.):
        """Initiates the dynamical dataset.

        Parameters
        ----------
        traj_data : list of arrays OR tuple of two arrays OR single numpy array
            Dynamical data on which to perform the delay embedding. This dataset can be of multiple types. If a list of arrays is provided, the data is interpreted as a list of trajectories, where each array is a single trajectory. If a tuple is provided, the first element in the tuple is an array containing all of the trajectory information stacked vertically. The first N_1 elements are the datapoints for the first trajectory, the next N_2 the datapoints for the second trajectory, and so on. The second element in the tuple is the edges of the trajectory: an array of [0,N_1,N_2,...]. Finally, if only a single numpy array is provided, the data is taken to come from a single numpy trajectory.
        lag : int
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.

        """
        self.lag = lag
        self.timestep = timestep

        if type(data) is tuple:
            flat_traj, traj_edges = data
        elif type(data) is list:
            flat_traj, traj_edges = manip.tlist_to_flat(data)
        elif type(data) is np.ndarray:
            flat_traj = data
            traj_edges = np.array([0, len(data)])
        else:
            raise ValueError("Unable to recognize the format of the input from the type: type must either be tuple, list, or numpy array")
        self.flat_traj = flat_traj
        self.traj_edges = traj_edges

    def compute_generator(self, lag=None):
        """Computes a Galerkin approximation of the generator.

        Parameters
        ----------
        lag : int
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.

        Returns
        -------
        L : 2d numpy array
            Matrix giving the Galerkin approximation of the generator.

        """
        if lag is None:
            lag = self.lag

        return

    def compute_transop(self, lag=None):
        """Computes a Galerkin approximation of the transfer operator.

        Parameters
        ----------
        lag : int
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.

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
        return manip.flat_to_tlist(self.flat_traj, self.traj_edges)

    def get_flat_data(self):
        """Returns the trajectory data in the flat format.

        Returns
        -------
        traj2D : 2D numpy array
            Numpy array containing the flattened trajectory information.
        traj_edges : 1D numpy array
            Numpy array where each element is the start of each trajectory: the n'th trajectory runs from traj_edges[n] to traj_edges[n+1]

        """
        return self.flat_traj, self.traj_edges

    def _get_initial_final_split(self, lag=None):
        """Returns the incides of the points in the flat trajectory of the initial and final sample points.
        In this context, initial means the first N-lag points, and final means the last N-lag points.
-
        Parameters
        ----------
        lag : int
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.

        Returns
        -------
        t_0_indices : 1D numpy array
            Indices in the flattened trajectory data of all the points at the initial times.

        t_0_indices : 1D numpy array
            Indices in the flattened trajectory data of all the points at the final times.

        """
        if lag is None:
            lag = self.lag
        ntraj = len(self.traj_edges) - 1
        t_0_indices = []
        t_lag_indices = []
        for i in range(ntraj):
            t_start = self.traj_edges[i]
            t_stop = self.traj_edges[i + 1]
            if t_stop - t_start > lag:
                t_0_indices += range(t_start, t_stop - lag)
                t_lag_indices += range(t_start + lag, t_stop)
        return np.array(t_0_indices), np.array(t_lag_indices)
