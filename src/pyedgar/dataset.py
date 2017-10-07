# -*- coding: utf-8 -*-
"""Class definitions for the dynamical dataset object used to store and work
with dynamical data.

@author: Erik

TODO: Expand inner product to allow alternate forms of input.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .data_manipulation import flat_to_tlist, tlist_to_flat


class DynamicalDataset(object):
    """This object aims to provide a handy framework for dealing with dynamical
    data."""

    def __init__(self, data, lag=1, timestep=1.):
        """Initiates the dynamical dataset.

        Parameters
        ----------
        traj_data : list of arrays OR tuple of two arrays OR single numpy array
            Dynamical data on which to perform the delay embedding. This dataset can be of multiple types. If a list of arrays is provided, the data is interpreted as a list of trajectories, where each array is a single trajectory. If a tuple is provided, the first element in the tuple is an array containing all of the trajectory information stacked vertically. The first N_1 elements are the datapoints for the first trajectory, the next N_2 the datapoints for the second trajectory, and so on. The second element in the tuple is the edges of the trajectory: an array of [0,N_1,N_2,...]. Finally, if only a single numpy array is provided, the data is taken to come from a single numpy trajectory.
        lag : int, optional
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.  Default is 1.
        timestep : scalar, optional
            The time between successive datapoints in the trajectory.  Default is 1.

        """
        self.lag = lag
        self.timestep = float(timestep)

        if type(data) is tuple:
            flat_traj, traj_edges = data
            flat_traj = np.array(flat_traj)
            traj_edges = list(traj_edges)
            if len(flat_traj) != traj_edges[-1]:
                raise ValueError("Final edge of the trajectory is not equal to the length of the data!")
        elif type(data) is list:
            flat_traj, traj_edges = tlist_to_flat(data)
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
        lag : int, optional
            Number of timepoints in the future to use for the finite difference in the discrete-time generator.

        Returns
        -------
        L : 2d numpy array
            Matrix giving the Galerkin approximation of the generator.

        """
        if lag is None:
            lag = self.lag

        t_0_indices, t_lag_indices = self.get_initial_final_split(lag)
        flat_traj_t_lag = self.flat_traj[t_lag_indices]
        flat_traj_t_0 = self.flat_traj[t_0_indices]
        M = len(t_0_indices)
        du = (flat_traj_t_lag - flat_traj_t_0)/(self.timestep * lag)
        L = np.dot(np.transpose(flat_traj_t_0), du) / M
        return L

    def compute_transop(self, lag=None):
        """Computes a Galerkin approximation of the transfer operator.

        Parameters
        ----------
        lag : int, optional
            Number of timepoints in the future to look into the future for the transfer operator.

        Returns
        -------
        P : 2d numpy array
            Matrix giving the Galerkin approximation of the transfer operator.

        """
        if lag is None:
            lag = self.lag

        t_0_indices, t_lag_indices = self.get_initial_final_split(lag)
        flat_traj_t_lag = self.flat_traj[t_lag_indices]
        flat_traj_t_0 = self.flat_traj[t_0_indices]
        M = len(t_0_indices)
        P = np.dot(np.transpose(flat_traj_t_0), flat_traj_t_lag) / M
        return P

    def initial_inner_product(self, dynamical_data, lag=None):
        """Calculates the inner product of a function against the given
        dynamical dataset.

        Parameters
        ----------
        dynamical_data : dynamical dataset object
            Other dynamical dataset object with which to perform the dot product.  If None, uses the lag from this object (not the one we're taking the dot product against).
        lag : int, optional
            Number of timepoints in the future to look into the future for the transfer operator.

        Returns
        -------
        product : numpy array
            Output of the estimate of the product of the two datasets against the initial density.

        """
        if (list(dynamical_data.traj_edges) != list(self.traj_edges)):
            raise ValueError("Trajectories are not the same length in the two datasets")

        if lag is None:
            lag = self.lag

        initial_indices = self.get_initial_final_split(lag=lag)[0]
        M = len(initial_indices)
        my_initial_traj = self.flat_traj[initial_indices]
        other_initial_traj = dynamical_data.flat_traj[initial_indices]
        return np.dot(my_initial_traj.T, other_initial_traj) / M

    def get_tlist(self):
        """Returns the trajectory data in the trajectory list format.

        Returns
        -------
        tlist : list of numpy arrays
            List, where each element is a trajectory of size N_n by d, where N_n is the length of the trajectory and d is the dimensionality of the system.

        """
        return flat_to_tlist(self.flat_traj, self.traj_edges)

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

    def get_initial_final_split(self, lag=None):
        """Returns the incides of the points in the flat trajectory of the initial and final sample points.
        In this context, initial means the first N-lag points, and final means the last N-lag points.
-
        Parameters
        ----------
        lag : int, optional
            Number of timepoints in the future to look into the future for the transfer operator.  Default is the value provided when constructing the dynamical dataset object.

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
            if (t_stop - t_start) > lag:
                t_0_indices += range(t_start, t_stop - lag)
                t_lag_indices += range(t_start + lag, t_stop)
        return np.array(t_0_indices), np.array(t_lag_indices)


def delay_embed(traj_data, n_embed, lag=1, verbosity=0):
    """Performs delay embedding on the trajectory data.  Takes in trajectory
    data of format types, and returns the delay embedded data in the same type.

    Parameters
    ----------
    traj_data : dataset object OR list of arrays OR tuple of two arrays OR single numpy array
        Dynamical data on which to perform the delay embedding.  This can be of multiple types; if the type is not a dataset object, the type dictates the format of the data.  See documentation for the dynamical dataset object for the types.
    n_embed : int
        The number of delay embeddings to perform.
    lag : int, optional
        The number of timesteps to look back in time for each delay. Default is 1.
    verbosity : int, optional
        The level of status messages that are output. Default is 0 (no messages).

    Returns
    -------
    embedded_data : dataset object OR list of arrays OR tuple of two arrays OR single numpy array
        Dynamical data with delay embedding performed, of the same type as the trajectory data.

    """
    if type(traj_data) is list:
        input_type = 'list_of_trajs'
        tlist = traj_data
    elif type(traj_data) is DynamicalDataset:
        input_type = 'DynamicalDataset'
        tlist = traj_data.get_tlist()
    elif type(traj_data) is tuple:
        input_type = 'flat'
        tlist = flat_to_tlist(traj_data[0], traj_data[1])
    elif type(traj_data) is np.ndarray:
        input_type = 'single_array'
        tlist = [traj_data]
    else:
        raise ValueError("Unable to recognize the format of the input from the type: type must either be tuple, list, DynamicalDataset, or numpy array")

    embed_traj_list = []
    for i, traj_i in enumerate(tlist):
        N_i = len(traj_i)
        if N_i - (lag * n_embed) <= 0:  # Must be longer than max embedding
            continue
        embed_traj_i = []
        for n in range(n_embed+1):
            start_ndx = lag * (n_embed - n)
            stop_ndx = N_i - (lag * n)
            embed_traj_i.append(traj_i[start_ndx:stop_ndx])
        embed_traj_i = np.concatenate(embed_traj_i, axis=1)
        embed_traj_list.append(embed_traj_i)

    if input_type == 'list_of_trajs':
        return embed_traj_list
    elif input_type == 'DynamicalDataset':
        new_dataset = DynamicalDataset(embed_traj_list, traj_data.lag,
                                       traj_data.timestep)
        return new_dataset
    elif input_type == 'flat':
        return tlist_to_flat(embed_traj_list)
    elif input_type == 'single_array':
        return embed_traj_list[0]
