# -*- coding: utf-8 -*-
"""A collection of useful functions  for manipulating trajectory data and
dynamical basis set objects.

@author: Erik

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def tlist_to_flat(trajs):
    """Flattens a list of two dimensional trajectories into a single two
    dimensional datastructure, and returns it along with a list of tuples
    giving the locations of each trajectory.

    Parameters
    ----------
    trajs : list of array-likes
        List where each element n is a array-like object of shape N_n x d, where N_n is the number of data points in that trajectory and d is the number of coordinates for each datapoint.

    Returns
    -------
    traj2D : 2D numpy array
        Numpy array containing the flattened trajectory information.
    traj_edges : 1D numpy array
        Numpy array where each element is the start of each trajectory: the n'th trajectory runs from traj_edges[n] to traj_edges[n+1]

    """
    # Check all trajectories are same order tensors.
    traj_orders = np.array([ti.ndim for ti in trajs])
    if np.any(traj_orders != traj_orders[0]):
        raise ValueError("Input Trajectories have varying dimension")

    if traj_orders[0] == 1:
        trajs = [t_i.reshape(-1, 1) for t_i in trajs]

    # Get dimensions of traj object.
    d = len(trajs[0][0])

    # Populate the large trajectory.
    traj_2d = []
    traj_edges = [0]
    len_traj_2d = 0
    for i, traj in enumerate(trajs):
        # Check that trajectory is of right format.
        if traj.ndim != 2:
            raise ValueError('Trajectory %d is not two dimensional!' % i)
        d2 = traj.shape[1]
        if d2 != d:
            raise ValueError('Trajectories are of incompatible dimension.  The first trajectory has dimension %d and trajectory %d has dimension %d' % (d, i, d2))
        traj_2d.append(traj)
        len_traj_2d += traj.shape[0]
        traj_edges.append(len_traj_2d)

    traj_2d = np.vstack(traj_2d)
    return traj_2d, np.array(traj_edges)


def flat_to_tlist(traj_2d, traj_edges):
    """Takes a flattened trajectory with stop and start points and reformats it
    into a list of separate trajectories.

    Parameters
    ----------
    traj2D : 2D numpy array
        Numpy array containing the flattened trajectory information.
    traj_edges : 1D numpy array
        Numpy array where each element is the start of each trajectory: the n'th trajectory runs from traj_edges[n] to traj_edges[n+1]

    Returns
    -------
    trajs : list of array-likes
        List where each element n is a array-like object of shape N_n x d, where N_n is the number of data points in that trajectory and d is the number of coordinates for each datapoint.

    """
    trajs = []
    ntraj = len(traj_edges) - 1
    for i in range(ntraj):
        start = traj_edges[i]
        stop = traj_edges[i + 1]
        trajs.append(traj_2d[start:stop])
    return trajs
