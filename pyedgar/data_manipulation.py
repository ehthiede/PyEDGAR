# -*- coding: utf-8 -*-
"""
A collection of useful functions  for manipulating trajectory data and dynamical basis set objects.
@author: Erik
"""

def tlist_to_flat(trajs):
    """Flattens a list of two dimensional trajectories into a single two dimensional datastructure, and returns it along with a list of tuples giving the locations of each trajectory.

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


def flat_to_tlist(traj_2d,traj_edges):
    """Takes a flattened trajectory with stop and start points and reformats it into a list of separate trajectories.

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


