# -*- coding: utf-8 -*-
"""Routines for constructing estimates of dynamical quantities on trajectory
data using Galerkin expansion.

@author: Erik

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.linalg as spl

from .dataset import DynamicalDataset


def compute_mfpt(basis, state_A, lag=None, timestep=1.):
    """Calculates the mean first passage time into state A as a function of
    each point.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.
        This method works much better if the basis set is zero on state A, however this is not a necessity.
    state_A : dynamical dataset object
        Dynamical dataset object whose values are 1 or 0, corresponding to whether or not the datapoint is in state A.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.  If not provided, uses value in the generator.
    timestep : scalar, optional
        Time between timepoints in the trajectory data.


    Returns
    -------
    mfpt : dynamical basis object
        Dynamical dataset object containing the values of the mean first passage time at each point.

    """
    # REWRITE ONCE YOU EXPAND INITIAL IP FUNCTION
    L = basis.compute_generator(lag=lag)
    basis_flat_traj, basis_traj_edges = basis.get_flat_data()
    state_A_flat_traj = state_A.get_flat_data()[0]
    complement_A = (state_A_flat_traj.astype('int')-1)
    comp_A_dset = DynamicalDataset((complement_A, basis_traj_edges))
    beta = basis.initial_inner_product(comp_A_dset)
    coeffs = spl.solve(L, beta)
    new_vals = np.dot(basis_flat_traj, coeffs)
    return DynamicalDataset((new_vals, basis_traj_edges), lag=basis.lag, timestep=basis.timestep)


def compute_committor(basis, stateA, stateB, test_fxn=None, lag=1):
    """Calculates the mean first passage time into state A as a function of
    each point.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.
        This method works much better if the basis set is zero on states A and B, however this is not a necessity.
    state_A : dynamical dataset object
        Dynamical dataset object whose values are 1 or 0, corresponding to whether or not the datapoint is in state A.
    state_B : dynamical dataset object
        Dynamical dataset object whose values are 1 or 0, corresponding to whether or not the datapoint is in state B.
    test_fxn : dynamical dataset object, optional
        The value of the test function obeying the inhomogenous boundary conditions.  If not given, taken to just be state B.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.

    Returns
    -------
    committor: dynamical basis object
        Dynamical dataset object containing the values of the committor at each point.

    """
    if lag is None:
        lag = basis.lag
    if test_fxn is None:
        test_fxn = stateB
    L = basis.compute_generator(lag=lag)
    # Calculate approximate action of generator on test fxn
    initial_points = basis.get_initial_final_split(lag)[0]
    test_fxn_data = test_fxn.get_flat_data()[0].flatten()
    test_fxn_diff_part = (test_fxn_data[lag:]-test_fxn_data[:-lag])/basis.timestep
    test_fxn_diff = np.zeros(test_fxn_data.shape)
    test_fxn_diff[:-lag] = test_fxn_diff_part
    test_fxn_diff = test_fxn_diff[initial_points]
    basis_flat_traj, basis_traj_edges = basis.get_flat_data()
    basis_initial_points = basis_flat_traj[initial_points]
    L_test = np.dot(basis_initial_points.T, test_fxn_diff)/len(initial_points)
    # Solve for comittor
    coeffs = spl.solve(L, -L_test)
    new_vals = np.dot(basis_flat_traj, coeffs)
    return DynamicalDataset((new_vals, basis_traj_edges), lag=basis.lag, timestep=basis.timestep)


def compute_change_of_measure(basis, lag=1, fix=1):
    """Calculates the value of the change of measure to the stationary distribution for each datapoint.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.
        This method works much better if the basis set is zero on states A and B, however this is not a necessity.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.
    fix : int, optional
        Basis set whose coefficient to hold at a fixed value.

    Returns
    -------
    change_of_measure : dynamical basis object
        Dynamical dataset object containing the values of the change of measure to the stationary distribution at each point.

    """
    if lag is None:
        lag = basis.lag
    L = basis.compute_generator(lag=lag)
    not_fixed = list(range(0, fix))+list(range(fix+1, len(L)))
    b = -L[fix, not_fixed]
    L_submat_transpose = (L[not_fixed, :][:, not_fixed]).T
    pi_notfixed = spl.solve(L_submat_transpose, b)  # coeffs of not fixed states
    pi = np.ones(len(L))
    pi[not_fixed] = pi_notfixed  # All coefficients
    basis_flat_traj, basis_traj_edges = basis.get_flat_data()
    pi_realspace = np.dot(basis_flat_traj, pi)  # Convert back to realspace.
    # As positivity is not guaranteed, try to ensure most values are positive
    pi_realspace *= np.sign(np.median(pi_realspace))
    return DynamicalDataset((pi_realspace, basis_traj_edges), lag=basis.lag, timestep=basis.timestep)


def compute_esystem(basis, lag=1, left=False, right=True):
    """Calculates the eigenvectors and eigenvalues of the generator through
    Galerkin expansion.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.
        This method works much better if the basis set is zero on states A and B, however this is not a necessity.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.
    left : bool, optional
        Whether or not to calculate the left eigenvectors  of the system.
    right : bool, optional
        Whether or not to calculate the right eigenvectors  of the system.

    Returns
    -------
    eigenvalues : numpy array
        Numpy array containing the eigenvalues of the generator.
    left_eigenvectors : dynamical dataset object, optional
        If left was set to true, the left eigenvectors are returned as a dynamical dataset object.
    right_eigenvectors : dynamical dataset object, optional
        If right was set to true, the right eigenvectors are returned as a dynamical dataset object.

    """
    if lag is None:
        lag = basis.lag
    basis_flat_traj, basis_traj_edges = basis.get_flat_data()
    L = basis.compute_generator(lag=lag)
    S = basis.initial_inner_product(basis,lag=lag)
    # Calculate, sort eigensystem
    if (left and right):
        evals, evecs_l, evecs_r = spl.eig(L,b=S,left=True,right=True)
        evals, [evecs_l, evecs_r] = _sort_esystem(evals, [evecs_l, evecs_r])
        expanded_evecs_l = np.dot(basis_flat_traj,evecs_l)
        expanded_evecs_r = np.dot(basis_flat_traj,evecs_r)
        return evals, evecs_l, evecs_r
    elif (left or right):
        evals, evecs = spl.eig(L,b=S,left=left,right=right)
        evals, [evecs] = _sort_esystem(evals, [evecs])
        expanded_evecs = np.dot(basis_flat_traj,evecs)
        return evals, evecs
    else:
        evals = spl.eig(L,b=S,left=False,right=False)
        return np.sort(evals)[::-1]

def _sort_esystem(evals, evec_collection):
    """Utility function that sorts a collection of eigenvetors in desceding
    order, and then sorts the eigenvectors accordingly.

    Parameters
    ----------
    evals : numpy array
        The unsorted eigenvalues.
    evec_collection : list of arrays
        List where each element in the list is a collection of eigenvectors.
        Each collection is sorted according to the eigenvalues.

    Returns
    -------
    sorted_evals : numpy array
        The sorted eigenvalues
    sorted_evecs : of arrays
        list where each element is a collection of sorted eigenvectors.

    """
    idx = evals.argsort()[::-1]
    sorted_evals = evals[idx]
    sorted_evecs = [evecs[:,idx] for evecs in evec_collection]
    return sorted_evals, sorted_evecs

