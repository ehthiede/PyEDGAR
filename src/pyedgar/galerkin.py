# -*- coding: utf-8 -*-
"""Routines for constructing estimates of dynamical quantities on trajectory
data using Galerkin expansion.

@author: Erik

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.linalg as spl

from .dataset import DynamicalDataset
from .data_manipulation import get_initial_final_split, tlist_to_flat


def compute_mfpt(basis, state_A, lag=None, timestep=None):
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
    if timestep is None:
        dt = basis.timestep
    else:
        dt = timestep
    if lag is None:
        lag = basis.lag
    basis_list = basis.get_tlist()
    stateA_list = state_A.get_tlist()
    comp_list = [(A_i - 1.) for A_i in stateA_list]
    soln = compute_fwd_FK(basis_list, comp_list, lag=lag, dt=dt)

    return DynamicalDataset(soln, lag=basis.lag, timestep=basis.timestep)


def compute_committor(basis, stateA, stateB, test_fxn=None, lag=1):
    """Calculates the forward committor into state A as a function of
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
    basis_list = basis.get_tlist()
    guess_list = test_fxn.get_tlist()
    h = [np.zeros(gi.shape) for gi in guess_list]
    soln = compute_fwd_FK(basis_list, h, r=guess_list, lag=lag, dt=1.)
    return DynamicalDataset(soln, lag=basis.lag, timestep=basis.timestep)


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
    # As positivity is not guaranteed, we try to ensure most values are positive
    pi_realspace *= np.sign(np.median(pi_realspace))
    return DynamicalDataset((pi_realspace, basis_traj_edges), lag=basis.lag, timestep=basis.timestep)


def compute_esystem(basis, lag=1, dt=1., left=False, right=True):
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
    basis_list = basis.get_tlist()
    K = compute_correlation_mat(basis_list, lag=lag)
    S = compute_stiffness_mat(basis_list, lag=lag)
    L = (K - S) / (lag * dt)
    # L = basis.compute_generator(lag=lag)
    # S = basis.initial_inner_product(basis, lag=lag)

    # Calculate, sort eigensystem
    if (left and right):
        evals, evecs_l, evecs_r = spl.eig(L, b=S, left=True, right=True)
        evals, [evecs_l, evecs_r] = _sort_esystem(evals, [evecs_l, evecs_r])
        expanded_evecs_l = np.dot(basis_flat_traj, evecs_l)
        expanded_evecs_r = np.dot(basis_flat_traj, evecs_r)
        return evals, expanded_evecs_l, expanded_evecs_r
    elif (left or right):
        evals, evecs = spl.eig(L, b=S, left=left, right=right)
        evals, [evecs] = _sort_esystem(evals, [evecs])
        expanded_evecs = np.dot(basis_flat_traj, evecs)
        return evals, expanded_evecs
    else:
        evals = spl.eig(L, b=S, left=False, right=False)
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
    sorted_evecs = [evecs[:, idx] for evecs in evec_collection]
    return sorted_evals, sorted_evecs


def compute_fwd_FK(basis, h, r=None, lag=1, dt=1., return_coeffs=False):
    """
    Solves the forward Feynman-Kac problem Lg=h on a domain D, with boundary
    conditions g=b on the complement of D. To account for the boundary
    conditions, we solve the homogeneous problem Lg = h - Lr, where r is the
    provided guess.

    Parameters
    ----------
    traj_data : list of arrays OR single numpy array
        Value of the basis functions at every time point.  Should only be nonzero
        for points on the domain.
    h : list of 1d arrays or single 1d array
        Value of the RHS of the FK formula.  This should only be nonzero at
        points on the domain, Domain.
    r : list of 1d arrays or single 1d array, optional
        Value of the guess function.  Should be equal to b every point off of
        the domain.  IF not provided, the boundary conditions are assumed to be
        homogeneous.
    lag : int
        Number of timepoints in the future to use for the finite difference in
        the discrete-time generator.  If not provided, defaults to 1.
    timestep : scalar, optional
        Time between timepoints in the trajectory data.  Defaults to 1.

    Returns
    -------
    g : list of arrays
        Estimated solution to the Feynman-Kac problem.
    coeffs : ndarray
        Coefficients for the solution, only returned if return_coeffs is True.
    """
    h = [h_i.reshape(-1, 1).astype('float') for h_i in h]
    L_basis = compute_generator(basis, lag=lag, dt=dt)
    h_i = compute_stiffness_mat(basis, h, lag=lag)
    if r is not None:
        r = [r_i.reshape(-1, 1) for r_i in r]
        L_guess = compute_generator(basis, r, lag=lag, dt=dt)
        h_i -= L_guess
    coeffs = spl.solve(L_basis, h_i.ravel())

    # Construct solution vector
    N_traj = len(basis)
    soln = [np.dot(basis[i], coeffs) for i in range(N_traj)]
    if r is not None:
        print([si.shape for si in soln])
        print([ri.shape for ri in r])
        print('shapes!')
        soln = [soln[i] + r[i].ravel() for i in range(N_traj)]

    # Return calculated values
    if return_coeffs:
        return soln, coeffs
    else:
        return soln


def compute_correlation_mat(Xs, Ys=None, lag=1, com=None):
    """
    """
    if Ys is None:
        Ys = Xs
    # Move to flat convention for easier processing.
    X_flat, traj_edges = tlist_to_flat(Xs)
    Y_flat = tlist_to_flat(Ys)[0]
    if com is not None:
        com_flat = tlist_to_flat(com)
        X_flat *= com_flat

    # Split into initial, final points
    initial_indices, final_indices = get_initial_final_split(traj_edges)
    Y_t_lag = Y_flat[final_indices]
    X_t_0 = X_flat[initial_indices]

    # Calculate correlation, stiffness matrices.
    N = X_t_0.shape[0]
    K = X_t_0.T.dot(Y_t_lag) / float(N)
    return K


def compute_stiffness_mat(Xs, Ys=None, lag=1, com=None):
    if Ys is None:
        Ys = Xs
    # Move to flat convention for easier processing.
    X_flat, traj_edges = tlist_to_flat(Xs)
    Y_flat = tlist_to_flat(Ys)[0]
    if com is not None:
        com_flat = tlist_to_flat(com)
        X_flat *= com_flat

    # Split into initial, final points
    initial_indices, final_indices = get_initial_final_split(traj_edges)
    Y_t_0 = Y_flat[initial_indices]
    X_t_0 = X_flat[initial_indices]
    N = X_t_0.shape[0]
    S = X_t_0.T.dot(Y_t_0) / float(N)
    return S


def compute_generator(Xs, Ys=None, lag=1, dt=1., com=None):
    if Ys is None:
        Ys = Xs
    K = compute_correlation_mat(Xs, Ys, lag=lag, com=com)
    S = compute_stiffness_mat(Xs, Ys, lag=lag, com=com)
    return (K - S) / (lag * dt)
