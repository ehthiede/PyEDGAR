# -*- coding: utf-8 -*-
"""Routines for constructing estimates of dynamical quantities on trajectory
data using Galerkin expansion.

@author: Erik

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.linalg as spl

from .data_manipulation import get_initial_final_split, tlist_to_flat, flat_to_tlist


def compute_mfpt(basis, stateA, lag=1, dt=1.):
    """Calculates the mean first passage time into state A as a function of
    each point.

    Parameters
    ----------
    basis : list of trajectories
        Basis for the Galerkin expansion. Must be zero in state A.
    state_A : list of trajectories
        List of trajectories where each element is 1 or 0, corresponding to whether or not the datapoint is in state A.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.  If not provided, uses value in the generator.
    timestep : scalar, optional
        Time between timepoints in the trajectory data.

    Returns
    -------
    mfpt : list of trajectories
        List of trajectories containing the values of the mean first passage time at each timepoint.

    """
    complement = [(A_i - 1.) for A_i in stateA]
    soln = compute_FK(basis, complement, lag=lag, dt=dt)

    post_processed_soln = []
    for ht_i in soln:
        ht_i[ht_i < 0.] = 0.
        post_processed_soln.append(ht_i)
    return post_processed_soln


def compute_committor(basis, guess_committor, lag=1):
    """Calculates the forward committor into state A as a function of
    each point.

    Parameters
    ----------
    basis : list of trajectories
        Basis for the Galerkin expansion. Must be zero in state A and B
    guess_committor : list of trajectories, optional
        The value of the guess function obeying the inhomogenous boundary conditions.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.

    Returns
    -------
    committor: dynamical basis object
        List of trajectories containing the values of the forward committor at each point.

    """
    h = [np.zeros(gi.shape) for gi in guess_committor]
    soln = compute_FK(basis, h, r=guess_committor, lag=lag, dt=1.)

    post_processed_soln = []
    for qi in soln:
        qi[qi > 1.] = 1.
        qi[qi < 0.] = 0.
        post_processed_soln.append(qi)
    return post_processed_soln


def compute_bwd_committor(basis, guess_committor, stationary_com, lag=1):
    """Calculates the backward into state A as a function of
    each point.

    Parameters
    ----------
    basis : list of trajectories
        Basis for the Galerkin expansion. Must be zero in state A and B
    guess_committor : list of trajectories, optional
        The value of the guess function obeying the inhomogenous boundary conditions.
    stationary_com : list of trajectories
        Values of the change of measure to the stationary distribution.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.

    Returns
    -------
    bwd_committor: dynamical basis object
        List of trajectories containing the values of the backward_committor at each point.
    """
    h = [np.zeros(gi.shape) for gi in guess_committor]
    soln = compute_adj_FK(basis, h, com=stationary_com, r=guess_committor, lag=lag, dt=1.)

    post_processed_soln = []
    for qi in soln:
        qi[qi > 1.] = 1.
        qi[qi < 0.] = 0.
        post_processed_soln.append(qi)
    return post_processed_soln


def compute_change_of_measure(basis, lag=1):
    """Calculates the value of the change of measure to the stationary distribution for each datapoint.

    Parameters
    ----------
    basis : list of trajectories
        Basis for the Galerkin expansion. Must be zero in state A and B
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.

    Returns
    -------
    change_of_measure : dynamical basis object
        List of trajectories containing the values of the change of measure to the stationary distribution at each point.

    """
    evals, evecs = compute_esystem(basis, lag, left=True, right=False)
    com = [ev_i[:, 0] for ev_i in evecs]

    # Ensure positivity
    com_sign = np.sign(np.sum(com))
    com = [ci * com_sign for ci in com]
    return com


def compute_esystem(basis, lag=1, dt=1., left=False, right=True):
    """Calculates the eigenvectors and eigenvalues of the generator through
    Galerkin expansion.

    Parameters
    ----------
    basis : list of trajectories
        List of trajectories containing the basis for the Galerkin expansion.
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
    left_eigenvectors : list of trajectories, optional
        If left was set to true, the left eigenvectors are returned as a list of trajectories.
    right_eigenvectors : list of trajectories, optional
        If right was set to true, the right eigenvectors are returned as a list of trajectories.

    """
    if lag is None:
        lag = basis.lag
    basis_list = basis
    basis_flat_traj, traj_edges = tlist_to_flat(basis_list)
    K = compute_correlation_mat(basis_list, lag=lag)
    S = compute_stiffness_mat(basis_list, lag=lag)
    L = (K - S) / (lag * dt)

    # Calculate, sort eigensystem
    if (left and right):
        evals, evecs_l, evecs_r = spl.eig(L, b=S, left=True, right=True)
        evals, [evecs_l, evecs_r] = _sort_esystem(evals, [evecs_l, evecs_r])
        evecs_l = flat_to_tlist(np.dot(basis_flat_traj, evecs_l), traj_edges)
        evecs_r = flat_to_tlist(np.dot(basis_flat_traj, evecs_r), traj_edges)
        return evals, evecs_l, evecs_r
    elif (left or right):
        evals, evecs = spl.eig(L, b=S, left=left, right=right)
        evals, [evecs] = _sort_esystem(evals, [evecs])
        evecs = np.dot(basis_flat_traj, evecs)
        evecs = flat_to_tlist(evecs, traj_edges)
        return evals, evecs
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


def compute_FK(basis, h, r=None, lag=1, dt=1., return_coeffs=False):
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
        soln = [soln[i] + r[i].ravel() for i in range(N_traj)]

    # Return calculated values
    if return_coeffs:
        return soln, coeffs
    else:
        return soln


def compute_adj_FK(basis, h, com=None, r=None, lag=1, dt=1., return_coeffs=False):
    """
    Solves the Feynman-Kac problem L^t dagger g=h on a domain D, with
    boundary conditions g=b on the complement of D. Here L^t is the adjoint of
    the generator with respect to the provided change of measure.  To account
    for the boundary conditions, we solve the homogeneous problem
    L^t g = h - L^t r, where r is the provided guess.

    Parameters
    ----------
    traj_data : list of arrays OR single numpy array
        Value of the basis functions at every time point.  Should only be nonzero
        for points on the domain.
    h : list of 1d arrays or single 1d array
        Value of the RHS of the FK formula.  This should only be nonzero at
        points on the domain, Domain.
    com : list of 1d arrays or single 1d array, optional
        Values of the change of measure against which to take the desired
        adjoint.  If not provided, takes the adjoint against the sampled meaure
    r : list of 1d arrays or single 1d array, optional
        Value of the guess function.  Should be equal to b every point off of
        the domain.  If not provided, the boundary conditions are assumed to be
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

    L_basis = compute_generator(basis, lag=lag, dt=dt, com=com)
    h_i = compute_stiffness_mat(basis, h, lag=lag, com=com)
    if r is not None:
        r = [r_i.reshape(-1, 1) for r_i in r]
        L_guess = compute_generator(r, basis, lag=lag, dt=dt, com=com).T
        h_i -= L_guess
    coeffs = spl.solve(L_basis.T, h_i.ravel())

    # Construct solution vector
    N_traj = len(basis)
    soln = [np.dot(basis[i], coeffs) for i in range(N_traj)]
    if r is not None:
        soln = [soln[i] + r[i].ravel() for i in range(N_traj)]

    # Return calculated values
    if return_coeffs:
        return soln, coeffs
    else:
        return soln


def compute_correlation_mat(Xs, Ys=None, lag=1, com=None):
    """Computes the time-lagged correlation matrix between two sets of observables.

    Parameters
    ----------
    Xs : list of trajectories
        List of trajectories for the first set of observables.
    Ys : list of trajectories, optional
        List of trajectories for the second set of observables.  If None, set to be X.
    lag : int, optional
        Lag to use in the correlation matrix.  Default is one step.
    com : list of trajectories
        Values of the change of measure against which to compute the average

    Returns
    -------
    K : numpy array
        The time-lagged correlation matrix between X and Y.
    """
    if Ys is None:
        Ys = Xs
    # Move to flat convention for easier processing.
    X_flat, traj_edges = tlist_to_flat(Xs)
    Y_flat = tlist_to_flat(Ys)[0]
    if com is not None:
        com_flat = tlist_to_flat(com)[0]
        X_flat *= com_flat

    # Split into initial, final points
    initial_indices, final_indices = get_initial_final_split(traj_edges, lag=lag)
    Y_t_lag = Y_flat[final_indices]
    X_t_0 = X_flat[initial_indices]

    # Calculate correlation, stiffness matrices.
    N = X_t_0.shape[0]
    K = X_t_0.T.dot(Y_t_lag) / float(N)
    return K


def compute_stiffness_mat(Xs, Ys=None, lag=1, com=None):
    """Computes the stiffness matrix between two sets of observables.

    Parameters
    ----------
    Xs : list of trajectories
        List of trajectories for the first set of observables.
    Ys : list of trajectories, optional
        List of trajectories for the second set of observables.  If None, set to be X.
    lag : int, optional
        Lag to use in the correlation matrix.  Default is one step.
        This is required as the stiffness is only evaluated over the initial points.
    com : list of trajectories
        Values of the change of measure against which to compute the average

    Returns
    -------
    S : numpy array
        The time-lagged stiffness matrix between X and Y.
    """
    if Ys is None:
        Ys = Xs
    # Move to flat convention for easier processing.
    X_flat, traj_edges = tlist_to_flat(Xs)
    Y_flat = tlist_to_flat(Ys)[0]
    if com is not None:
        com_flat = tlist_to_flat(com)[0]
        X_flat *= com_flat

    # Split into initial, final points
    initial_indices, final_indices = get_initial_final_split(traj_edges, lag=lag)
    Y_t_0 = Y_flat[initial_indices]
    X_t_0 = X_flat[initial_indices]
    N = X_t_0.shape[0]
    S = X_t_0.T.dot(Y_t_0) / float(N)
    return S


def compute_generator(Xs, Ys=None, lag=1, dt=1., com=None):
    """
    Computes the matrix of inner product elements against the generator.

    Parameters
    ----------
    Xs : list of trajectories
        List of trajectories for the first set of observables.
    Ys : list of trajectories, optional
        List of trajectories for the second set of observables.  If None, set to be X.
    lag : int, optional
        Lag to use in the correlation matrix.  Default is one step.
    dt: float, optional
        time per step of dynamics. Default is one time unit.
    com : list of trajectories
        Values of the change of measure against which to compute the average.

    Returns
    -------
    L : numpy array
        The approximation to the inner product <X, L Y>.
    """
    if Ys is None:
        Ys = Xs
    K = compute_correlation_mat(Xs, Ys, lag=lag, com=com)
    S = compute_stiffness_mat(Xs, Ys, lag=lag, com=com)
    return (K - S) / (lag * dt)
