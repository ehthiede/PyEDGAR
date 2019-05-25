# -*- coding: utf-8 -*-
"""Routines and Class definitions for constructing basis sets using the
diffusion maps algorithm.

@author: Erik

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from pydiffmap.diffusion_map import DiffusionMap
from .data_manipulation import _flat_to_orig, _as_flat


class DiffusionAtlas(object):
    """The diffusion atlas is a factory object for constructing diffusion map
    bases with various boundary conditions.
    """

    def __init__(self, dmap_object=None):
        """
        Builds the Diffusion Atlas a diffusion map object.

        Parameters
        ----------
        dmap_object : pyDiffMap DiffusionMap object, optional
            Diffusion map object to use in the atlas.  If None, uses default
            parameters, which are similar to the LSDmap.
        """
        if dmap_object is None:
            dmap_object = DiffusionMap.from_sklearn(alpha=0, k=500,
                                                    bandwidth_type='-1/d',
                                                    epsilon='bgh_generous')
        self.dmap = dmap_object

    @classmethod
    def from_kernel(cls, kernel_object, alpha=0.5, weight_fxn=None,
                    density_fxn=None, bandwidth_normalize=False, oos='nystroem'):
        """
        Builds the Diffusion Atlas using a pyDiffMap kernel.
        See the pyDiffMap.DiffusionMap constructor for a description of arguments.
        """
        dmap = DiffusionMap(kernel_object=kernel_object, alpha=alpha,
                            weight_fxn=weight_fxn, density_fxn=density_fxn,
                            bandwidth_normalize=bandwidth_normalize, oos=oos)
        return cls(dmap)

    @classmethod
    def from_sklearn(cls, alpha=0.5, k=64, kernel_type='gaussian', epsilon='bgh', neighbor_params=None,
                     metric='euclidean', metric_params=None, weight_fxn=None, density_fxn=None, bandwidth_type=None,
                     bandwidth_normalize=False, oos='nystroem'):
        """
        Builds the Diffusion Atlas using the standard pyDiffMap kernel.
        See the pyDiffMap.DiffusionMap.from_sklearn for a description of arguments.
        """
        dmap = DiffusionMap.from_sklearn(alpha=alpha, k=k, kernel_type=kernel_type, epsilon=epsilon, neighbor_params=neighbor_params, metric=metric, metric_params=metric_params, weight_fxn=weight_fxn, density_fxn=density_fxn, bandwidth_type=bandwidth_type, bandwidth_normalize=bandwidth_normalize, oos=oos)
        return cls(dmap)

    def fit(self, data):
        """Constructs the diffusion map on the dataset.

        Parameters
        ----------
        data : 2D array-like OR list of trajectories OR Flat data format
            Dataset on which to construct the diffusion map.
        """
        # Default Parameter Selection and Type Cleaning
        data, edges, input_type = _as_flat(data)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        self.data = data
        self.edges = edges
        self.input_type = input_type
        self.dmap.construct_Lmat(data)
        return self

    def make_dirichlet_basis(self, k, in_domain=None, return_evals=False):
        """Creates a diffusion map basis set that obeys the homogeneous
        Dirichlet boundary conditions on the domain.  This is done by taking
        the eigenfunctions of the diffusion map submatrix on the domain.

        Parameters
        ----------
        k : int
            Number of basis functions to create.
        in_domain : 1D array-like, OR list of such arrays, OR flat data format, optional
            Array of the same shape as the data, where each element is 1 or True if that datapoint is inside the domain, and 0 or False if it is in the domain.  Naturally, this must be the length as the current dataset.  If None (default), all points assumed to be in the domain.
        return_evals : Boolean, optional
            Whether or not to return the eigenvalues as well.  These are useful for out of sample extension.


        Returns
        -------
        basis : Dataset of same type as the data
            The basis functions evaluated on each datapoint.  Of the same type as the input data.
        evals : 1D numpy array, optional
            The eigenvalues corresponding to each basis vector.  Only returned if return_evals is True.

        """
        submat = self.dmap.L
        npoints = submat.shape[0]
        # Take the submatrix if necessary
        if in_domain is not None:
            in_domain = _as_flat(in_domain)[0].ravel()
            domain = np.where(in_domain > 0)[0]
            submat = submat[domain][:, domain]
        evals, evecs = spsl.eigs(submat, k, which='LR')
        # Sort by eigenvalue.
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        # If using a submatrix, expand back to full size
        if in_domain is not None:
            full_evecs = np.zeros((npoints, k))
            full_evecs[domain, :] = np.real(evecs)
        else:
            full_evecs = evecs
        full_evecs = _flat_to_orig(full_evecs, self.edges, self.input_type)
        if return_evals:
            return full_evecs, evals
        else:
            return full_evecs

    def extend_dirichlet_basis(self, Y, in_domain, basis, evals):
        """
        Performs out-of-sample extension an a dirichlet basis set.

        Parameters
        ----------
        Y : 2D array-like OR list of trajectories OR flat data format
            Data for which to perform the out-of-sample extension.
        in_domain : 1D array-like, OR list of such arrays, OR flat data format
            Dataset of the same shape as the input datapoints, where each element is 1 or True if that datapoint is inside the domain, and 0 or False if it is in the domain.
        basis : 2D array-like OR list of trajectories OR Flat data format
            The basis functions.
        evals : 1D numpy array
            The eigenvalues corresponding to each basis vector.

        Returns
        -------
        basis_extended : Dataset of same type as the data
            Transformed value of the given values.
        """
        Y, edges, Y_input_type = _as_flat(Y)
        Y = np.asanyarray(Y)
        in_domain = _as_flat(in_domain)[0].ravel()
        basis = _as_flat(basis)[0]
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        if np.array_equal(Y, self.dmap.data):
            return _flat_to_orig(basis, edges, Y_input_type)

        if self.dmap.oos == "nystroem":
            basis_extended = nystroem_oos(self.dmap, Y, basis, evals)
        elif self.dmap.oos == "power":
            basis_extended = power_oos(self.dmap, Y, basis, evals)
        else:
            raise ValueError('Did not understand the OOS algorithm specified')
        outside_locs = np.where(1 - in_domain)[0]
        basis_extended[outside_locs] = 0.
        return _flat_to_orig(basis_extended, edges, Y_input_type)

    def make_FK_soln(self, b, in_domain):
        """
        Solves a Feynman-Kac problem on the data.
        Specifically, solves Lx = b on the domain and x=b off of the domain.
        In the DGA framework, this is intended to be used to solve for guess functions.

        Parameters
        ----------
        b : 1D array-like, OR list of such arrays, OR flat data format.
            Dataset of the same shape as the input datapoints.  Right hand side of the Feynman-Kac equation.
        in_domain : 1D array-like, OR list of such arrays, OR flat data format.
            Dataset of the same shape as the input datapoints, where each element is 1 or True if that datapoint is inside the domain, and 0 or False if it is in the domain.

        Returns
        -------
        soln : Dataset of same type as the data. 
            Solution to the Feynman-Kac problem.
        """
        in_domain = _as_flat(in_domain)[0].ravel()
        b = _as_flat(b)[0].ravel()
        domain_locs = np.where(in_domain)[0]
        complement_locs = np.where(1. - in_domain)[0]

        # Solve the FK problem
        L_sub = self.dmap.L[domain_locs, :]
        L_comp = L_sub[:, complement_locs]
        b_comp = b[complement_locs]
        Lb = L_comp.dot(b_comp)
        L_sub = L_sub[:, domain_locs]

        # Add the boundary conditions back in.
        soln_sub = spsl.spsolve(L_sub, b[domain_locs] - Lb)
        soln = np.copy(b)
        soln[domain_locs] = soln_sub
        return _flat_to_orig(soln, self.edges, self.input_type)

    def extend_FK_soln(self, soln, Y, b, in_domain):
        """
        Extends the values of the Feynman-Kac solution onto new points.
        In the DGA framework, this is intended to be used to extend guess functions onto new datapoints.

        Parameters
        ----------
        soln : Dataset of same type as the data. 
            Solution to the Feynman-Kac problem on the original type.
        Y : 2D array-like OR list of trajectories OR flat data format
            Data for which to perform the out-of-sample extension.
        b :1D array-like, OR list of such arrays, OR flat data format.
            Values of the right hand-side for the OOS points.
        in_domain : 1D array-like, OR list of such arrays, OR flat data format.
            Dataset of the same shape as the input datapoints, where each element is 1 or True if that datapoint is inside the domain, and 0 or False if it is in the domain.

        Returns
        -------
        extended_soln : Dataset of same type as the data.
            Solution to the Feynman-Kac problem.
        """
        Y, edges, Y_input_type = _as_flat(Y)
        b = _as_flat(b)[0].ravel()
        in_domain = _as_flat(in_domain)[0].ravel()

        Y = np.asanyarray(Y)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        domain_locs = np.where(in_domain)[0]
        Y_sub = Y[domain_locs]
        L_yx, L_yy = _get_L_oos(self.dmap, Y_sub)
        # L_yx, L_yy = _get_L_oos(self.dmap, Y
        soln_sub = b[domain_locs] - L_yx.dot(soln)
        soln_sub /= L_yy
        soln = np.copy(b)
        soln[domain_locs] = np.copy(soln_sub)
        return _flat_to_orig(soln, edges, Y_input_type)


def nystroem_oos(dmap_object, Y, evecs, evals):
    """
    Performs Nystroem out-of-sample extension to calculate the values of the diffusion coordinates at each given point.

    Parameters
    ----------
    dmap_object : DiffusionMap object
        Diffusion map upon which to perform the out-of-sample extension.
    Y : array-like, shape (n_query, n_features)
        Data for which to perform the out-of-sample extension.

    Returns
    -------
    phi : numpy array, shape (n_query, n_eigenvectors)
        Transformed value of the given values.
    """
    # check if Y is equal to data. If yes, no computation needed.
    # compute the values of the kernel matrix
    kernel_extended = dmap_object.local_kernel.compute(Y)
    weights = dmap_object._compute_weights(dmap_object.local_kernel.data)
    P = dmap_object._left_normalize(dmap_object._right_normalize(kernel_extended, dmap_object.right_norm_vec, weights))
    oos_evecs = P * evecs
    # evals_p = dmap_object.local_kernel.epsilon_fitted * dmap_object.evals + 1.
    # oos_dmap = np.dot(oos_evecs, np.diag(1. / evals_p))
    return oos_evecs


def power_oos(dmap_object, Y, evecs, evals):
    """
    Performs out-of-sample extension to calculate the values of the diffusion coordinates at each given point using the power-like method.

    Parameters
    ----------
    dmap_object : DiffusionMap object
        Diffusion map upon which to perform the out-of-sample extension.
    Y : array-like, shape (n_query, n_features)
        Data for which to perform the out-of-sample extension.

    Returns
    -------
    phi : numpy array, shape (n_query, n_eigenvectors)
        Transformed value of the given values.
    """
    L_yx, L_yy = _get_L_oos(dmap_object, Y)
    adj_evals = evals - L_yy.reshape(-1, 1)
    dot_part = np.array(L_yx.dot(evecs))
    return (1. / adj_evals) * dot_part


def _get_L_oos(dmap_object, Y):
    M = int(Y.shape[0])
    k_yx, y_bandwidths = dmap_object.local_kernel.compute(Y, return_bandwidths=True)  # Evaluate on ref points
    yy_right_norm_vec = dmap_object._make_right_norm_vec(k_yx, y_bandwidths)[1]
    k_yy_diag = dmap_object.local_kernel.kernel_fxn(0, dmap_object.epsilon_fitted)
    data_full = np.vstack([dmap_object.local_kernel.data, Y])
    k_full = sps.hstack([k_yx, sps.eye(M) * k_yy_diag])
    right_norm_full = np.hstack([dmap_object.right_norm_vec, yy_right_norm_vec])
    weights = dmap_object._compute_weights(data_full)

    P = dmap_object._left_normalize(dmap_object._right_normalize(k_full, right_norm_full, weights))
    L = dmap_object._build_generator(P, dmap_object.epsilon_fitted, y_bandwidths)
    L_yx = L[:, :-M]
    L_yy = np.array(L[:, -M:].diagonal())
    return L_yx, L_yy
