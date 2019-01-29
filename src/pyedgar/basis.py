# -*- coding: utf-8 -*-
"""Routines and Class definitions for constructing basis sets using the
diffusion maps algorithm.

@author: Erik

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse.linalg as spsl

from pydiffmap.diffusion_map import DiffusionMap


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
                                                    bandwidth_type='1/d')
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
        data : 2D array-like or dynamical dataset
            Two-dimensional dataset used to create the diffusion map.
        rho : 1d array-like or None, optional
            Bandwidth function to be used in the variable bandwidth kernel.
            If None, the code estimates the density of the data q using a kernel density estimate,
            and sets the bandwidth to q_\epsilon^beta.
        point_weights : 1D array-like or None, optional
            Importance sampling weights for each datapoint.

        """
        # Default Parameter Selection and Type Cleaning
        data = np.asanyarray(data)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        self.data = data
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
        in_domain : 1D array-like, optional
            Array of the same length as the data, where each element is 1 or True if that datapoint is inside the domain, and 0 or False if it is in the domain.  Naturally, this must be the length as the current dataset.  If None (default), all points assumed to be in the domain.


        Returns
        -------
        basis : 2D numpy array
            The basis functions.
        evals : 1D numpy array
            The eigenvalues corresponding to each basis vector.

        """
        submat = self.dmap.L
        npoints = submat.shape[0]
        # Take submatrix if necessary
        if in_domain is not None:
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
        if return_evals:
            return full_evecs, evals
        else:
            return full_evecs
