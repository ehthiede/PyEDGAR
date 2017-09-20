# -*- coding: utf-8 -*-
"""Routines and Class definitions for constructing basis sets using the
diffusion maps algorithm.

@author: Erik

"""
from __future__ import absolute_import, division, print_function

import numpy as np


class DiffusionAtlas(object):
    """The diffusion atlas is a factory object for constructing diffusion map
    bases with various boundary conditions."""

    def __init__(self, nneighbors=600, rho=None, point_weights=None, d=None,
                 alpha='0', beta='-1/(d+2)', epses=2.**np.arange(-40, 41),
                 rho_norm=True, metric='euclidean', metric_params=None):
        """Constructs the factory object.  The factory object can then be
        called to make diffusion map bases of various boundary conditions.

        Parameters
        ----------
        nneighb : int or None, optional
            Number of neighbors to include in constructing the diffusion map.  If None, all neighbors are used.  Default is 600 neighbors
        rho : 1d array-like or None, optional
            Bandwidth function to be used in the variable bandwidth kernel.
            If None, the code estimates the density of the data q using a kernel density estimate,
            and sets the bandwidth to q_\epsilon^beta.
        weights : 1D array-like or None, optional
            Importance sampling weights for each datapoint.
        d : int or None, optional
            Dimension of the system. If None and alpha or beta settings require the dimensionality,
            the dimension is estimated using the kernel density estimate,
            if a kernel density estimate is performed.
        alpha : float or string, optional
            Parameter for left normalization of the Diffusion map.
            Either a float, or a string that can be interpreted as a mathematical expression.
            The variable "d" stands for dimension, so "1/d" sets the alpha to one over the system dimension.
            Default is 0
        beta : float or string, optional
            Parameter for constructing the bandwidth function for the Diffusion map.  If rho is None, the bandwidth function will be set to q_\epsilon^beta, where q_\epsilon is an estimate of the density constructed using a kernel density estimate.  If rho is provided, this parameter is unused.  As with alpha, this will interpret strings that are evaluatable expressions.  Default is -1/(d+2)
        epses: float or 1d array, optional
            Bandwidth constant to use.  If float, uses that value for the bandwidth.  If array, performs automatic bandwidth detection according to the algorithm given by Berry and Giannakis and Harlim.  Default is all powers of 2 from 2^-40 to 2^40.
        rho_norm : bool, optional
            Whether or not to normalize q and L by rho(x)^2.  Default is True (perform normalization)
        metric : string
            Metric to use for computing distance.  Default is "Euclidean".  See sklearn documentation for more options.
        metric_params : dict
            Additional parameters needed for estimating the metric.

        """
        pass

    def fit(self, data):
        """Constructs the diffusion map on the dataset.

        Parameters
        ----------
        data : 2D array-like or dynamical dataset
            Two-dimensional dataset used to create the diffusion map.

        """
        return

    def make_dirichlet_basis(self, in_domain, k):
        """Creates a diffusion map basis set that obeys the homogeneous
        Dirichlet boundary conditions on the domain.  This is done by taking
        the eigenfunctions of the diffusion map submatrix on the domain.

        Parameters
        ----------
        in_domain : 1D array-like
            Array of the same length as the data, where each element is 1 or True if that datapoint is in the domain, and 0 or False if it is outside the domain.  Naturally, this must be the length as the current dataset.
        k : int
            Number of basis functions to create.

        Returns
        -------
        basis : 2D array-like
            The basis functions.

        """
        return

    def extend_oos(coords, function_values):
        """Performs out-of-sample extension for given values of a function.

        Parameters
        ----------
        coords : 1 or 2d array
            The new coordinates upon which to perform the out-of-sample extension.
        function_values : 1d array
            The value of the function at each point in the diffusion map.

        Returns
        -------
        new_values : 1d array
            The estimated value of the function at the new points.

        """
        return
