# -*- coding: utf-8 -*-
"""
Routines for constructing estimates of dynamical quantities on trajectory data using Galerkin expansion.
@author: Erik
"""


def get_mfpt(basis,state_A,lag=1,dt=1.):
    """Calculates the mean first passage time into state A as a function of each point.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.  This method works much better if the basis set is zero on state A, however this is not a necessity. 
    state_A : dynamical dataset object
        Dynamical dataset object whose values are 1 or 0, corresponding to whether or not the datapoint is in state A.
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.
    dt : scalar, optional
        Time between timepoints in the trajectory data.
    

    Returns
    -------
    mfpt : dynamical basis object
        Dynamical dataset object containing the values of the mean first passage time at each point. 

    """
    return

def get_committor(basis,stateA,stateB,test_fxn=None,lag=1):
    """Calculates the mean first passage time into state A as a function of each point.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.  This method works much better if the basis set is zero on states A and B, however this is not a necessity. 
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
    return


def get_stationary_distrib(basis,lag=1):
    """Calculates the value of the stationary distribution for each datapoint.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.  This method works much better if the basis set is zero on states A and B, however this is not a necessity. 
    lag : int, optional
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.

    Returns
    -------
    stationary_distribution : dynamical basis object
        Dynamical dataset object containing the values of the stationary distribution at each point. 

    """
    return


def get_esystem(basis,lag=1,left=False,right=True):
    """Calculates the eigenvectors and eigenvalues of the generator through Galerkin expansion.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.  This method works much better if the basis set is zero on states A and B, however this is not a necessity. 
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
    return
