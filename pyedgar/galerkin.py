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
    lag : int
        Number of timepoints in the future to use for the discrete-time generator.
    dt : scalar 
        Time between timepoints in the trajectory data.
    

    Returns
    -------
    mfpt : dynamical basis object
        Dynamical dataset object containing the values of the mean first passage time at each point. 

    """


def get_committor(basis,stateA,stateB,test_fxn=None,delay=1):
    """Calculates the mean first passage time into state A as a function of each point.

    Parameters
    ----------
    basis : dynamical dataset object
        Dynamical dataset object containing the basis for the Galerkin expansion.  This method works much better if the basis set is zero on states A and B, however this is not a necessity. 
    state_A : dynamical dataset object
        Dynamical dataset object whose values are 1 or 0, corresponding to whether or not the datapoint is in state A.
    state_B : dynamical dataset object
        Dynamical dataset object whose values are 1 or 0, corresponding to whether or not the datapoint is in state B.
    test_fxn : dynamical dataset object
        The value of the test function obeying the inhomogenous boundary conditions.  If not given, taken to just be state B. 
    lag : int
        Number of timepoints in the future to use for the discrete-time generator.
    dt : scalar 
        Time between timepoints in the trajectory data.
    

    Returns
    -------
    committor: dynamical basis object
        Dynamical dataset object containing the values of the committor at each point. 

    """


def get_stationary_distrib(basis,traj_edges,test_set=None,delay=1,fix=0):
    """

    """


def get_esystem(basis,traj_edges,test_set=None,delay=1):
    """

    """
