# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from pyedgar import DynamicalDataset, galerkin

# Make a "random" walk starting uniformly on [-1,1] which literally moves left/right 1 out of 4 tries.


@pytest.fixture(scope='module')
def make_random_walk():
    N = 21
    xg = np.linspace(-1, 1, N)
    traj_list = []
    for i in xrange(N):
        x = xg[i]
        x_bwd = xg[np.maximum((i-1), 0)]
        x_fwd = xg[np.minimum((i+1), N-1)]
        traj_list += [np.array([x, x_bwd]), np.array([x, x]),
                      np.array([x, x]), np.array([x, x_fwd])]
    totally_flat_traj = np.array(traj_list).flatten()
    flat_traj = totally_flat_traj.reshape(-1, 1)
    traj_edges = np.arange(0, len(flat_traj)+1, 2)
    basis = np.array([totally_flat_traj == y for y in xg]).astype('int')
    basis = basis.T
    return (basis, flat_traj, traj_edges)


def test_mfpt(make_random_walk):
    error_tol = 1E-10
    basis, flat_traj, traj_edges = make_random_walk
    basis = basis[:, 5:16]
    nbasis = 11
    stateA = (flat_traj < -.5) * (flat_traj > 0.5)
    stateA_dset = DynamicalDataset((stateA, traj_edges))
    basis_dset = DynamicalDataset((basis, traj_edges))
    mfpt = galerkin.compute_mfpt(basis_dset, stateA_dset).get_flat_data()[0]
    for i in range(nbasis):
        basis_vector = basis[:, i]
        mfpt_true = 2*(i+1)*(11-i)
        print(np.where(basis_vector > 0)[0])
        mfpt_i = mfpt[np.where(basis_vector > 0)[0]].flatten()
        error = (mfpt_i - mfpt_true)
        assert(np.all(error < error_tol))


def test_committor(make_random_walk):
    error_tol = 1E-10
    basis, flat_traj, traj_edges = make_random_walk
    basis = basis[:, 5:16]
    nbasis = 11
    stateA = (flat_traj < -.5)
    stateB = (flat_traj > 0.5)
    stateA_dset = DynamicalDataset((stateA, traj_edges))
    stateB_dset = DynamicalDataset((stateB, traj_edges))
    basis_dset = DynamicalDataset((basis, traj_edges))
    committor = galerkin.compute_committor(basis_dset, stateA_dset, stateB_dset).get_flat_data()[0]
    for i in range(nbasis):
        basis_vector = basis[:, i]
        committor_true = (i+1)*.25/3.
        committor_i = committor[np.where(basis_vector > 0)[0]].flatten()
        error = (committor_i - committor_true)
        assert(np.all(error < error_tol))


def test_change_of_measure(make_random_walk):
    error_tol = 1E-10
    basis, flat_traj, traj_edges = make_random_walk
    basis_dset = DynamicalDataset((basis, traj_edges))
    change_of_measure = galerkin.compute_change_of_measure(basis_dset)
    change_of_measure = (change_of_measure.get_flat_data()[0]).flatten()
    change_of_measure /= np.sum(change_of_measure)
    error = change_of_measure - 1/21
    assert(np.all(error < error_tol))
