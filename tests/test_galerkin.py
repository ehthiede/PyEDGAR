# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from pyedgar import galerkin
from pyedgar.data_manipulation import tlist_to_flat, flat_to_tlist

# Make a "random" walk starting uniformly on [-1,1] which literally moves left/right 1 out of 4 tries.


@pytest.fixture(scope='module')
def make_random_walk():
    N = 21
    xg = np.linspace(-1, 1, N)
    traj_list = []
    for i in range(N):
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


@pytest.mark.parametrize('timestep', [0.1, 1.0, 1.])
def test_mfpt(make_random_walk, timestep):
    error_tol = 1E-10
    basis, flat_traj, traj_edges = make_random_walk
    basis = basis[:, 5:16]
    nbasis = 11
    stateA = (flat_traj < -.5) * (flat_traj > 0.5)
    basis_list = flat_to_tlist(basis, traj_edges)
    stateA_list = flat_to_tlist(stateA, traj_edges)
    mfpt = galerkin.compute_mfpt(basis_list, stateA_list, dt=timestep)
    mfpt = tlist_to_flat(mfpt)[0]

    for i in range(nbasis):
        basis_vector = basis[:, i]
        mfpt_true = 2*(i+1)*(11-i)
        if timestep is not None:
            mfpt_true *= timestep
        mfpt_i = mfpt[np.where(basis_vector > 0)[0]].flatten()
        error = (mfpt_i - mfpt_true)
        assert(np.all(error < error_tol))


def test_committor(make_random_walk):
    error_tol = 1E-10
    basis, flat_traj, traj_edges = make_random_walk
    basis = basis[:, 5:16]
    nbasis = 11
    stateB = (flat_traj > 0.5).astype('int')
    basis_list = flat_to_tlist(basis, traj_edges)
    stateB_list = flat_to_tlist(stateB, traj_edges)
    committor = galerkin.compute_committor(basis_list, stateB_list)
    committor = tlist_to_flat(committor)[0]
    for i in range(nbasis):
        basis_vector = basis[:, i]
        committor_true = (i+1)*.25/3.
        committor_i = committor[np.where(basis_vector > 0)[0]].flatten()
        error = (committor_i - committor_true)
        assert(np.all(error < error_tol))


def test_bwd_committor(make_random_walk):
    error_tol = 1E-10
    basis, flat_traj, traj_edges = make_random_walk
    basis = basis[:, 5:16].astype('float')
    nbasis = 11
    stateB = (flat_traj > 0.5).astype('float')
    stat_com = np.ones((basis.shape[0], 1)).astype('float')
    basis_list = flat_to_tlist(basis, traj_edges)
    stateB_list = flat_to_tlist(stateB, traj_edges)
    stat_com_list = flat_to_tlist(stat_com, traj_edges)
    committor = galerkin.compute_bwd_committor(basis_list, stateB_list, stat_com_list)
    committor = tlist_to_flat(committor)[0]
    for i in range(nbasis):
        basis_vector = basis[:, i]
        committor_true = (i+1)*.25/3.
        committor_i = committor[np.where(basis_vector > 0)[0]].flatten()
        error = (committor_i - committor_true)
        assert(np.all(error < error_tol))


def test_change_of_measure(make_random_walk):
    error_tol = 1E-10
    basis, flat_traj, traj_edges = make_random_walk
    basis_list = flat_to_tlist(basis, traj_edges)
    change_of_measure = galerkin.compute_change_of_measure(basis_list)
    # change_of_measure = (change_of_measure.get_flat_data()[0]).flatten()
    change_of_measure = tlist_to_flat(change_of_measure)[0]
    change_of_measure /= np.sum(change_of_measure)
    error = change_of_measure - 1/21
    assert(np.all(error < error_tol))


class TestEsystem(object):
    eval_error_tol = 1E-5
    evec_error_tol = .1
    precomputed_evals = np.array([0, -5.5836E-3, -2.2214E-2, -4.9516E-2, -8.6881E-2])
    xax = np.linspace(-1., 1., 21)
    true_evec_coeffs = np.array([np.ones(xax.shape),
                                 -np.sin(.5*np.pi*xax),
                                 -np.cos(np.pi*xax),
                                 np.sin(1.5*np.pi*xax),
                                 np.cos(2.*np.pi*xax)]).T
    true_evec_coeffs /= np.linalg.norm(true_evec_coeffs, axis=0)

    @pytest.mark.parametrize('left', [True, False])
    @pytest.mark.parametrize('right', [True, False])
    def test_top_evecs_are_correct(self, make_random_walk, left, right):
        # Setup
        basis, flat_traj, traj_edges = make_random_walk
        # basis_dset = DynamicalDataset((basis, traj_edges))
        basis_list = flat_to_tlist(basis, traj_edges)
        true_evecs = np.dot(basis, self.true_evec_coeffs)
        # Evaluate eigensystem
        if (left and right):
            evals, left_evecs, right_evecs = galerkin.compute_esystem(basis_list, left=True, right=True)
        elif left:
            evals, left_evecs = galerkin.compute_esystem(basis_list, left=True, right=False)
        elif right:
            evals, right_evecs = galerkin.compute_esystem(basis_list, left=False, right=True)
        else:
            return
        # Move to flat convention for easier preparation
        if left:
            left_evecs = tlist_to_flat(left_evecs)[0]
        if right:
            right_evecs = tlist_to_flat(right_evecs)[0]
        # Check right, left eigenvectors.
        for i in range(5):
            if left:
                left_evec_i = left_evecs[:, i]
                left_evec_i *= np.sign(left_evec_i[0])/np.linalg.norm(left_evec_i)
                left_evec_i_diff = left_evec_i - true_evecs[:, i]
                left_evec_i_error = np.linalg.norm(left_evec_i_diff)
                left_evec_i_error /= len(left_evec_i_diff)
                assert(left_evec_i_error < self.evec_error_tol)
            if right:
                right_evec_i = right_evecs[:, i]
                right_evec_i *= np.sign(right_evec_i[0])
                right_evec_i_diff = right_evec_i - true_evecs[:, i]
                right_evec_i_error = np.linalg.norm(right_evec_i_diff)
                right_evec_i_error /= len(right_evec_i_diff)
                assert(right_evec_i_error < self.evec_error_tol)

    @pytest.mark.parametrize('left', [True, False])
    @pytest.mark.parametrize('right', [True, False])
    def test_top_evals_are_correct(self, make_random_walk, left, right):
        # Setup
        basis, flat_traj, traj_edges = make_random_walk
        # basis_dset = DynamicalDataset((basis, traj_edges))
        basis_list = flat_to_tlist(basis, traj_edges)
        # Evaluate eigensystem
        if (left or right):
            evals = galerkin.compute_esystem(basis_list, left=left, right=right)[0]
            print(evals, left, right)
        else:
            evals = galerkin.compute_esystem(basis_list, left=False, right=False)
        evals_error = np.linalg.norm(evals[:5] - self.precomputed_evals)
        assert(evals_error < self.eval_error_tol)
