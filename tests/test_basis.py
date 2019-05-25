# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
import pyedgar.basis

from pyedgar import data_manipulation as manip


class TestBasisFormation(object):
    evec_error_tol = .1

    @pytest.mark.parametrize('input_type', ['flat', 'tlist', 'array'])
    def test_full_basis_flat(self, flat_1d_system, input_type):
        xax = np.linspace(-1, 1, 201)
        traj_edges = np.array([0, 50, len(xax)])
        # Define the true eigenvectors of the random walk.
        true_evecs = np.array([np.ones(xax.shape),
                               -np.sin(.5*np.pi*xax),
                               -np.cos(np.pi*xax),
                               np.sin(1.5*np.pi*xax),
                               np.cos(2.*np.pi*xax)]).T
        true_evecs /= np.linalg.norm(true_evecs, axis=0)
        n_evecs = len(true_evecs[0])

        # Convert to right input data format
        if input_type is 'flat':
            xax = (xax, traj_edges)
        elif input_type is 'tlist':
            xax = manip.flat_to_tlist(xax, traj_edges)
        # Construct the diffusion map atlas.
        diff_atlas = pyedgar.basis.DiffusionAtlas()
        diff_atlas.fit(xax)
        basis, evals = diff_atlas.make_dirichlet_basis(k=n_evecs, return_evals=True)
        if input_type is 'flat':
            basis = basis[0]
        elif input_type is 'tlist':
            basis = manip.tlist_to_flat(basis)[0]
        basis /= np.linalg.norm(basis, axis=0) * np.sign(basis[0])
        for i in range(n_evecs):
            basis_i = basis[:, i]
            true_evec_i = true_evecs[:, i]
            diff = basis_i - true_evec_i
            assert(_calculate_RMSE(diff) < self.evec_error_tol)

    @pytest.mark.parametrize('input_type', ['flat', 'tlist', 'array'])
    def test_bounded_flat(self, flat_1d_system, input_type):
        # Define parameters
        xax, in_domain, true_evecs = flat_1d_system[:3]
        traj_edges = np.array([0, 50, len(xax)])
        if input_type is 'flat':
            xax = (xax, traj_edges)
            in_domain = (in_domain, traj_edges)
        elif input_type is 'tlist':
            xax = manip.flat_to_tlist(xax, traj_edges)
            in_domain = manip.flat_to_tlist(in_domain, traj_edges)
        true_evecs = np.copy(true_evecs)
        true_evecs /= np.linalg.norm(true_evecs, axis=0)
        n_evecs = len(true_evecs[0])

        # Construct the diffusion map atlas.
        diff_atlas = pyedgar.basis.DiffusionAtlas()
        diff_atlas.fit(xax)
        basis, evals = diff_atlas.make_dirichlet_basis(k=n_evecs, in_domain=in_domain, return_evals=True)
        if input_type is 'flat':
            basis = basis[0]
        elif input_type is 'tlist':
            basis = manip.tlist_to_flat(basis)[0]
        basis /= np.linalg.norm(basis, axis=0)
        for i in range(n_evecs):
            basis_i = basis[:, i]
            true_evec_i = true_evecs[:, i]
            # Check for different signs
            err_1 = _calculate_RMSE(basis_i - true_evec_i)
            err_2 = _calculate_RMSE(basis_i + true_evec_i)
            error = np.minimum(err_1, err_2)
            assert(error < self.evec_error_tol)

    @pytest.mark.parametrize('method', ['nystroem', 'power'])
    @pytest.mark.parametrize('repeat', [True, False])
    @pytest.mark.parametrize('input_type', ['flat', 'tlist', 'array'])
    def test_basis_extension_flat(self, flat_1d_system, method, repeat, input_type):
        data = flat_1d_system
        xax, in_domain_x, true_evecs_x, yax, in_domain_y, true_evecs_y = data
        if repeat:
            yax, in_domain_y, true_evecs_y = (xax, in_domain_x, true_evecs_x)
        Nx, n_evecs = true_evecs_x.shape

        # Make various inputs
        traj_edges = np.array([0, 50, len(xax)])
        if input_type is 'flat':
            yax = (yax, traj_edges)
            in_domain_y = (in_domain_y, traj_edges)
        elif input_type is 'tlist':
            yax = manip.flat_to_tlist(yax, traj_edges)
            in_domain_y = (in_domain_y, traj_edges)

        # Construct the diffusion map atlas.
        diff_atlas = pyedgar.basis.DiffusionAtlas.from_sklearn(oos=method)
        diff_atlas.fit(xax)
        evals = np.pi**2 * np.arange(1, 6)**2
        basis_extended = diff_atlas.extend_dirichlet_basis(yax, in_domain_y, true_evecs_x, evals)
        if input_type is 'flat':
            basis_extended = basis_extended[0]
        elif input_type is 'tlist':
            basis_extended = manip.tlist_to_flat(basis_extended)[0]
        for i in range(n_evecs):
            basis_i = basis_extended[:, i]
            true_evec_i = true_evecs_y[:, i]
            # Check for different signs
            err_1 = _calculate_RMSE(basis_i - true_evec_i)
            err_2 = _calculate_RMSE(basis_i + true_evec_i)
            error = np.minimum(err_1, err_2)
            assert(error < self.evec_error_tol)

    @pytest.mark.parametrize('input_type', ['flat', 'tlist', 'array'])
    def test_guess_flat(self, flat_1d_system, input_type):
        # Define parameters
        xax, in_domain, true_evecs = flat_1d_system[:3]
        comm_b = (xax >= 0.5).astype('float')
        true_soln = np.copy(comm_b)
        domain_locs = np.where(in_domain > 0.)[0]
        true_soln[domain_locs] = xax[domain_locs] + 0.5

        # Convert to right input data format
        traj_edges = np.array([0, 50, len(xax)])
        if input_type is 'flat':
            xax = (xax, traj_edges)
            in_domain = (in_domain, traj_edges)
        elif input_type is 'tlist':
            xax = manip.flat_to_tlist(xax, traj_edges)
            in_domain = manip.flat_to_tlist(in_domain, traj_edges)
            comm_b = manip.flat_to_tlist(comm_b, traj_edges)
        # Construct the diffusion map atlas.
        diff_atlas = pyedgar.basis.DiffusionAtlas()
        diff_atlas.fit(xax)
        soln = diff_atlas.make_FK_soln(comm_b, in_domain)
        
        if input_type is 'flat':
            soln = soln[0].ravel()
        elif input_type is 'tlist':
            soln = manip.tlist_to_flat(soln)[0].ravel()

        print(_calculate_RMSE(true_soln - soln))
        assert(_calculate_RMSE(true_soln - soln) < self.evec_error_tol)

    @pytest.mark.parametrize('repeat', [True, False])
    @pytest.mark.parametrize('input_type', ['flat', 'tlist', 'array'])
    def test_guess_extension_flat(self, flat_1d_system, repeat, input_type):
        # Load data
        data = flat_1d_system
        xax, in_domain_x, true_evecs_x, yax, in_domain_y, true_evecs_y = data
        if repeat:
            yax, in_domain_y = (xax, in_domain_x)
        comm_b_x = (xax >= 0.5).astype('float')
        comm_b_y = (yax >= 0.5).astype('float')

        # Get True Solutions
        true_soln_x = np.copy(comm_b_x)
        domain_locs_x = np.where(in_domain_x > 0.)[0]
        true_soln_x[domain_locs_x] = xax[domain_locs_x] + 0.5
        true_soln_y = np.copy(comm_b_y)
        domain_locs_y = np.where(in_domain_y > 0.)[0]
        true_soln_y[domain_locs_y] = yax[domain_locs_y] + 0.5
        
        # Make various inputs
        traj_edges = np.array([0, 50, len(xax)])
        if input_type is 'flat':
            yax = (yax, traj_edges)
            in_domain_y = (in_domain_y, traj_edges)
        elif input_type is 'tlist':
            yax = manip.flat_to_tlist(yax, traj_edges)
            comm_b_y = manip.flat_to_tlist(comm_b_y, traj_edges)

        # Construct the diffusion map atlas.
        diff_atlas = pyedgar.basis.DiffusionAtlas.from_sklearn()
        diff_atlas.fit(xax)
        soln = diff_atlas.extend_FK_soln(true_soln_x, yax, comm_b_y, in_domain_y)
        
        if input_type is 'flat':
            soln = soln[0].ravel()
        elif input_type is 'tlist':
            soln = manip.tlist_to_flat(soln)[0].ravel()
        assert(_calculate_RMSE(true_soln_y - soln) < self.evec_error_tol)


def _calculate_RMSE(vec):
    return np.sqrt(np.mean(vec**2))
