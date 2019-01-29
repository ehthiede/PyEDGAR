# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pyedgar.basis


class TestBasisFormation(object):
    evec_error_tol = .1

    def test_full_basis_flat(self):
        # Define parameters
        xax = np.linspace(-1, 1, 201)
        # Define the true eigenvectors of the random walk.
        true_evecs = np.array([np.ones(xax.shape),
                               -np.sin(.5*np.pi*xax),
                               -np.cos(np.pi*xax),
                               np.sin(1.5*np.pi*xax),
                               np.cos(2.*np.pi*xax)]).T
        true_evecs /= np.linalg.norm(true_evecs, axis=0)
        n_evecs = len(true_evecs[0])
        # Construct the diffusion map atlas.
        diff_atlas = pyedgar.basis.DiffusionAtlas()
        diff_atlas.fit(xax)
        basis = diff_atlas.make_dirichlet_basis(k=n_evecs)
        basis /= np.linalg.norm(basis, axis=0) * np.sign(basis[0])
        for i in range(n_evecs):
            basis_i = basis[:, i]
            true_evec_i = true_evecs[:, i]
            diff = basis_i - true_evec_i
            assert(np.linalg.norm(diff) < self.evec_error_tol)

    def test_bounded_flat(self):
        # Define parameters
        xax = np.linspace(-1, 1, 201)
        in_domain = ((xax > -0.5) * (xax < 0.5)).astype('int')
        domain_locs = np.where(in_domain)[0]
        xsub = xax[domain_locs]
        true_evecs_sub = np.array([np.cos(np.pi*xsub),
                                   np.sin(2.*np.pi*xsub),
                                   np.cos(3.*np.pi*xsub),
                                   np.sin(4*np.pi*xsub),
                                   np.cos(5.*np.pi*xsub)]).T
        n_evecs = len(true_evecs_sub[0])
        true_evecs = np.zeros((201, n_evecs))
        true_evecs[domain_locs] = true_evecs_sub
        true_evecs /= np.linalg.norm(true_evecs, axis=0)

        # Construct the diffusion map atlas.
        diff_atlas = pyedgar.basis.DiffusionAtlas()
        diff_atlas.fit(xax)
        basis = diff_atlas.make_dirichlet_basis(k=n_evecs, in_domain=in_domain)
        basis /= np.linalg.norm(basis, axis=0)
        for i in range(n_evecs):
            basis_i = basis[:, i]
            true_evec_i = true_evecs[:, i]
            # Check for different signs
            err_1 = np.linalg.norm(basis_i - true_evec_i)
            err_2 = np.linalg.norm(basis_i + true_evec_i)
            error = np.minimum(err_1, err_2)
            assert(error < self.evec_error_tol)
