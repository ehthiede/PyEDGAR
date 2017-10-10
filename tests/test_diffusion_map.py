# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pyedgar.diffusion_map as dmap


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
        diff_atlas = dmap.DiffusionAtlas()
        diff_atlas.fit(xax)
        basis = diff_atlas.make_dirichlet_basis(k=n_evecs)[0]
        basis /= np.linalg.norm(basis, axis=0) * np.sign(basis[0])
        for i in range(n_evecs):
            basis_i = basis[:, i]
            true_evec_i = true_evecs[:, i]
            diff = basis_i - true_evec_i
            assert(np.linalg.norm(diff) < self.evec_error_tol)
