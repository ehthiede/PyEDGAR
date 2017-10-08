# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

import pyedgar.diffusion_map as dmap

@pytest.fixture(scope='module')
def make_flat_potential_data():
    x = np.linspace(-1,1,201)

@pytest.mark.skip(reason="Code not finished yet")
class TestBasisFormation(object):

    def test_full_basis_flat()
        x_1d = np.linspace(-1,1,201)
        diff_atlas = dmap.diffusion_atlas()
        diff_atlas.fit(x_1d)
        basis = diff_atlas.make_dirichlet_basis(5)
    
