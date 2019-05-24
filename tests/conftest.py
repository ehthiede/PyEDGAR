# -*- coding: utf-8 -*-
"""
This is the conftest file.   Here, we setup the fixtures (test data, etc.) for the rest of the tests.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest


@pytest.fixture(scope='session')
def working_flat_and_tlist():
    flat = np.zeros((20, 3))
    flat[:, 0] = np.arange(20) - 5
    flat[:, 1] = np.arange(0, -20, -1) + 3
    flat[:, 2] = np.arange(20) + 1
    te = [0, 8, 11, 20]
    tlist = [flat[te[0]:te[1]], flat[te[1]:te[2]], flat[te[2]:te[3]]]
    return (flat, te, tlist)


@pytest.fixture(scope='session')
def flat_1d_system():
    xax = np.linspace(-1, 1, 201)
    yax = np.linspace(-1, 1, 75)
    in_domain_x = ((xax > -0.5) * (xax < 0.5)).astype('int')
    in_domain_y = ((yax > -0.5) * (yax < 0.5)).astype('int')

    true_evecs_x = get_bounded_1d_flat_evecs(xax, in_domain_x)
    true_evecs_y = get_bounded_1d_flat_evecs(yax, in_domain_y)

    return (xax, in_domain_x, true_evecs_x, yax, in_domain_y, true_evecs_y)


def get_bounded_1d_flat_evecs(points, in_domain):
    domain_locs = np.where(in_domain)[0]
    points_sub = points[domain_locs]
    true_evecs_sub = np.array([np.cos(np.pi * points_sub),
                               np.sin(2. * np.pi * points_sub),
                               np.cos(3. * np.pi * points_sub),
                               np.sin(4 * np.pi * points_sub),
                               np.cos(5. * np.pi * points_sub)]).T
    n_evecs = len(true_evecs_sub[0])
    true_evecs = np.zeros((points.shape[0], n_evecs))
    true_evecs[domain_locs] = true_evecs_sub
    return true_evecs
