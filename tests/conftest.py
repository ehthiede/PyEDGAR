# -*- coding: utf-8 -*-
"""
This is the conftest file.   Here, we setup the fixtures (test data, etc.) for the rest of the tests.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

lags = [None] + range(1, 4)


@pytest.fixture(scope="session")
def working_flat_and_tlist():
    flat = np.zeros((20, 3))
    flat[:, 0] = np.arange(20) - 5
    flat[:, 1] = np.arange(0, -20, -1) + 3
    flat[:, 2] = np.arange(20) + 1
    te = [0, 8, 11, 20]
    tlist = [flat[te[0]:te[1]], flat[te[1]:te[2]], flat[te[2]:te[3]]]
    return (flat, te, tlist)
