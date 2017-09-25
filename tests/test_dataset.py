# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np

from test_data_manipulation import working_flat_and_tlist


def test_dataset_creation():
    flat = np.zeros((20, 3))
    flat[:, 0] = np.arange(20) - 5
    flat[:, 1] = -1 * np.arange(20) + 3
    flat[:, 2] = np.arange(20) + 1


print(working_flat_and_tlist)
