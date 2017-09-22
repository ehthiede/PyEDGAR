# -*- coding: utf-8 -*-
from pyedgar import data_manipulation as manip
import pytest
import numpy as np


@pytest.fixture
def working_flat_and_tlist():
    flat = np.zeros((20, 3))
    flat[:, 0] = np.arange(20) - 5
    flat[:, 1] = -1 * np.arange(20) + 3
    flat[:, 2] = np.arange(20) + 1
    te = [0, 8, 11, 20]
    tlist = [flat[te[0]:te[1]], flat[te[1]:te[2]], flat[te[2]:te[3]]]
    return (flat, te, tlist)

# Tests for the tlist to flat function.


def test_tlist_to_flat__basic_functionality(working_flat_and_tlist):
    flat, traj_edges, tlist = working_flat_and_tlist
    test_flat, test_traj_edges = manip.tlist_to_flat(tlist)
    flat_diff = test_flat - flat
    te_diff = np.array(test_traj_edges) - np.array(traj_edges)
    assert(not (flat_diff).any())
    assert(not (te_diff).any())


# Tests for the flat to tlist function.
def test_flat_to_tlist__basic_functionality(working_flat_and_tlist):
    flat, traj_edges, tlist = working_flat_and_tlist
    print flat, traj_edges
    test_tlist = manip.flat_to_tlist(flat, traj_edges)
    for test_traj, traj in zip(test_tlist, tlist):
        assert(not (test_traj - traj).any())


#
