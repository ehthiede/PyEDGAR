# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pyedgar import data_manipulation as manip


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
    test_tlist = manip.flat_to_tlist(flat, traj_edges)
    for test_traj, traj in zip(test_tlist, tlist):
        assert(not (test_traj - traj).any())


#
