# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pyedgar import data_manipulation as manip


def test_tlist_to_flat(working_flat_and_tlist):
    flat, traj_edges, tlist = working_flat_and_tlist
    test_flat, test_traj_edges = manip.tlist_to_flat(tlist)
    assert(np.all(test_flat == flat))
    assert(list(test_traj_edges) == traj_edges)


def test_flat_to_tlist(working_flat_and_tlist):
    flat, traj_edges, tlist = working_flat_and_tlist
    test_tlist = manip.flat_to_tlist(flat, traj_edges)
    for test_traj, traj in zip(test_tlist, tlist):
        assert(np.all(test_traj == traj))
