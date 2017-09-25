# -*- coding: utf-8 -*-
"""
Test functions for testing the dynamical dataset object class.

TODO:
    - Parameterize this code!
    - Implement tests for returning datastructurs
    - Implement tests for initial / final split
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pyedgar import DynamicalDataset


class TestDatasetCreation(object):
    def test_from_flat(self, working_flat_and_tlist):
        flat, traj_edges, tlist = working_flat_and_tlist
        DS_from_flat = DynamicalDataset((flat, traj_edges))
        assert(np.all(DS_from_flat.traj_edges == traj_edges))
        assert(np.all(DS_from_flat.flat_traj == flat))
        assert(np.all(DS_from_flat.lag == 1))
        assert(np.all(DS_from_flat.timestep == 1.))

    def test_from_tlist(self, working_flat_and_tlist):
        flat, traj_edges, tlist = working_flat_and_tlist
        DS_from_flat = DynamicalDataset(tlist)
        assert(np.all(DS_from_flat.traj_edges == traj_edges))
        assert(np.all(DS_from_flat.flat_traj == flat))
        assert(np.all(DS_from_flat.lag == 1))
        assert(np.all(DS_from_flat.timestep == 1.))

    def test_from_single_traj(self, working_flat_and_tlist):
        flat, traj_edges, tlist = working_flat_and_tlist
        traj = tlist[0]
        traj_edges = np.array([0, len(traj)])
        DS_from_flat = DynamicalDataset(traj)
        assert(np.all(DS_from_flat.traj_edges == traj_edges))
        assert(np.all(DS_from_flat.flat_traj == traj))
        assert(np.all(DS_from_flat.lag == 1))
        assert(np.all(DS_from_flat.timestep == 1.))
