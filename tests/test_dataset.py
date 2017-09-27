# -*- coding: utf-8 -*-
"""
Test functions for testing the dynamical dataset object class.

TODO:
    - Parameterize this code?
    - Implement tests for returning datastructurs
    - Implement tests for initial / final split
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from pyedgar import DynamicalDataset

lags = range(1, 4)
timesteps = [1., 0.5, 2, 1E4]


class TestDatasetCreation(object):
    @pytest.mark.parametrize('lag', lags)
    @pytest.mark.parametrize('timestep', timesteps)
    def test_from_flat(self, working_flat_and_tlist, lag, timestep):
        flat, traj_edges, tlist = working_flat_and_tlist
        ds_from_flat = DynamicalDataset((flat, traj_edges), lag=lag, timestep=timestep)
        assert(np.all(ds_from_flat.traj_edges == traj_edges))
        assert(np.all(ds_from_flat.flat_traj == flat))
        assert(ds_from_flat.lag == lag)
        assert(ds_from_flat.timestep == timestep)

    @pytest.mark.parametrize('lag', lags)
    @pytest.mark.parametrize('timestep', timesteps)
    def test_from_tlist(self, working_flat_and_tlist, lag, timestep):
        flat, traj_edges, tlist = working_flat_and_tlist
        ds_from_flat = DynamicalDataset(tlist, lag=lag, timestep=timestep)
        assert(np.all(ds_from_flat.traj_edges == traj_edges))
        assert(np.all(ds_from_flat.flat_traj == flat))
        assert(np.all(ds_from_flat.lag == lag))
        assert(np.all(ds_from_flat.timestep == timestep))

    @pytest.mark.parametrize('lag', lags)
    @pytest.mark.parametrize('timestep', timesteps)
    def test_from_single_traj(self, working_flat_and_tlist, lag, timestep):
        flat, traj_edges, tlist = working_flat_and_tlist
        traj = tlist[0]
        traj_edges = np.array([0, len(traj)])
        ds_from_flat = DynamicalDataset(traj, lag=lag, timestep=timestep)
        assert(np.all(ds_from_flat.traj_edges == traj_edges))
        assert(np.all(ds_from_flat.flat_traj == traj))
        assert(np.all(ds_from_flat.lag == lag))
        assert(np.all(ds_from_flat.timestep == timestep))

    def test_bad_input(self):
        with pytest.raises(ValueError) as excinfo:
            ds_bad = DynamicalDataset('bad_input')



class TestDataReturn():
    
