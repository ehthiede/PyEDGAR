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
        DS_from_flat = DynamicalDataset((flat, traj_edges), lag=lag, timestep=timestep)
        assert(np.all(DS_from_flat.traj_edges == traj_edges))
        assert(np.all(DS_from_flat.flat_traj == flat))
        if lag is None:
            assert(np.all(DS_from_flat.lag == 1))
        else:
            assert(np.all(DS_from_flat.lag == lag))
        if timestep is None:
            assert(np.all(DS_from_flat.timestep == 1.))
        else:
            assert(np.all(DS_from_flat.timestep == timestep))

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


class TestDatasetSplit(object):
    flat_data = np.ones((20, 2))
    traj_edges = [0, 10, 13, 20]
    init_indices_lag_2 = np.array(range(0, 8) + [10] + range(13, 18))
    final_indices_lag_2 = np.array(range(2, 10) + [12] + range(15, 20))
    init_indices_lag_4 = np.array(range(0, 6) + range(13, 16))
    final_indices_lag_4 = np.array(range(4, 10) + range(17, 20))

    def test_use_dataset_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=2)
        init_indices, final_indices = ddata._get_initial_final_split()
        assert(np.all(init_indices == self.init_indices_lag_2))
        assert(np.all(final_indices == self.final_indices_lag_2))

    def test_use_custom_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=1)
        init_indices, final_indices = ddata._get_initial_final_split(lag=2)
        assert(np.all(init_indices == self.init_indices_lag_2))
        assert(np.all(final_indices == self.final_indices_lag_2))

    def test_lag_longer_than_traj(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=4)
        init_indices, final_indices = ddata._get_initial_final_split()
        assert(np.all(init_indices == self.init_indices_lag_4))
        assert(np.all(final_indices == self.final_indices_lag_4))


class TestGenerator(object):
    flat_data = np.ones((11, 2))
    flat_data[:5, 0] = np.arange(1, 6)
    flat_data[5:, 0] = np.arange(5, 11)
    traj_edges = [0, 5, 11]
    generator_at_lag_1 = np.array([[5., 0.], [1., 0.]])
    generator_at_lag_4 = np.array([[4., 0.], [1., 0.]])

    def test_basic_functionality(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        generator = ddata.compute_generator()
        assert(np.all(generator == self.generator_at_lag_1))

    def test_nondefault_input_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        generator_lag_specified = ddata.compute_generator(lag=4)
        assert(np.all(generator_lag_specified == self.generator_at_lag_4))

    def test_nondefault_dataset_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=4)
        generator_lag_from_dset = ddata.compute_generator()
        assert(np.all(generator_lag_from_dset == self.generator_at_lag_4))

    def test_nondefault_timestep(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), timestep=2.)
        generator = ddata.compute_generator()
        assert(np.all(generator == self.generator_at_lag_1/2.))


class TestTransferOperator(object):
    flat_data = np.ones((10, 2))
    flat_data[:6, 0] = 2.
    flat_data[6:, 0] = 4.
    traj_edges = [0, 6, 10]
    transop_at_lag_1 = np.array([[8.5, 2.75], [2.75, 1]])
    transop_at_lag_3 = np.array([[7., 2.5], [2.5, 1.]])

    def test_basic_functionality(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        transop = ddata.compute_transop()
        assert(np.all(transop == self.transop_at_lag_1))

    def test_nondefault_input_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        transop_lag_specified = ddata.compute_transop(lag=3)
        assert(np.all(transop_lag_specified == self.transop_at_lag_3))

    def test_nondefault_dataset_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=3)
        transop_lag_from_dset = ddata.compute_transop()
        assert(np.all(transop_lag_from_dset == self.transop_at_lag_3))
