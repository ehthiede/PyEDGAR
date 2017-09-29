# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from pyedgar import DynamicalDataset
from pyedgar.dataset import delay_embed

lags = list(range(1, 4))
timesteps = [1., 0.5, 2, 1E4]


class TestDatasetCreation(object):
    @pytest.mark.parametrize('lag', lags)
    @pytest.mark.parametrize('timestep', timesteps)
    def test_from_flat(self, working_flat_and_tlist, lag, timestep):
        flat, traj_edges, tlist = working_flat_and_tlist
        DS_from_flat = DynamicalDataset((flat, traj_edges), lag=lag, timestep=timestep)
        new_flat_traj, new_traj_edges = DS_from_flat.get_flat_data()
        assert(np.all(new_traj_edges == traj_edges))
        assert(np.all(new_flat_traj == flat))
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
        DS_from_tlist = DynamicalDataset(tlist)
        new_flat_traj, new_traj_edges = DS_from_tlist.get_flat_data()
        assert(np.all(new_traj_edges == traj_edges))
        assert(np.all(new_flat_traj == flat))
        assert(np.all(DS_from_tlist.lag == 1))
        assert(np.all(DS_from_tlist.timestep == 1.))

    def test_from_single_traj(self, working_flat_and_tlist):
        flat, traj_edges, tlist = working_flat_and_tlist
        traj = tlist[0]
        traj_edges = np.array([0, len(traj)])
        DS_from_single = DynamicalDataset(traj)
        new_flat_traj, new_traj_edges = DS_from_single.get_flat_data()
        assert(np.all(new_traj_edges == traj_edges))
        assert(np.all(new_flat_traj == traj))
        assert(np.all(DS_from_single.lag == 1))
        assert(np.all(DS_from_single.timestep == 1.))


class TestDatasetSplit(object):
    flat_data = np.ones((20, 2))
    traj_edges = [0, 10, 13, 20]
    init_indices_lag_2 = np.array(list(range(0, 8)) + [10] + list(range(13, 18)))
    final_indices_lag_2 = np.array(list(range(2, 10)) + [12]
                                   + list(range(15, 20)))
    init_indices_lag_4 = np.array(list(range(0, 6)) + list(range(13, 16)))
    final_indices_lag_4 = np.array(list(range(4, 10)) + list(range(17, 20)))

    def test_use_dataset_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=2)
        init_indices, final_indices = ddata.get_initial_final_split()
        assert(np.all(init_indices == self.init_indices_lag_2))
        assert(np.all(final_indices == self.final_indices_lag_2))

    def test_use_custom_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=1)
        init_indices, final_indices = ddata.get_initial_final_split(lag=2)
        assert(np.all(init_indices == self.init_indices_lag_2))
        assert(np.all(final_indices == self.final_indices_lag_2))

    def test_lag_longer_than_traj(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=4)
        init_indices, final_indices = ddata.get_initial_final_split()
        assert(np.all(init_indices == self.init_indices_lag_4))
        assert(np.all(final_indices == self.final_indices_lag_4))


class TestGenerator(object):
    flat_data = np.ones((11, 2))
    flat_data[:5, 0] = np.arange(1, 6)
    flat_data[5:, 0] = np.arange(5, 11)
    traj_edges = [0, 5, 11]
    true_gen_at_lag_1 = np.array([[5., 0.], [1., 0.]])
    true_gen_at_lag_4 = np.array([[4., 0.], [1., 0.]])

    def test_basic_functionality(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        generator = ddata.compute_generator()
        assert(np.all(generator == self.true_gen_at_lag_1))

    def test_nondefault_input_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        generator_lag_specified = ddata.compute_generator(lag=4)
        assert(np.all(generator_lag_specified == self.true_gen_at_lag_4))

    def test_nondefault_dataset_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=4)
        generator_lag_from_dset = ddata.compute_generator()
        assert(np.all(generator_lag_from_dset == self.true_gen_at_lag_4))

    def test_nondefault_timestep(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), timestep=2.)
        generator = ddata.compute_generator()
        assert(np.all(generator == self.true_gen_at_lag_1/2.))


class TestTransferOperator(object):
    flat_data = np.ones((10, 2))
    flat_data[:6, 0] = 2.
    flat_data[6:, 0] = 4.
    traj_edges = [0, 6, 10]
    true_t_op_at_lag_1 = np.array([[8.5, 2.75], [2.75, 1]])
    true_t_op_at_lag_3 = np.array([[7., 2.5], [2.5, 1.]])

    def test_basic_functionality(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        transop = ddata.compute_transop()
        assert(np.all(transop == self.true_t_op_at_lag_1))

    def test_nondefault_input_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges))
        transop_lag_specified = ddata.compute_transop(lag=3)
        assert(np.all(transop_lag_specified == self.true_t_op_at_lag_3))

    def test_nondefault_dataset_lag(self):
        ddata = DynamicalDataset((self.flat_data, self.traj_edges), lag=3)
        transop_lag_from_dset = ddata.compute_transop()
        assert(np.all(transop_lag_from_dset == self.true_t_op_at_lag_3))


class TestInitialInnerProduct(object):
    traj_edges = [0, 10, 28, 30]
    test_data_1 = np.ones((30, 2))
    test_data_1[:, 0] = np.arange(30)
    test_data_1[10:28, 1] = 0.5
    test_data_2 = np.ones(30)*2.
    test_data_2[10:28] = 4.
    test_data_2[28:] = -100.

    def test_initial_inner_product(self):
        true_ip = np.array([49., 2.])
        ddata_1 = DynamicalDataset((self.test_data_1, self.traj_edges), lag=2)
        ddata_2 = DynamicalDataset((self.test_data_2, self.traj_edges), lag=3)
        test_ip = ddata_1.initial_inner_product(ddata_2)
        assert(np.all(true_ip == test_ip))
        ddata_3 = DynamicalDataset((self.test_data_1, self.traj_edges), lag=1)
        test_ip = ddata_3.initial_inner_product(ddata_2, lag=2)
        assert(np.all(true_ip == test_ip))

    def test_mismatched_trajs(self):
        new_traj_edges = [0, 10, 26, 30]
        ddata_1 = DynamicalDataset((self.test_data_1, self.traj_edges), lag=2)
        ddata_2 = DynamicalDataset((self.test_data_2, new_traj_edges), lag=2)
        with pytest.raises(ValueError):
            ddata_1.initial_inner_product(ddata_2)


class TestDelayEmbedding():
    test_data = np.empty((30, 2))
    test_data[:, 0] = np.arange(30)
    test_data[:, 1] = -np.arange(30)
    traj_edges = [0, 10, 25, 30]
    pt_1 = test_data[:10]
    pt_2 = test_data[10:25]
    pt_3 = test_data[25:]
    n_embed = 2
    lag = 3
    pt_1_embedded = np.concatenate((pt_1[6:], pt_1[3:-3], pt_1[:-6]), axis=1)
    pt_2_embedded = np.concatenate((pt_2[6:], pt_2[3:-3], pt_2[:-6]), axis=1)

    def test_delay_embed_on_tlist(self):
        tlist = [self.pt_1, self.pt_2, self.pt_3]
        print(tlist)
        embedded_tlist = delay_embed(tlist, self.n_embed, lag=self.lag)
        assert(type(embedded_tlist) is list)
        assert(len(embedded_tlist) == 2)
        assert(np.all(embedded_tlist[0] == self.pt_1_embedded))
        assert(np.all(embedded_tlist[1] == self.pt_2_embedded))

    def test_delay_embed_on_flat(self):
        true_embed_traj_flat = np.concatenate((self.pt_1_embedded,
                                               self.pt_2_embedded), axis=0)
        true_embed_traj_edges = [0, 4, 13]
        flat_data = (self.test_data, self.traj_edges)
        embed_traj, embed_edges = delay_embed(flat_data, self.n_embed,
                                              lag=self.lag)
        assert(np.all(embed_traj == true_embed_traj_flat))
        assert(list(embed_edges) == true_embed_traj_edges)
