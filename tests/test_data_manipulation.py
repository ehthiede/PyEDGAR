# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from pyedgar import data_manipulation as manip
from pyedgar.data_manipulation import delay_embed


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


def test__as_flat(working_flat_and_tlist):
    flat, traj_edges, tlist = working_flat_and_tlist
    from_tlist, from_tlist_edges, from_tlist_type = manip._as_flat(tlist)
    assert(np.array_equal(from_tlist, flat))
    assert(np.array_equal(from_tlist_edges, traj_edges))
    assert(from_tlist_type == "list_of_trajs")

    from_flat, from_flat_edges, from_flat_type = manip._as_flat((flat, traj_edges))
    assert(np.array_equal(from_flat, flat))
    assert(np.array_equal(from_flat_edges, traj_edges))
    assert(from_flat_type == "flat")


@pytest.mark.parametrize('input_type', ['list_of_trajs', 'flat'])
def test__flat_to_orig(working_flat_and_tlist, input_type):
    flat, traj_edges, tlist = working_flat_and_tlist
    rebuilt_data = manip._flat_to_orig(flat, traj_edges, input_type)
    if input_type is 'list_of_trajs':
        for rebuilt_i, tlist_i in zip(rebuilt_data, tlist):
            assert(np.array_equal(rebuilt_i, tlist_i))
    elif input_type is 'flat':
        new_flat, new_edges = rebuilt_data
        assert(np.array_equal(new_flat, flat))
        assert(np.array_equal(traj_edges, new_edges))


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


class TestDatasetSplit(object):
    flat_data = np.ones((20, 2))
    traj_edges = [0, 10, 13, 20]
    init_indices_lag_2 = np.array(list(range(0, 8)) + [10] + list(range(13, 18)))
    final_indices_lag_2 = np.array(list(range(2, 10)) + [12]
                                   + list(range(15, 20)))
    init_indices_lag_4 = np.array(list(range(0, 6)) + list(range(13, 16)))
    final_indices_lag_4 = np.array(list(range(4, 10)) + list(range(17, 20)))

    def test_use_custom_lag(self):
        init_indices, final_indices = manip.get_initial_final_split(self.traj_edges, lag=2)
        assert(np.all(init_indices == self.init_indices_lag_2))
        assert(np.all(final_indices == self.final_indices_lag_2))

    def test_lag_longer_than_traj(self):
        init_indices, final_indices = manip.get_initial_final_split(self.traj_edges, lag=4)
        assert(np.all(init_indices == self.init_indices_lag_4))
        assert(np.all(final_indices == self.final_indices_lag_4))
