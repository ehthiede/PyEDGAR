# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pyedgar import data_manipulation as manip


def test_tlist_to_flat(working_flat_and_tlist):
    flat, traj_edges, tlist = working_flat_and_tlist
    test_flat, test_traj_edges = manip.tlist_to_flat(tlist)
    flat_diff = test_flat - flat
    te_diff = np.array(test_traj_edges) - np.array(traj_edges)
    assert(np.all(test_flit == flat))
    assert(list(test_traj_edges) == traj_edges)

def test_flat_to_tlist(working_flat_and_tlist):
    flat, traj_edges, tlist = working_flat_and_tlist
    test_tlist = manip.flat_to_tlist(flat, traj_edges)
    for test_traj, traj in zip(test_tlist, tlist):
        assert(np.all(test_traj == traj))

class TestDelayEmbedding():
    test_data = np.empty((30, 2)) 
    test_data[:, 0] = np.arange(30)
    test_data[:, 1] = -np.arange(30)
    traj_edges = [0, 10, 25, 30]
    pt_1 = test_data[:10]
    pt_2 = test_data[10:25]
    pt_3 = test_data[25:]
    self.n_embed = 2
    self.lag = 3
    pt_1_embedded = np.concatenate((pt_1[6:], pt_1[3:-3], pt_1[:-6]), axis=1)
    pt_2_embedded = np.concatenate((pt_2[6:], pt_2[3:-3], pt_2[:-6]), axis=1)

    def test_delay_embed_on_tlist(self):
        tlist = [self.pt_1,self.pt_2,self.pt_3]
        embedded_tlist = manip.delay_embed(tlist, self.n_embed, lag=self.lag) 
        assert(type(embedded_tlist) is list)
        assert(len(embedded_tlist) == 2)
        assert(np.all(embedded_tlist[0] == self.pt_1))
        assert(np.all(embedded_tlist[1] == self.pt_2))

    def test_delay_embed_on_flat(self):
        true_embed_traj_flat = np.concatenate((self.pt_1_embedded,
                                               self.pt_2_embedded), axis=0)
        true_embed_traj_edges = [0, 4, 13] 
        embed_traj, embed_edges = manip.delay_embed((test_data, traj_edges), 
                                                    self.n_embed, lag=self.lag)
        assert(np.all(embed_traj == true_embed_traj_flat))
        assert(list(embed_edges) == true_embed_traj_edges)
        
