# -*- coding: utf-8 -*-
# Basic package tests


def test_pyedgar():
    # Check that packages we want are importable.
    import pyedgar
    assert pyedgar
    assert pyedgar.data_manipulation
    assert pyedgar.basis
    assert pyedgar.galerkin
#    assert pyedgar.DynamicalDataset
