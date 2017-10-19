"""The PyEDGAR package is a collection of scripts and tools for constructing
Galerkin approximations on trajectory data."""
from __future__ import absolute_import

# Set explicit packages
from . import data_manipulation, basis, galerkin
from .dataset import DynamicalDataset

__author__ = "Erik Henning Thiede"
__license__ = "GPL"
__maintainer__ = "Erik Henning Thiede"
__email__ = "thiede@uchicago.edu"
__version__ = "0.1.0"
__all__ = ['galerkin', 'basis',
           'data_manipulation', 'DynamicalDataset']
