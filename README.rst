========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
        
..    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/pyedgar/badge/?style=flat
    :target: https://pyedgar.readthedocs.io
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/ehthiede/PyEDGAR.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/ehthiede/PyEDGAR

.. |codecov| image:: https://codecov.io/gh/ehthiede/PyEDGAR/branch/master/graph/badge.svg
    :alt: Coverage Status
    :target: https://codecov.io/gh/ehthiede/PyEDGAR

.. |version| image:: https://img.shields.io/pypi/v/pyedgar.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/pyedgar

.. |commits-since| image:: https://img.shields.io/github/commits-since/ehthiede/PyEDGAR/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ehthiede/PyEDGAR/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/pyedgar.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/pyedgar

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pyedgar.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/pyedgar

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pyedgar.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/pyedgar


.. end-badges

Package for performing dynamical Galerkin expansion on trajectory data.  Currently in pre-alpha.

* Free software: MIT license

Installation
============

To install the code, download the directory from github, navigate into the folder, and run

::

    pip install -e .

We are currently working on getting the package onto pip.

Documentation
=============

https://PyEDGAR.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
