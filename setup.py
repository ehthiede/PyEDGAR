#!/usr/bin/env python

from setuptools import setup,find_packages

setup(name='pyEDGAR',
    version='0.0.1',
    description="Code for the construction of dynamical galerkin approximation schemes.",
    license='GPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering'
        ],
#    packages=find_packages(),
    packages=['pyedgar'],
    install_requires=['numpy','scipy'],

    )
