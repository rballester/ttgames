#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

setup(
    name='ttgames',
    version='0.1.0',
    description="Tensor Approximation of Cooperative Games and Their Semivalues",
    long_description="ttgames is a Python package that uses the tensor train decomposition format to compute semivalues (like the Shapley or Banzhaf values) out of a cooperative game.",
    url='https://github.com/rballester/ttgames',
    author="Rafael Ballester-Ripoll",
    author_email='rafael.ballester@ie.edu',
    packages=[
        'ttgames',
    ],
    include_package_data=True,
    install_requires=[
        'tntorch',
        'torch',
        'numpy',
    ],
    license="BSD",
    zip_safe=False,
    keywords='ttgames',
    classifiers=[
        'License :: OSI Approved',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require='pytest'
)
