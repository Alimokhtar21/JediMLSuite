#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import JediML

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

long_description = open('README.rst').read()

requirements_lines = [line.strip() for line in open('requirements.txt').readlines()]
install_requires = list(filter(None, requirements_lines))

setup(
    name='JediML',
    version=JediML.__version__,
    description='Demo machine learning algorithm suite',
    long_description=long_description,
    keywords='Machine Learning, Perceptron Learning Algorithm, PLA, Pocket Perceptron Learning Algorithm, Pocket PLA, Linear Regression, Logistic Regression, Ridge Regression, Binay Classifier, Multi Classifier, SVM, Support Vector Machine, Decision Tree, AdaBoost',
    author='Hsuan-Tien Lin, Jacqueline Pang',
    author_email='jp2338@cornell.edu',
    url='https://github.com/pangjac/JediMLSuite',
    license='MIT',
    install_requires=install_requires,
    packages=['JediML'],
    package_dir={'JediML': 'JediML'},
    package_data={'JediML': ['*.*', 'dataset/*.*', 'dataset/digits/*.*', 'dataset/digits/traning_digits/*.*', 'dataset/digits/test_digits/*.*']},
    test_suite='test_Jedi_ml',
    zip_safe=False,
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
)
