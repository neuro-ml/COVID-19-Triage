#!/usr/bin/env python

from setuptools import setup, find_packages

from covid_triage import __version__


with open('README.md', encoding='utf-8') as file:
    readme = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.readlines()

setup(
    name='COVID-19-Triage',
    packages=find_packages(include=('covid_triage',)),
    version=__version__,
    description='Pytorch library for CT-based COVID-19 Triage: Deep Multitask Learning Improves Joint Identification and Severity Quantification paper',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/neuro-ml/COVID-19-Triage',
    keywords=['COVID-19', 'triage', 'convolutional neural network', 'chest CT'],
    author='Mikhail Goncharov, Maxim Pisov, Boris Shirokikh & Alexey Shevtsov.',
    install_requires=requirements,
    python_requires='>=3.6',
)
