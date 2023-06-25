#!/usr/bin/env python3

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='mlcircuit',
      version='0.0.1',
      description='An end-to-end ML pipeline package.',
      author='Charbel Bou Maroun',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['mlcircuit', 'mlcircuit.framework'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: MIT License"
      ],
      install_requires=['numpy', 'requests', 'tqdm', 'networkx'],
      python_requires='>=3.8',
      extras_require={
          'testing': [
              "torch",
              "tensorflow",
              "pytest",
              "pytest-xdist",
              "onnx"
          ],
      },
      include_package_data=True)
