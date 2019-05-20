from setuptools import find_packages
from distutils.core import setup, Extension
import os
import numpy as np
numpy_include_dir = os.path.join(os.path.dirname(np.__file__), 'core', 'include', 'numpy')

setup(name='py_TORCS',
      version='1.0',
      description='RL environment TORCS wrapped in Python',
      author='Xiangyu Chen',
      author_email='chenxiangyu0339@gmail.com',
      url='https://github.com/cxy1997/pyTORCS/',
      license='',
      py_modules=['py_TORCS.py_TORCS'],
      ext_modules=[Extension('py_TORCS/TORCS_ctrl', ['py_TORCS/TORCS_ctrl.c'],
      install_requires=[
          'numpy',
      ],
      packages=['game_config'],
      package_dir={'game_config': 'py_TORCS/game_config'},
      include_package_data=True,
      package_data={'game_config': ['*.dat']},
      include_dirs=[numpy_include_dir])],
    )
