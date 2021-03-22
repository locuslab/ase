from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize('dynamic_programming/value_iteration_cy.pyx'),
    include_dirs=[numpy.get_include()]
)
