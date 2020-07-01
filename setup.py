from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('dynamic_programming/value_iteration_cy.pyx')
)
