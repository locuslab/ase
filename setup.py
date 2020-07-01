from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('value_iteration_cy.pyx')
)
