import unittest
import scipy.sparse as sp

from constants import *

from dynamic_programming import value_iteration
import value_iteration_cy


class TestSparseVI(unittest.TestCase):
    def test_optimize_T(self):
        num_states = 1000
        num_actions = 10
        
        transition_matrix = construct_transition_matrix(num_states, num_actions, num_support=200)
        
        values = np.random.uniform(-1, 1, num_states).astype(dtype=DTYPE)
        eps_T = np.random.uniform(0.01, 2, num_states * num_actions).astype(dtype=DTYPE)
        
        T = transition_matrix.copy()
        value_iteration_cy.optimize_T_sparse(T.data, T.indices, T.indptr, -values, eps_T, num_states * num_actions)
        
        assert float_equal(np.asarray(T.sum(axis=1)), 1).all()

        T2 = transition_matrix.copy()
        value_iteration.optimize_T_fast_4(T2.data, T2.indices, T2.indptr, -values, eps_T, num_states * num_actions)
        
        assert float_equal(T.toarray(), T2.toarray()).all()

    def test_next_values(self):
        num_states = 1000
        num_actions = 10
        
        T = construct_transition_matrix(num_states, num_actions, num_support=200)
    
        values = np.random.uniform(-1, 1, num_states).astype(dtype=DTYPE)
        
        next_values = value_iteration_cy.next_values_sparse(T.data, T.indices, T.indptr, values, num_states * num_actions)
        
        next_values_2 = np.sum(T.toarray() * values[np.newaxis, :], axis=1)

        assert float_equal(next_values, next_values_2).all()


def construct_transition_matrix(num_states, num_actions, num_support):
    transition_matrix = sp.random(num_states * num_actions, num_states, num_support / num_states, format='csr', dtype=DTYPE)
    D = sp.diags(1. / np.asarray(transition_matrix.sum(1)).T[0])
    transition_matrix = D @ transition_matrix
    
    assert (transition_matrix.data >= 0).all()
    assert float_equal(np.asarray(transition_matrix.sum(axis=1)), 1).all()
    
    return transition_matrix


def float_equal(a, b):
    return np.abs(a-b) <= FLOAT_TOLERANCE


if __name__ == '__main__':
    unittest.main()
