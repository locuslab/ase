import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef optimize_T_sparse(np.ndarray[DTYPE_t, ndim=1] T_data,
                        np.ndarray[int, ndim=1] T_indices,
                        np.ndarray[int, ndim=1] T_indprt,
                        np.ndarray[DTYPE_t, ndim=1] T_support,
                        np.ndarray[DTYPE_t, ndim=1] V,
                        np.ndarray[DTYPE_t, ndim=1] eps_T,
                        int n):
    cdef long i
    cdef long j
    cdef long k
    cdef long m
    cdef long p
    cdef long q
    cdef long best_ind
    cdef float budget
    cdef np.ndarray[int, ndim=1] T_ind
    cdef np.ndarray[long, ndim=1] sorted_T_ind
    cdef float transfer
    
    for i in range(n):
        p = T_indprt[i]
        q = T_indprt[i + 1]
        T_ind = T_indices[p:q]
        sorted_T_ind = np.argsort(V[T_ind])
        m = sorted_T_ind.shape[0]
        
        # Pick the best index over the support assuming that T = 0 when T_support = 0
        for k in range(m):
            best_ind = sorted_T_ind[m - k - 1L]
            if T_support[p + best_ind]:
                break
        
        budget = eps_T[i] / 2.0
        j = 0
        while budget > 1e-12 and j < m - k:
            transfer = min(budget, T_data[p + sorted_T_ind[j]])
            T_data[p + best_ind] += transfer
            T_data[p + sorted_T_ind[j]] -= transfer
            budget -= transfer
            j += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[float, ndim=1] next_values_sparse(np.ndarray[DTYPE_t, ndim=1] T_data,
                                                   np.ndarray[int, ndim=1] T_indices,
                                                   np.ndarray[int, ndim=1] T_indprt,
                                                   np.ndarray[DTYPE_t, ndim=1] V,
                                                   int n):
    
    cdef long i
    cdef long j
    cdef long p
    cdef long q
    cdef np.ndarray[int, ndim=1] T_ind
    
    cdef np.ndarray[float, ndim=1] next_values = np.zeros(n, dtype=DTYPE)
    
    for i in range(n):
        p = T_indprt[i]
        q = T_indprt[i + 1]
        for j in range(p, q):
            next_values[i] += T_data[j] * V[T_indices[j]]
    
    return next_values
