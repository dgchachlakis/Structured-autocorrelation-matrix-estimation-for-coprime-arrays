import numpy as np
def smoothing_matrix(coarray_length):
    m = 0
    B1 = np.zeros((coarray_length, coarray_length - m - 1))
    B2 = np.eye(coarray_length)
    B3 = np.zeros((coarray_length, m))
    F = np.block([[B1, B2, B3]])
    for m in range(1, coarray_length):
        B1 = np.zeros((coarray_length, coarray_length - m - 1))
        B2 = np.eye(coarray_length)
        B3 = np.zeros((coarray_length, m))
        B = np.block([[B1, B2, B3]])
        F = np.concatenate((F, B), axis = 1)
    return F