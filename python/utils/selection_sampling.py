import numpy as np
def selection_sampling(Jdict, array_length, coarray_length):
    
    I = np.eye(array_length ** 2)
    E = np.zeros((array_length ** 2, 2 * coarray_length - 1))
    for n in range(2 * coarray_length - 1):
        J = Jdict[1 - coarray_length + n]
        for e in J:
            break
        E[:, n] = I[:, e]
    return E