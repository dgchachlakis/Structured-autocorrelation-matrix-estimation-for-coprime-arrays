import numpy as np
def pair_wise_distances(vector):
    N = vector.shape[0]
    o = np.ones(N, )
    v1 = np.outer(vector, o)
    v2 = np.outer(o, vector)
    return (v1-v2).flatten()    