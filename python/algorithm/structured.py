import numpy as np


def structured(matrix, d, tol=1e-6):
    met_prev = np.linalg.norm(matrix) ** 2
    while True:
        matrix = nearest_toeplitz(matrix)
        matrix = nearest_psd(matrix)
        matrix = eigenvalues_correction(matrix, d)
        met = np.linalg.norm(matrix) ** 2
        if met_prev - met < tol:
            break
        met_prev = met
    return matrix


def nearest_psd(matrix):
    u, s, vt = np.linalg.svd(matrix)
    s[s < 0] = 0
    return u @ np.diag(s) @ vt


def nearest_toeplitz(matrix):
    Lp = matrix.shape[0]
    for i in range(-Lp+1, Lp, 1):
        temp1 = np.eye(Lp, k=i) * np.mean(np.diagonal(matrix, i))
        temp2 = np.eye(Lp, k=i) * matrix
        matrix += temp1 - temp2
    return matrix


def eigenvalues_correction(matrix, d):
    u, s, vt = np.linalg.svd(matrix)
    ss = np.sort(s)
    idx = np.argsort(s)
    u = u[:, idx]
    vt = vt[idx, :]
    cor = np.mean(ss[:d])
    ss[:d] = cor
    return u @ np.diag(ss) @ vt