import numpy as np
def spatial_smoothing(smoothing_matrix, autocorrelations):
    return smoothing_matrix @ np.kron (np.eye(smoothing_matrix.shape[0]), autocorrelations[:, None])