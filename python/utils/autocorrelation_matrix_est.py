import numpy as np
def autocorrelation_matrix_est(snapshots):
    return snapshots @ np.conj(snapshots).T / snapshots.shape[1]