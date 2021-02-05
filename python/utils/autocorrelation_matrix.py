import numpy as np
def autocorrelation_matrix(response_matrix, source_powers, noise_power):
    return response_matrix @ np.diag(source_powers) @ np.conj(response_matrix).T + noise_power * np.eye(response_matrix.shape[0])