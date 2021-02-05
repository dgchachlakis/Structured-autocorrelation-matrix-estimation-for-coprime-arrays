import numpy as np
def snapshots(response_matrix, source_powers, noise_power, number_of_snapshots):
    L , K = response_matrix.shape   
    Y = np.zeros((L, number_of_snapshots)) + 1j * np.zeros((L, number_of_snapshots))
    symbols = np.diag(np.sqrt(source_powers)) @ (np.random.randn(K, number_of_snapshots) + 1j * np.random.randn(K, number_of_snapshots)) / np.sqrt(2)
    awgn = np.sqrt(noise_power) * (np.random.randn(L, number_of_snapshots) +  1j * np.random.randn(L, number_of_snapshots)) / np.sqrt(2)
    Y = response_matrix @ symbols + awgn
    return Y