import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from utils import *
from algorithm import *
# channel
carrier_frequency = 1.5 * 10 ** 8
propagation_speed = 3 * 10 ** 8
wavelength = propagation_speed / carrier_frequency
unit_spacing = wavelength / 2
channel = (carrier_frequency, propagation_speed)
# Coprime array with coprimes M, N such that M < N
M = 2
N = 3
p = ca_element_locations(M, N, channel)
# DoA sources
thetas = np.array([-np.pi / 3, -np.pi / 4, np.pi / 5])
# source and noise powers
source_powers = np.array([1, 1/2, 1])
noise_power = 1
# Array response matrix
S = response_matrix(thetas, p, channel)
# Nominal Physical autocrrelation matrix
R = autocorrelation_matrix(S, source_powers, noise_power)
# autocorrelation sampling matrices
Jdict = form_index_sets(M, N, pair_wise_distances(p), channel)
Esel = selection_sampling(Jdict, array_length(M, N), coarray_length(M, N))
Eavg = averaging_sampling(Jdict, array_length(M, N), coarray_length(M, N))
# Nominal coarray autocorrelation matrix
F = smoothing_matrix(coarray_length(M, N))
Z = spatial_smoothing(F, Esel.T @ R.flatten('F'))
# Sample support axis and number of realizations
number_of_snapshots_axis = [10, 100, 1000, 10000]
number_of_realizations = 150
# Zero - padding
err_sel = np.zeros((len(number_of_snapshots_axis), number_of_realizations))
err_avg = np.zeros((len(number_of_snapshots_axis), number_of_realizations))
err_structured = np.zeros(
    (len(number_of_snapshots_axis), number_of_realizations))
for i, Q in enumerate(number_of_snapshots_axis):
    for j in range(number_of_realizations):
        Y = snapshots(S, source_powers, noise_power, Q)
        Rest = autocorrelation_matrix_est(Y)
        r = Rest.flatten('F')
        Zsel = spatial_smoothing(F, Esel.T @ r)
        Zavg = spatial_smoothing(F, Eavg.T @ r)
        Zin = sqrtm(Zavg @ np.conj(Zavg.T))
        Zstructured = structured(Zin, coarray_length(M, N)-thetas.shape[0])
        err_sel[i, j] = np.linalg.norm(Z - Zsel, 'fro') ** 2
        err_avg[i, j] = np.linalg.norm(Z - Zavg, 'fro') ** 2
        err_structured[i, j] = np.linalg.norm(Z - Zstructured, 'fro') ** 2
# Compute the sample-average MSE of each method
err_sel = np.mean(err_sel, axis=1)
err_avg = np.mean(err_avg, axis=1)
err_structured = np.mean(err_structured, axis=1)
# Plot and compare MSEs
plt.figure()
plt.loglog(number_of_snapshots_axis, err_sel, '+-r', label="Selection")
plt.loglog(number_of_snapshots_axis, err_avg, '^-b', label="Averaging")
plt.loglog(number_of_snapshots_axis, err_structured,
           'x-k', label="Structured (proposed)")
plt.legend()
plt.grid(color='k', linestyle=':', linewidth=1)
plt.ylabel('MSE')
plt.xlabel('Sample support')
plt.show()