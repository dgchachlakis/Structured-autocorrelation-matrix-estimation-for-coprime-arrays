import numpy as np
import utils.response_vector as response_vector
def response_matrix(thetas, element_locations, channel):
    (carrier_frequency, propagation_speed) = channel
    wavelength = propagation_speed / carrier_frequency
    unit_spacing = wavelength / 2
    L = element_locations.shape[0]
    K = thetas.shape[0]
    S = np.zeros((L, K)) + 1j * np.zeros((L, K))
    for i, theta in enumerate(thetas):
        S[:, i] = response_vector(theta, element_locations, carrier_frequency, propagation_speed)
    return S