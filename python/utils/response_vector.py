import numpy as np
def response_vector(theta, element_locations, carrier_frequency, propagation_speed):
    const = 2 * np.pi * carrier_frequency / propagation_speed
    return np.exp( - 1j * const * np.sin(theta) * element_locations)