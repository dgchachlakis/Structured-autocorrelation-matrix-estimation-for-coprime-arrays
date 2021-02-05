import numpy as np
def ca_element_locations(M, N, channel):
    (carrier_frequency, propagation_speed) = channel
    wavelength = propagation_speed / carrier_frequency
    unit_spacing = wavelength / 2
    L = 2 * M + N - 1
    p = np.zeros(L, )
    for i in range(N):
        p[i] = i * M * unit_spacing
    for i in range(2 * M - 1):
        p[N+i] = (i + 1) * N * unit_spacing
    p.sort()
    return p
