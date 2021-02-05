from .coarray_length import coarray_length
def form_index_sets(M, N , pdist, channel):
    (carrier_frequency, propagation_speed) = channel
    wavelength = propagation_speed / carrier_frequency
    unit_spacing = wavelength / 2
    co_length = coarray_length(M, N)
    idx_dict = dict()
    for i, n in enumerate(pdist):
        nn = n / unit_spacing
        if abs(nn) < co_length:
            try:
                idx_dict[nn].add(i)
            except KeyError:
                idx_dict[nn] = {i}
    return idx_dict
        