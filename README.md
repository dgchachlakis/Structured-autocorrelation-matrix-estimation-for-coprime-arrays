
## Structured autocorrelation matrix estimation for coprime arrays

In this repo we implement (in Python) an algorithmic frawework designed to compute an autocorrelation matrix estimate for the coarray which satisfies structure-properties of the true autocorrelation matrix [[1]](https://doi.org/10.1016/j.sigpro.2021.107987) --i.e., it is (i) Positive-Definite, (ii) Hermitian, (iii) Toeplitz, and (iv) has equal noise-subspace eigenvalues. 

---

Science Direct: https://doi.org/10.1016/j.sigpro.2021.107987

---

**Example**

We form array with coprime naturals M, N. 
```Python
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
```



We consider DoA sources, awgn, and form the true autocorrelation matrix of the physical array.
```python
    # DoA sources
    thetas = np.array([-np.pi / 3, -np.pi / 4, np.pi / 5])
    # source and noise powers
    source_powers = np.array([1, 1/2, 1])
    noise_power = 1
    # Array response matrix
    S = response_matrix(thetas, p, channel)
    # Nominal Physical autocrrelation matrix
    R = autocorrelation_matrix(S, source_powers, noise_power)
```

Then, we form the standard selection and averaging sampling matrices. We compute the true autocorrelation matrix of the coarray. 
```python
    # autocorrelation sampling matrices
    Jdict = form_index_sets(M, N, pair_wise_distances(p), channel)
    Esel = selection_sampling(Jdict, array_length(M, N), coarray_length(M, N))
    Eavg = averaging_sampling(Jdict, array_length(M, N), coarray_length(M, N))
    # Nominal coarray autocorrelation matrix
    F = smoothing_matrix(coarray_length(M, N))
    Z = spatial_smoothing(F, Esel.T @ R.flatten('F'))
```


We compute (empirically) the MSE attained by each method.
```python
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
```

We plot the computed MSEs. 
```python
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
    plt.title('Deterministic DoAs')
    plt.show()
```

For the above parameter configuration, we plot the illustrate the results in the following Figure:
![](mse.png)

---
**Questions/issues**
Inquiries regarding the scripts provided below are cordially welcome. In case you spot a bug, please let me know. 

---
**Citing**

If you use our algorithms, please cite [[1]](https://doi.org/10.1016/j.sigpro.2021.107987).

```bibtex
@article{chachlakis2021structured,
  title={Structured Autocorrelation Matrix Estimation for Coprime Arrays},
  author={Chachlakis, Dimitris G and Markopoulos, Panos P},
  journal={Signal Processing},
  pages={107987},
  year={2021},
  publisher={Elsevier}
}
```
[[1]](https://doi.org/10.1016/j.sigpro.2021.107987) D. G. Chachlakis and P. P. Markopoulos, ``Structured Autocorrelation Matrix Estimation for Coprime Arrays," in Signal Processing, p. 107987, 2021.

---

**Related works**

The following works might be of interest:

* [[2]](https://ieeexplore.ieee.org/document/8313121) D. G. Chachlakis, P. P. Markopoulos and F. Ahmad, "The Mean-Squared-Error of autocorrelation sampling in coprime arrays," 2017 IEEE 7th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP), Curacao, 2017, pp. 1-5, doi: 10.1109/CAMSAP.2017.8313121.
* [[3]](https://ieeexplore.ieee.org/document/8461676) D. G. Chachlakis, P. P. Markopoulos and F. Ahmad, "Mmse-Based Autocorrelation Sampling for Comprime Arrays," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Calgary, AB, 2018, pp. 3474-3478, doi: 10.1109/ICASSP.2018.8461676.
