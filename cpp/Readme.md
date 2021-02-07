
## Structured autocorrelation matrix estimation for coprime arrays

In this repo we implement (in C++) an algorithmic frawework designed to compute an autocorrelation matrix estimate for the coarray which satisfies structure-properties of the true autocorrelation matrix [[1]](https://doi.org/10.1016/j.sigpro.2021.107987) --i.e., it is (i) Positive-Definite, (ii) Hermitian, (iii) Toeplitz, and (iv) has equal noise-subspace eigenvalues. 

---

Science Direct: https://doi.org/10.1016/j.sigpro.2021.107987

---

**Example**

First, we load the required libraries.
```cpp
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xstrided_view.hpp>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include "utils.hpp"
#include "proposed_algorithm.hpp"

using namespace std;
using namespace xt;
```
We consider coprime array with coprime natureals M, N.
```cpp
 // Coprime array
    M = 2;
    N = 3;
    L = array_length(M, N);
    Lp = coarray_length(M, N);
    carrier_frequency = (double)3 / 2 * pow(10, 8);
    propagation_speed = 3 * pow(10, 8);
    // Element locations of the physical array
    p = ca_element_locations(M, N, carrier_frequency, propagation_speed);
```
Then, we form the standard selection and averaging sampling matrices. 
```cpp
    // Set of differences of element-locations
    pdist = pair_wise_distances(p);
    // Set of indices corresponding to each 'element' of the coarray
    Jsets = form_index_sets(M, N, pdist, carrier_frequency, propagation_speed);
    // Set of indices corresponding to each 'element' of the coarray
    Jsets = form_index_sets(M, N, pdist, carrier_frequency, propagation_speed);
    // Selection and averaging sampling matrices
    Esel = selection_sampling(Jsets, L, Lp);
    Eavg = averaging_sampling(Jsets, L, Lp);
```
We consider a set of DoA sources with some powers (linear scale). Noise power (linear scale).
```cpp
    // DoA sources
    number_of_sources = 3;
    thetas = {M_PI_4, M_PI_4 / 2, M_PI_4 / 3};
    // Powers
    noise_power = 1;
    source_powers = {1, 1, 1};
```
We form the nominal (true) autocorrelation matrix of the coarray.
```cpp
    // Array response matrix
    S = response_matrix(thetas, carrier_frequency, propagation_speed, p);
    // Smoothing matrix
    F = smoothing_matrix(Lp);
    // Nominal autocorrelation matrix (Physical array)
    R = autocorrelation_matrix(S, source_powers, noise_power);
    r = ravel(R);
    // Nominal autocorrelation matrix (coarray)
    Z = spatial_smoothing(F, linalg::dot(transpose(Esel), r));
```
We consider multiple statistically independent realizations of noise and compute (empirically) the Mean-Squared-Estimation (MSE) error of each method while the sample-support varies. 
```cpp
// Received-snapshots and autocorrelation matrix estimation
    number_of_snapshots = {1, 10, 100, 1000};
    number_of_realizations = 500;
    int le = number_of_snapshots.size();
    xarray<double> err_sel = zeros<double>({le, number_of_realizations});
    xarray<double> err_avg = zeros<double>({le, number_of_realizations});
    xarray<double> err_structured = zeros<double>({le, number_of_realizations});
    for (int j = 0; j < number_of_snapshots.size(); j++)
    {
        int Q = number_of_snapshots(j);
        for (int i = 0; i < number_of_realizations; i++)
        {
            received_snapshots = snapshots(S, source_powers, noise_power, Q);
            Rest = autocorrelation_matrix_est(received_snapshots);
            rest = ravel(Rest);
            // Standard estimates
            Zsel = spatial_smoothing(F, linalg::dot(transpose(Esel), rest));
            Zavg = spatial_smoothing(F, linalg::dot(transpose(Eavg), rest));
            // Proposed estimate
            permitted_iterations = 100;
            Zin = sqrtm(linalg::dot(Zavg, transpose(conj(Zavg))));
            Zstructured = structured_estimate(Zin, Lp - number_of_sources, permitted_iterations, false);
            err_sel(j, i) = real(pow(linalg::norm(Zsel - Z), 2));
            err_avg(j, i) = real(pow(linalg::norm(Zavg - Z), 2));
            err_structured(j, i) = real(pow(linalg::norm(Zstructured - Z), 2));
        }
    }
```
Finally, we display the MSE attained by each method for every value of sample-support. 
```cpp
    for (int j = 0; j < number_of_snapshots.size(); j++)
    {
        int Q = number_of_snapshots(j);
        cout << "Sample support:" << Q << endl;
        cout << "\t MSE (selection):\t\t" << real(mean(row(err_sel, j))) << endl;
        cout << "\t MSE (averaging):\t\t" << real(mean(row(err_avg, j))) << endl;
        cout << "\t MSE (structured-proposed):\t" << real(mean(row(err_structured, j))) << endl;
    }
```
For the above parameter configuration, the empirical MSE is computed as 

```cmd
Sample support:1
         MSE (selection):                1038.809281
         MSE (averaging):                602.146571
         MSE (structured-proposed):      505.758647
Sample support:10
         MSE (selection):                94.786362
         MSE (averaging):                50.54998 
         MSE (structured-proposed):      42.104254
Sample support:100
         MSE (selection):                10.481542
         MSE (averaging):                5.681443
         MSE (structured-proposed):      4.248675
Sample support:1000
         MSE (selection):                1.056646
         MSE (averaging):                0.57852 
         MSE (structured-proposed):      0.443505
```

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
