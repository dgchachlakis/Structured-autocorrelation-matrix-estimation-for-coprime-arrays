
## Structured autocorrelation matrix estimation for coprime arrays

In this repo we implement (in C++ and Python) an algorithmic frawework designed to compute an autocorrelation matrix estimate for the coarray which satisfies structure-properties of the true autocorrelation matrix [[1]](https://doi.org/10.1016/j.sigpro.2021.107987) --i.e., it is (i) Positive-Definite, (ii) Hermitian, (iii) Toeplitz, and (iv) has equal noise-subspace eigenvalues. 

---

Science Direct: https://doi.org/10.1016/j.sigpro.2021.107987

---

**Example**

For usage examples, please open the cpp and python directories for C++ and Python, respectively. 


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
