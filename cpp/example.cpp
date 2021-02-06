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

int main()
{

    int M, N, L, Lp, number_of_sources, permitted_iterations, number_of_realizations;
    double carrier_frequency, propagation_speed, noise_power;
    xarray<double> p, pdist, Esel, Eavg, thetas, F, source_powers, number_of_snapshots;
    xarray<complex<double>> S, received_snapshots, R, Rest, r, rest, Z, Zsel, Zavg, Zin, Zstructured;
    map<int, vector<int>> Jsets;
    // Coprime array
    M = 2;
    N = 3;
    L = array_length(M, N);
    Lp = coarray_length(M, N);
    carrier_frequency = (double)3 / 2 * pow(10, 8);
    propagation_speed = 3 * pow(10, 8);
    // Element locations of the physical array
    p = ca_element_locations(M, N, carrier_frequency, propagation_speed);
    // Set of differences of element-locations
    pdist = pair_wise_distances(p);
    // Set of indices corresponding to each 'element' of the coarray
    Jsets = form_index_sets(M, N, pdist, carrier_frequency, propagation_speed);
    // Selection and averaging sampling matrices
    Esel = selection_sampling(Jsets, L, Lp);
    Eavg = averaging_sampling(Jsets, L, Lp);
    // DoA sources
    number_of_sources = 3;
    thetas = {M_PI_4, M_PI_4 / 2, M_PI_4 / 3};
    // Array response matrix
    S = response_matrix(thetas, carrier_frequency, propagation_speed, p);
    // Smoothing matrix
    F = smoothing_matrix(Lp);
    // Powers
    noise_power = 1;
    source_powers = {1, 1, 1};
    // Nominal autocorrelation matrix (Physical array)
    R = autocorrelation_matrix(S, source_powers, noise_power);
    r = ravel(R);
    // Nominal autocorrelation matrix (coarray)
    Z = spatial_smoothing(F, linalg::dot(transpose(Esel), r));
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

    for (int j = 0; j < number_of_snapshots.size(); j++)
    {
        int Q = number_of_snapshots(j);
        cout << "Sample support:" << Q << endl;
        cout << "\t MSE (selection):\t\t" << real(mean(row(err_sel, j))) << endl;
        cout << "\t MSE (averaging):\t\t" << real(mean(row(err_avg, j))) << endl;
        cout << "\t MSE (structured-proposed):\t" << real(mean(row(err_structured, j))) << endl;
    }
    return 1;
}