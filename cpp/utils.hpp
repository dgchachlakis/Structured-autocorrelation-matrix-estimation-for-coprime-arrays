#include <iostream>
#include <complex>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xsort.hpp>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>

using namespace std;
using namespace xt;

int array_length(int M, int N)
{
    return (2 * M + N - 1);
}

int coarray_length(int M, int N)
{
    return (M * N + M);
}

map<int, vector<int>> form_index_sets(int M, int N, xarray<double> pdist, double carrier_frequency, double propagation_speed)
{
    double wavelength = propagation_speed / carrier_frequency;
    double unit_spacing = wavelength / 2;
    int co_length = coarray_length(M, N);
    map<int, vector<int>> dict;
    for (int cnt = 0; cnt < pdist.size(); cnt++)
    {
        int nn = (int)pdist(cnt) / unit_spacing;
        if (abs(nn) < co_length)
        {
            dict[nn].push_back(cnt);
        }
    }
    return dict;
}

xarray<double> ca_element_locations(int M, int N, double carrier_frequency, double propagation_speed)
{
    int L = 2 * M + N - 1;
    double wavelength = propagation_speed / carrier_frequency;
    double unit_spacing = wavelength / 2;
    xarray<double> p = zeros<double>({L, 1});
    for (int i = 0; i < N; i++)
    {
        p(i, 0) = i * M * unit_spacing;
    }
    for (int i = 0; i < 2 * M - 1; i++)
    {
        p(N + i, 0) = (i + 1) * N * unit_spacing;
    }
    xarray<double> pp = sort(p, 0);
    return pp;
}

xarray<double> pair_wise_distances(xarray<double> v)
{
    int N = v.size();
    xarray<double> o = ones<double>({N, 1});
    xarray<double> v1 = linalg::dot(v, transpose(o));
    xarray<double> v2 = linalg::dot(o, transpose(v));
    xarray<double> dif = v1 - v2;
    xarray<double> fl = flatten(dif);
    return transpose(fl);
}

xarray<double> selection_sampling(map<int, vector<int>> Jdict, int array_length, int coarray_length)
{
    map<int, vector<int>>::iterator it;

    int Ls = pow(array_length, 2);
    xarray<double> I = eye(Ls);
    xarray<double> E = zeros<double>({Ls, 2 * coarray_length - 1});
    for (int i = 0; i < 2 * coarray_length - 1; i++)
    {
        it = Jdict.find(1 - coarray_length + i);
        vector<int> j = it->second;
        col(E, i) += col(I, j[0]);
    }
    return E;
}

xarray<double> averaging_sampling(map<int, vector<int>> Jdict, int array_length, int coarray_length)
{
    map<int, vector<int>>::iterator it;
    int Ls = pow(array_length, 2);
    xarray<double> I = eye(Ls);
    xarray<double> E = zeros<double>({Ls, 2 * coarray_length - 1});
    for (int i = 0; i < 2 * coarray_length - 1; i++)
    {
        it = Jdict.find(1 - coarray_length + i);
        vector<int> j = it->second;
        for (int k = 0; k < j.size(); k++)
        {
            col(E, i) += col(I, j[k]);
        }
        col(E, i) = col(E, i) / j.size();
    }
    return E;
}

xarray<complex<double>> response_vector(double theta, double carrier_frequency, double propagation_speed, xarray<double> element_locations)
{
    using namespace complex_literals;
    double cc = (double)2 * M_PI * carrier_frequency / propagation_speed;
    xarray<complex<double>> rv;
    rv = exp(-1i * cc * sin(theta) * element_locations);
    return col(rv, 0);
}

xarray<complex<double>> response_matrix(xarray<double> thetas, double carrier_frequency, double propagation_speed, xarray<double> element_locations)
{
    double wavelength = propagation_speed / carrier_frequency;
    double unit_spacing = wavelength / 2;
    int L = element_locations.size();
    int K = thetas.size();
    xarray<complex<double>> S = zeros<complex<double>>({L, K});
    for (int i = 0; i < K; i++)
    {
        col(S, i) = response_vector(thetas(i), carrier_frequency, propagation_speed, element_locations);
    }
    return S;
}

xarray<double> smoothing_matrix(int coarray_length)
{
    int m = 0;
    xarray<double> Fm;
    xarray<double> B1 = zeros<double>({coarray_length, coarray_length - m - 1});
    xarray<double> B2 = eye(coarray_length);
    xarray<double> B3 = zeros<double>({coarray_length, m});
    xarray<double> F = concatenate(xtuple(B1, B2, B3), 1);
    for (m = 1; m < coarray_length; m++)
    {
        B1 = zeros<double>({coarray_length, coarray_length - m - 1});
        B3 = zeros<double>({coarray_length, m});
        Fm = concatenate(xtuple(B1, B2, B3), 1);
        F = concatenate(xtuple(F, Fm), 1);
    }
    return F;
}

xarray<complex<double>> snapshots(xarray<complex<double>> response_matrix, xarray<double> source_powers, double noise_power, int number_of_snapshots)
{
    using namespace complex_literals;
    int L = response_matrix.shape(0);
    int K = response_matrix.shape(1);
    xarray<complex<double>> Y = zeros<double>({L, number_of_snapshots}) + 1i * zeros<double>({L, number_of_snapshots});
    xarray<double> D = sqrt(diag(source_powers));
    xarray<complex<double>> symbols = linalg::dot(D, (random::randn<double>({K, number_of_snapshots}) + 1i * random::randn<double>({K, number_of_snapshots})) / sqrt(2));
    xarray<complex<double>> awgn = sqrt(noise_power) * (random::randn<double>({L, number_of_snapshots}) + 1i * random::randn<double>({L, number_of_snapshots})) / sqrt(2);
    Y = linalg::dot(response_matrix, symbols) + awgn;
    return Y;
}

xarray<complex<double>> autocorrelation_matrix_est(xarray<complex<double>> snapshots)
{
    xarray<complex<double>> Rest = linalg::dot(snapshots, transpose(conj(snapshots))) / snapshots.shape(1);
    return Rest;
}

xarray<complex<double>> autocorrelation_matrix(xarray<complex<double>> response_matrix, xarray<double> source_powers, double noise_power)
{
    xarray<complex<double>> R = linalg::dot(response_matrix, linalg::dot(diag(source_powers), transpose(conj(response_matrix)))) + noise_power * eye(response_matrix.shape(0));
    return R;
}

xarray<complex<double>> spatial_smoothing(xarray<double> smoothing_matrix, xarray<complex<double>> autocorrelations)
{
    xarray<complex<double>> Z;
    int Lp = smoothing_matrix.shape(0);
    xarray<complex<double>> I = eye(Lp);
    Z = linalg::dot(smoothing_matrix, linalg::kron(I, autocorrelations.reshape({2 * Lp - 1, 1})));
    return Z;
}