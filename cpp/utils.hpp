#include <iostream>
#include <complex>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xsort.hpp>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>


int array_length(int M, int N)
{
    return (2 * M + N - 1);
}

int coarray_length(int M, int N)
{
    return (M * N + M);
}

std::map<int, std::vector<int>> form_index_sets(int M, int N, xt::xarray<double> pdist, double carrier_frequency, double propagation_speed)
{
    double wavelength = propagation_speed / carrier_frequency;
    double unit_spacing = wavelength / 2;
    int co_length = coarray_length(M, N);
    std::map<int, std::vector<int>> dict;
    for (int cnt=0; cnt<pdist.size(); cnt++)
    {
      int nn = (int) pdist(cnt) / unit_spacing;
      if (abs(nn) < co_length)
      {
        dict[nn].push_back(cnt);
      }
    }
    return dict;
}

xt::xarray<double> ca_element_locations(int M, int N, double carrier_frequency, double propagation_speed)
{
    int L = 2 * M + N - 1;
    double wavelength = propagation_speed / carrier_frequency;
    double unit_spacing = wavelength / 2;
    std::cout << wavelength << std::endl;
    std::cout << unit_spacing << std::endl;

    xt::xarray<double> p = xt::zeros<double>({L, 1});
    for (int i=0; i<N;  i++)
    {
        p(i,0) = i * M * unit_spacing;
    }
    for (int i=0; i<2 * M - 1; i++)
    {
        p(N+i,0) = (i + 1) * N * unit_spacing;
    }   
    xt::xarray<double> pp =xt::sort(p,0);
    return pp;
}

xt::xarray<double> pair_wise_distances(xt::xarray<double> v)
{
    int N = v.size();
    xt::xarray<double> o  = xt::ones<double>({N, 1});
    xt::xarray<double> v1 = xt::linalg::dot(v, xt::transpose(o));
    xt::xarray<double> v2 = xt::linalg::dot(o, xt::transpose(v));
    xt::xarray<double> dif = v1 - v2;
    xt::xarray<double> fl = xt::flatten(dif);
    return xt::transpose(fl);
}

xt::xarray<double> selection_sampling(std::map< int, std::vector<int> > Jdict, int array_length, int coarray_length)
{
    std::map<int,std::vector<int>>::iterator it;

    int Ls = pow(array_length, 2);
    xt::xarray<double> I = xt::eye(Ls);
    xt::xarray<double> E = xt::zeros<double>({Ls,2 * coarray_length - 1});
    for (int i = 0; i < 2 * coarray_length -1 ; i++)
    {
        it = Jdict.find(1 - coarray_length + i);
        std::vector<int> j = it->second;
        xt::col(E, i) = xt::col(I, j[0]);
        break;
    }
    return E;
}

xt::xarray<double> averaging_sampling(std::map< int, std::vector<int> > Jdict, int array_length, int coarray_length)
{
    std::map<int,std::vector<int>>::iterator it;

    int Ls = pow(array_length, 2);
    xt::xarray<double> I = xt::eye(Ls);
    xt::xarray<double> E = xt::zeros<double>({Ls,2 * coarray_length - 1});
    for (int i = 0; i < 2 * coarray_length -1 ; i++)
    {
        it = Jdict.find(1 - coarray_length + i);
        std::vector<int> j = it->second;
        for (int k = 0; k < j.size(); k++)
        {
            xt::col(E, i) += xt::col(I, j[k]);
        }
        xt::col(E, i) = xt::col(E, i) / j.size();
    }
    return E;
}

xt::xarray<std::complex<double>> response_vector(double theta, double carrier_frequency, double propagation_speed, xt::xarray<double> element_locations)
{
    using namespace std::complex_literals;
    double cc = (double) 2 * M_PI * carrier_frequency / propagation_speed;
    xt::xarray<std::complex<double>> rv;
    rv = exp( -1i * cc * sin(theta) * element_locations);
    return xt::col(rv, 0);
}

xt::xarray<std::complex<double>> response_matrix(xt::xarray<double> thetas, double carrier_frequency, double propagation_speed, xt::xarray<double> element_locations)
{
    double wavelength = propagation_speed / carrier_frequency;
    double unit_spacing = wavelength / 2;
    int L = element_locations.size();
    int K = thetas.size();
    xt::xarray<std::complex<double>> S = xt::zeros<std::complex<double>>({L, K});
    for (int i = 0; i < K; i++)
    {
        xt::col(S, i) = response_vector(thetas(i), carrier_frequency, propagation_speed, element_locations);
    }
    return S;
}