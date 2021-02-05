#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include "utils.hpp"

int main()
{
    int M = 2;
    int N = 3;
    double carrier_frequency = (double) 3/2 * pow(10,8);
    double propagation_speed = 3 * pow(10,8);
    int L = array_length(M, N);
    int Lp = coarray_length(M, N);

    xt::xarray<double> elem = ca_element_locations(M, N, carrier_frequency, propagation_speed);
    std::cout << elem;
    xt::xarray<double> dif = pair_wise_distances(elem);
    std::cout << dif << std::endl;
    std::map<int, std::vector<int>> Jsets = form_index_sets(M, N, dif, carrier_frequency, propagation_speed);
    for(std::map<int, std::vector<int> > ::const_iterator it = Jsets.begin(); it != Jsets.end(); ++it)
    {
        std::vector<int> jset = it->second;
        for (std::vector<int>::const_iterator i = jset.begin(); i != jset.end(); ++i)
        {
            std::cout << *i << ' ';
        }
        std::cout << std::endl;  
    }

    xt::xarray<double> Esel = selection_sampling(Jsets,  L, Lp);
    std::cout << Esel <<std::endl;

    xt::xarray<double> Eavg = averaging_sampling(Jsets,  L, Lp);
    std::cout << Eavg << std::endl;
    

    xt::xarray<std::complex<double>> rv = response_vector(M_PI_4, carrier_frequency, propagation_speed,  elem);
    std::cout << rv << std::endl;

    xt::xarray<double> thetas = { M_PI_4, M_PI_4/2, M_PI_4/3};
    xt::xarray<std::complex<double>> S = response_matrix(thetas, carrier_frequency, propagation_speed, elem);
    std::cout << S << std::endl;


    std::cout << "\n";    
    return 0;
    
}