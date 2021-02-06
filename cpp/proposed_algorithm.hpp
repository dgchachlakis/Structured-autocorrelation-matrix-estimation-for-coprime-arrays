#include <iostream>
#include <complex>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xreducer.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xcomplex.hpp>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>

using namespace std;
using namespace xt;

xarray<complex<double>> nearest_psd(xarray<complex<double>> A)
{
    xarray<complex<double>> Y;
    auto eig = linalg::eig(A);
    auto eigvals = get<0>(eig);
    auto eigvecs = get<1>(eig);
    xarray<double> eigsreal = real(eigvals);
    for (int it = 0; it < eigsreal.size(); it++)
    {
        if (eigsreal[it] < 0)
            eigsreal[it] = 0;
    }
    Y = linalg::dot(eigvecs, linalg::dot(diag(eigsreal), transpose(conj(eigvecs))));
    return Y;
}

xarray<complex<double>> nearest_toeplitz(xarray<complex<double>> A)
{
    xarray<complex<double>> Y;
    xarray<complex<double>> temp, temp2;
    xarray<complex<double>> current_diag;
    xarray<complex<double>> current_mean;
    Y = A;
    int Lp = A.shape(0);
    for (int i = -Lp + 1; i < Lp; i++)
    {
        current_diag = diagonal(A, i);
        current_mean = mean(current_diag);
        temp = eye(Lp, i) * current_mean;
        temp2 = eye(Lp, i) * Y;
        Y = Y - temp2 + temp;
    }
    return Y;
}

xarray<complex<double>> eigen_value_correction(xarray<complex<double>> A, int d)
{
    int x;
    int Lp = A.shape(0);
    xarray<complex<double>> Y = zeros<double>({Lp, Lp});
    xarray<complex<double>> evsort = zeros<double>({Lp, Lp});
    xarray<complex<double>> evasort = zeros<double>({Lp});
    auto eig = linalg::eig(A);
    auto eigvals = get<0>(eig);
    auto eigvecs = get<1>(eig);
    xarray<double> eigsreal = real(eigvals);
    xarray<double> idx = argsort(eigsreal);

    for (int i = 0; i < idx.size(); i++)
    {
        x = (int)idx[i];
        col(evsort, i) += col(eigvecs, x);
        evasort[i] = eigsreal[x];
    }
    complex<double> ss;
    for (int j = 0; j < d; j++)
        ss += evasort[j];
    ss = ss / (complex<double>)d;
    xarray<complex<double>> evals_corr = zeros<double>({Lp});
    for (int j = 0; j < Lp; j++)
    {
        if (j < d)
            evals_corr[j] += ss;
        if (j >= d)
            evals_corr[j] += evasort[j];
    }
    Y = linalg::dot(evsort, linalg::dot(diag(evals_corr), transpose(conj(evsort))));
    return Y;
}

xarray<complex<double>> sqrtm(xarray<complex<double>> A)
{
    int x;
    int Lp = A.shape(0);
    xarray<complex<double>> Y = zeros<double>({Lp, Lp});
    auto eig = linalg::eig(A);
    auto eigvals = get<0>(eig);
    auto eigvecs = get<1>(eig);
    for (int i = 0; i < eigvals.size(); i++)
        eigvals[i] = sqrt(abs(eigvals[i]));
    Y = linalg::dot(eigvecs, linalg::dot(diag(eigvals), transpose(conj(eigvecs))));
    return Y;
}

xarray<complex<double>> structured_estimate(xarray<complex<double>> A, int d, int maxiter, bool display)
{

    double dif, tol = pow(10, -6);
    xarray<complex<double>> Y, YY, met_prev, met;
    Y = A;
    if (display == true)
    {
        cout << "*---------------------------------*" << endl;
        cout << "Iteration"
                  << "\t"
                  << "Squared norm" << endl;
        cout << "*---------------------------------*" << endl;
        cout << endl;
    }
    YY = linalg::dot(transpose(conj(Y)), Y);
    met_prev = sum(diagonal(YY));
    for (int i = 0; i < maxiter; i++)
    {
        Y = nearest_toeplitz(Y);
        Y = nearest_psd(Y);
        Y = eigen_value_correction(Y, d);
        YY = linalg::dot(transpose(conj(Y)), Y);
        met = sum(diagonal(YY));

        if (display == true)
            cout << i + 1 << "\t\t" << real(met(0)) << endl;
        dif = (double)real(met_prev(0)) - real(met(0));

        if (dif < tol)
            break;
        met_prev = met;
    }
    return Y;
}