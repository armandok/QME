#pragma once

#include "defs.hpp"
#include <cmath>
namespace QME 
{

std::pair<scalar_t, scalar_t> get_rotation_and_translation_error(
    matrix3_t essential,
    matrix_t rot12,
    vector3_t t12
)
{
    Eigen::JacobiSVD<matrix_t> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);

    matrix3_t U = svd.matrixU();
    matrix3_t Vt = svd.matrixV().transpose();

    if (U.determinant() * Vt.determinant() < 0)
    {
        U.col(2) *= -1;
    }

    assert(U.determinant() * Vt.determinant()>0);

    matrix3_t W ;
    W << 0, -1, 0,
         1, 0, 0,
         0, 0, 1;
    
    matrix3_t rotation1 = U * W * Vt;
    matrix3_t rotation2 = U * W.transpose() * Vt;

    vector3_t translation = svd.matrixU().col(2);

    
    scalar_t tr1 = (rot12.transpose()*rotation1).trace();
    scalar_t tr2 = (rot12.transpose()*rotation2).trace();

    scalar_t rot_error_cos = std::min((std::max( tr1 , tr2 )-1)/2, 1.0);
    scalar_t rot_error_deg = std::acos(rot_error_cos)*180/M_PI;
    
    scalar_t trans_err_cos = std::min(std::abs(translation.dot(t12)), 1.0);
    scalar_t trans_err_deg = std::acos(trans_err_cos)*180/M_PI;

    return std::make_pair(rot_error_deg,trans_err_deg);
}

matrix_t get_data_matrix(const measurements_t m)
{
    matrix_t C = matrix_t::Zero(9,9);
    scalar_t N = m.size();

    for (auto bearings: m)
    {
        bearing_t b1 = bearings.first;
        bearing_t b2 = bearings.second;

        matrix_t v; 
        v.resize(9,1);

        v << b1(0)*b2, b1(1)*b2, b1(2)*b2;
        
        C += v * v.transpose()/N;
    }
    return C;
}

scalar_t get_algebraic_error(
    const matrix3_t essential,
    const measurements_t m
)
{
    scalar_t result = 0;
    scalar_t N = m.size();

    for (auto measurement: m)
    {
        scalar_t prod = measurement.first.transpose() * essential * measurement.second;
        result += prod * (prod * N);
    }
    return result;
}

} // end of namespace QME