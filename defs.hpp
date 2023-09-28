#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

#include <chrono>
#include "stopwatch.hpp"

namespace QME 
{
using scalar_t          = double;
using vector_t          = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using matrix_t          = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

using vector3_t         = Eigen::Matrix<scalar_t, 3, 1>;
using bearing_t         = vector3_t;

using matrix3_t         = Eigen::Matrix<scalar_t, 3, 3>;
using rotation_t        = matrix3_t;
using measurements_t    = std::vector<std::pair<bearing_t,bearing_t>>;

using Vector    = matrix_t;
using Variable  = matrix_t;

enum class Initialization { Eigen, Random };

} // end of namespace QME