#pragma once

#include <iostream>
#include "defs.hpp"
#include "problem.hpp"

namespace QME 
{

struct QMEOpts
{
    scalar_t grad_norm_tol = 1e-7;
    scalar_t rel_func_decrease_tol = 1e-7;
    scalar_t stepsize_tol = 1e-4;
    size_t max_iterations = 1000;
    size_t max_tCG_iterations = 10000;
    double max_computation_time_sec = 5;
    scalar_t STPCG_kappa = 0.1;
    scalar_t STPCG_theta = .5;
    size_t r0 = 1;
    size_t rmax = 12;
    scalar_t min_eig_num_tol = 1e-7;
    Initialization initialization = Initialization::Eigen;
    bool verbose = true;
    bool log_iterates = false;
    bool do_local_optimization = true;
};

enum QMEStatus
{
    GlobalOpt,
    SaddlePoint,
    EigImprecision,
    MaxRank,
    ElapsedTime
};

struct QMEResult
{
    matrix_t Vopt;
    scalar_t f_V;
    scalar_t gradnorm;
    scalar_t f_Vhat;
    matrix_t Vhat;
    scalar_t f_Vhat_local;
    matrix_t Vhat_local;
    scalar_t f_min_bound;
    scalar_t f_max_bound;
    unsigned int total_computation_time_ms;
    unsigned int local_optimization_time_ms;
    
    std::vector<unsigned int> staircase_times_ms;

    std::vector<std::vector<scalar_t>> function_values;
    std::vector<std::vector<scalar_t>> gradient_norms;
    std::vector<std::vector<double>> elapsed_optimization_times;
    std::vector<unsigned int> verification_times;
    std::vector<scalar_t> escape_direction_curvatures;
    std::vector<std::vector<matrix_t>> iterates;
    QMEStatus status;
};

QMEResult QME(Problem& problem,
              const QMEOpts& options = QMEOpts(),
              const matrix_t& V0 = matrix_t());

QMEResult QME(const measurements_t& measurements,
              const QMEOpts& options = QMEOpts(),
              const matrix_t& V0 = matrix_t());

bool escape_saddle(const Problem &problem, const matrix_t& V, std::vector<scalar_t> theta_vec,
                   const std::vector<vector_t> &v, scalar_t gradient_tolerance, matrix_t &Vplus);


} // end of namespace QME