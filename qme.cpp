#include <functional>
#include <algorithm>
#include <chrono>

#include "qme.hpp"

#include "Optimization/Riemannian/TNT.h"

namespace QME
{

QMEResult QME(Problem& problem,
              const QMEOpts& options,
              const matrix_t& V0)
{
    if (options.verbose)
    {
        std::cout << "========= QME ==========" << std::endl << std::endl;
        std::cout << " Initial level of Riemannian staircase: " << options.r0
                << std::endl;
        std::cout << " Maximum level of Riemannian staircase: " << options.rmax
                << std::endl;
        std::cout << " Tolerance for accepting an eigenvalue as numerically "
                    "nonnegative in optimality verification: "
                << options.min_eig_num_tol << std::endl;
        std::cout << " Initialization method: "
              << (options.initialization == Initialization::Eigen   ? "eigen"
                                                                    : "random");
        
        if (options.log_iterates)
        {
            std::cout << " Logging entire sequence of Riemannian Staircase iterates"
                    << std::endl;
        }
        std::cout << "Riemannian trust-region settings:" << std::endl;
        std::cout << " Stopping tolerance for norm of Riemannian gradient: "
                << options.grad_norm_tol << std::endl;
    }
    
    QMEResult result;
    result.status = MaxRank;

    matrix_t V;
    matrix_t NablaF_V;

    QME::precise_stopwatch stopwatch_total;
    
    // Objective
    Optimization::Objective<matrix_t, scalar_t, matrix_t> F =
        [&problem](const matrix_t &V, const matrix_t &NablaF_V)
        {
            return problem.evaluate_objective(V);
        };
    
    // Local quadratic model constructor
    Optimization::Riemannian::QuadraticModel<matrix_t, matrix_t, matrix_t> QM =
        [&problem](const matrix_t &V, matrix_t &grad,
                Optimization::Riemannian::LinearOperator<matrix_t, matrix_t,
                                                          matrix_t> &HessOp,
                 matrix_t &NablaF_V)
            {
                // Compute and cache Euclidean gradient at the current iterate
                NablaF_V = problem.euclidean_gradient(V);
                
                // Compute Riemannian gradient from Euclidean gradient
                grad = problem.riemannian_gradient(V, NablaF_V);
                
                // Define linear operator for computing Riemannian Hessian-vector
                // products
                HessOp = [&problem](const matrix_t &V, const matrix_t &Vdot,
                                    const matrix_t &NablaF_V)
                    {
                        return problem.riemannian_hessian_vector_product(V, NablaF_V, Vdot);
                    };
            };

    Optimization::Riemannian::RiemannianMetric<matrix_t, matrix_t, scalar_t, matrix_t>
    metric = [&problem](const matrix_t &V, const matrix_t &T1, const matrix_t &T2,
                          const matrix_t &NablaF_V)
        {
            return (T1 * T2.transpose()).trace();
        };

    // Retraction operator
    Optimization::Riemannian::Retraction<matrix_t, matrix_t, matrix_t> retraction =
        [&problem](const matrix_t &V, const matrix_t &Vdot, const matrix_t &NablaF_V) 
            {
                return problem.retract(V, Vdot);
            };
    
    // Not using a preconditioner
    std::optional<
        Optimization::Riemannian::LinearOperator<matrix_t, matrix_t, matrix_t>>
        precon = std::nullopt;
    
    if (options.verbose) {std::cout << "Initialization:" << std::endl;}
    
    problem.set_relaxation_rank(options.r0);

    if (V0.size() != 0)
    {
        if (options.verbose) {std::cout << "Using user-supplied initial iterate V0" << std::endl;}
        V = V0;
    }
    else
    {
        if (options.initialization == Initialization::Eigen)
        {
            if (options.verbose) {std::cout << "Data matrix initialization (eigen)" << std::endl;}
            V = problem.data_matrix_initialization();
            result.f_min_bound = problem.get_objective_min_bound();
        }
        else
        {
            if (options.verbose) {std::cout << "Random initialization (random)" << std::endl;}
            V = problem.random_sample();
        }
    }
    result.f_max_bound = problem.evaluate_objective(V);
    if (options.verbose)
    {
        // Compute and display the initial objective value
        std::cout << "Initial objective value: " << result.f_max_bound
              << std::endl;
    }

    /// RIEMANNIAN STAIRCASE
    // Configure optimization parameters
    Optimization::Riemannian::TNTParams<scalar_t> params;
    params.gradient_tolerance = options.grad_norm_tol;
    params.relative_decrease_tolerance = options.rel_func_decrease_tol;
    params.stepsize_tolerance = options.stepsize_tol;
    params.max_iterations = options.max_iterations;
    params.max_TPCG_iterations = options.max_tCG_iterations;
    params.kappa_fgr = options.STPCG_kappa;
    params.theta = options.STPCG_theta;
    params.log_iterates = options.log_iterates;
    params.verbose = options.verbose;
    
    for (size_t r = options.r0; r <= options.rmax; r++)
    {
        if (options.verbose)
        {std::cout << std::endl << "====== RIEMANNIAN STAIRCASE (level r = " << r << ") ======" << std::endl;}
        /// Run optimization!
        Optimization::Riemannian::TNTResult<matrix_t, scalar_t> tnt_result =
            Optimization::Riemannian::TNT<matrix_t, matrix_t, scalar_t, matrix_t>(
                F, QM, metric, retraction, V, NablaF_V, precon, params);
        
        result.Vopt = tnt_result.x;
        result.f_V = tnt_result.f;
        result.gradnorm = problem.riemannian_gradient(result.Vopt).norm();
        result.function_values.push_back(tnt_result.objective_values);
        result.gradient_norms.push_back(tnt_result.gradient_norms);
        result.elapsed_optimization_times.push_back(tnt_result.time);
        if (options.log_iterates)
        {result.iterates.push_back(tnt_result.iterates);}

        if (options.verbose)
        {
            // Display some output to the user
            std::cout << std::endl
                        << "Found first-order critical point with value F(Y) = "
                        << result.f_V
                        << "!  Elapsed computation time: " << tnt_result.elapsed_time
                        << " seconds" << std::endl
                        << std::endl;
            std::cout << "Checking second order optimality ... " << std::endl;
        }
        QME::precise_stopwatch stopwatch_verify;
        std::vector<vector_t> v_vec;
        std::vector<scalar_t> theta_vec;
        bool is_global_opt = problem.verify_solution(tnt_result.x, options.min_eig_num_tol, theta_vec,
                                v_vec);
        auto verification_time = stopwatch_verify.elapsed_time<unsigned int, std::chrono::microseconds>();
        auto total_time = stopwatch_total.elapsed_time<unsigned int, std::chrono::microseconds>();
        if (options.verbose){std::cout << "> > > > Duration: " << total_time << std::endl;}
        result.staircase_times_ms.push_back(total_time);

        if (is_global_opt)
        {
            // results.Vopt is a second-order critical point (global optimum)
            if (options.verbose)
                std::cout
                    << "Found second-order critical point! Elapsed computation time: "
                    << verification_time << " microseconds." << std::endl;
            result.status = GlobalOpt;
            break;
        } // global optimality
        else
        {
        /// ESCAPE FROM SADDLE!
            if (options.verbose)
            {
                /* std::cout << "Saddle point detected! Curvature along escape direction: "
                        << theta << ".  Elapsed computation time: "
                        << 0 << " seconds"  << std::endl; */
                std::cout << "Saddle point detected! Curvature along escape direction: "
                        << theta_vec[0] << ", " << theta_vec.back() << " size: " << theta_vec.size()
                        << ".  Elapsed computation time: "
                        << 0 << " seconds"  << std::endl;
                    
            }
            problem.set_relaxation_rank(r + 1);

            matrix_t Vplus;
            
            bool escape_success = escape_saddle(
                problem, result.Vopt, theta_vec, v_vec, options.grad_norm_tol, Vplus);
            if (escape_success)
            {
                // Update initialization point for next level in the Staircase
                V = Vplus;
            }
            else
            {
                if (options.verbose)
                std::cout
                    << "WARNING!  BACKTRACKING LINE SEARCH FAILED TO ESCAPE FROM "
                        "SADDLE POINT!  (Try decreasing the preconditioned "
                        "gradient norm tolerance)"
                    << std::endl;
                result.status = SaddlePoint;
                break;
            }
        } // saddle point

        total_time = stopwatch_total.elapsed_time<unsigned int, std::chrono::microseconds>();
        if (options.verbose) {std::cout << "> > > > Duration: " << total_time << std::endl;}
    } // Riemannian Staircase

    /// POST-PROCESSING

    if (options.verbose)
    {
        std::cout << std::endl
                << std::endl
                << "===== END RIEMANNIAN STAIRCASE =====" << std::endl
                << std::endl;

        switch (result.status)
        {
            case GlobalOpt:
                std::cout << "Found global optimum!" << std::endl;
                break;
            case EigImprecision:
                std::cout << "WARNING: Escape direction computation did not achieve "
                    "sufficient accuracy; solution may not be globally optimal!"
                        << std::endl;
                break;
            case SaddlePoint:
                std::cout << "WARNING: Line search was unable to escape saddle point!  "
                    "Solution is not globally optimal!"
                        << std::endl;
                break;
            case MaxRank:
                std::cout << "WARNING: Riemannian Staircase reached the maximum "
                    "permitted level before finding global optimum!"
                    << std::endl;
                break;
            case ElapsedTime:
                std::cout << "WARNING: Algorithm exhausted the allotted computation "
                    "time before finding global optimum!"
                    << std::endl;
                break;
        }
    }

    if (options.verbose){std::cout << std::endl << "Rounding solution ... ";}
    QME::precise_stopwatch stopwatch_rounding;
    result.Vhat = problem.round_solution(result.Vopt);
    result.f_Vhat = problem.evaluate_objective(result.Vhat);
    auto rounding_time = stopwatch_rounding.elapsed_time<unsigned int, std::chrono::microseconds>();
    if (options.verbose){std::cout << "Rounding time: " << rounding_time << " (ms)" << std::endl;}

    result.total_computation_time_ms = stopwatch_total.elapsed_time<unsigned int, std::chrono::microseconds>();
    if (options.verbose){std::cout << "> > > > Duration: " << result.total_computation_time_ms << std::endl;}

    /// LOCAL OPTIMIZATION
    
    if (options.do_local_optimization)
    {
        problem.set_relaxation_rank(1);
        QME::precise_stopwatch stopwatch_local; 
        Optimization::Riemannian::TNTResult<matrix_t, scalar_t> tnt_result =
            Optimization::Riemannian::TNT<matrix_t, matrix_t, scalar_t, matrix_t>(
                F, QM, metric, retraction, result.Vhat, NablaF_V, precon, params);
        
        result.local_optimization_time_ms = stopwatch_local.elapsed_time<unsigned int, std::chrono::microseconds>();
        if (options.verbose)
        {
            std::cout << "Local Optim Objective: " << tnt_result.f << std::endl;
            std::cout << "Local Optim Duration: " << result.local_optimization_time_ms << std::endl;
        }

        // Only save the results if the objective is lower than before
        if (tnt_result.f < result.f_Vhat)
        {
            result.Vhat_local = tnt_result.x;
            result.f_Vhat_local = tnt_result.f;
        }
        else
        {
            result.Vhat_local = result.Vhat;
            result.f_Vhat_local = result.f_Vhat;
        }
    }

    if (options.verbose) 
    {
        std::cout << "===== END QME =====" << std::endl << std::endl;
    }
    return result;
}

QMEResult QME(const measurements_t& measurements,
              const QMEOpts& options,
              const matrix_t& V0)
{
    Problem prob(measurements);
    return QME(prob, options, V0);
}


bool escape_saddle(const Problem &problem, const matrix_t& V, std::vector<scalar_t> theta_vec,
                   const std::vector<vector_t> &v_vec, scalar_t gradient_tolerance, matrix_t &Vplus)
{
    scalar_t f_init = problem.evaluate_objective(V);

    size_t r = problem.relaxation_rank();
    
    matrix_t V_augmented = matrix_t::Zero(4*r, 3);
    V_augmented.topRows(4*r-4) = V;


    // Vectors of trial stepsizes and corresponding function values
    std::vector<double> alphas;
    std::vector<double> fvals;

    bool reduced_objective = false;

    matrix_t Vtest;
    for(size_t idx=0; idx<theta_vec.size(); idx++)
    {
        scalar_t theta = theta_vec[idx];
        vector_t v = v_vec[idx];

        matrix_t Vdot = matrix_t::Zero(4*r, 3);
        Vdot.bottomRows<4>().col(0) = v.segment(0,4);
        Vdot.bottomRows<4>().col(1) = v.segment(4,4);
        Vdot.bottomRows<4>().col(2) = v.segment(8,4);

        scalar_t alpha_min = 1e-9; // Minimum stepsize
        scalar_t m1 = 64;
        scalar_t m2 = 10;
        scalar_t m3 = 10;
        scalar_t alpha = m3 *
            std::max(m1 * alpha_min, m2 * gradient_tolerance / fabs(theta));

        /// Backtracking line search
        while (alpha >= alpha_min)
        {
            // Retract along the given tangent vector using the given stepsize
            Vtest = problem.retract(V_augmented, alpha * Vdot);

            scalar_t f_test = problem.evaluate_objective(Vtest);
            matrix_t grad_fVtest = problem.riemannian_gradient(Vtest);
            scalar_t grad_fVtest_norm = grad_fVtest.norm();

            alphas.push_back(alpha);
            fvals.push_back(f_test);

            if ((f_test < f_init) && (grad_fVtest_norm > gradient_tolerance))
            {
                // Accept this trial point and return success
                Vplus = Vtest;
                return true;
            }
            alpha /= 2;

        }

        auto fmin_iter = std::min_element(fvals.begin(), fvals.end());
        auto min_idx = std::distance(fvals.begin(), fmin_iter);

        double f_min = fvals[min_idx];
        double a_min = alphas[min_idx];

        if (f_min < f_init)
        {
            // If this trial point strictly decreased the objective value, accept it and
            // return success
            Vplus = problem.retract(V_augmented, a_min * Vdot);
            return true;
        }
    }

    // NO trial point decreased the objective value: we were unable to escape
    // the saddle point!
    return false;

}

} // end of namespace QME