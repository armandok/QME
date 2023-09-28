#pragma once

#include <iostream>
#include "defs.hpp"
#include "reduced_quintessential.hpp"

namespace QME 
{

class Problem  {
public:
    Problem();

    Problem(const measurements_t &measurements);

    /** Set the maximum rank of the rank-restricted semidefinite relaxation */
    void set_relaxation_rank(size_t rank);

    /** Returns the current relaxation rank r of this problem */
    size_t relaxation_rank() const { return m_rank; }

    matrix_t sdp_matrix(const matrix_t &V) const;

    /** Given a matrix Y, this function computes and returns the matrix product
    CY, where C is the data matrix */
    matrix_t data_matrix_product(const matrix_t &Y) const;

    matrix_t data_matrix_product_stiefel(const matrix_t &V) const;

    /** Given a matrix Y, this function computes and returns F(Y), the value of
     * the objective evaluated at Y */
    scalar_t evaluate_objective(const matrix_t &Y) const;

    /** Given a matrix Y, this function computes and returns nabla F(Y), the
     * *Euclidean* gradient of F at Y. */
    matrix_t euclidean_gradient(const matrix_t &Y) const;

    /** Given a matrix Y in the domain D of the relaxation and the *Euclidean*
     * gradient nabla F(Y) at Y, this function computes and returns the
     * *Riemannian* gradient grad F(Y) of F at Y */
    matrix_t riemannian_gradient(const matrix_t &Y, const matrix_t &nablaF_Y) const;

    /** Given a matrix Y in the domain D of the relaxation, this function computes
     * and returns grad F(Y), the *Riemannian* gradient of F at Y */
    matrix_t riemannian_gradient(const matrix_t &Y) const;

    /** Given a matrix Y in the domain D of the relaxation, the *Euclidean*
     * gradient nablaF_Y of F at Y, and a tangent vector dotY in T_Y(D), the
     * tangent space of the domain of the optimization problem at Y, this function
     * computes and returns Hess F(Y)[dotY], the action of the
     * Riemannian Hessian on dotY */
    matrix_t riemannian_hessian_vector_product(const matrix_t &Y,
                                            const matrix_t &nablaF_Y,
                                            const matrix_t &dotY) const;

    
    matrix_t riemannian_Hessian_vector_product(const matrix_t &V,
                                            const matrix_t &dotV) const;

    /** Given a matrix Y in the domain D of the relaxation and a tangent vector
     * dotY in T_Y(E), the tangent space of Y considered as a generic matrix, this
     * function computes and returns the orthogonal projection of dotY onto
     * T_D(Y), the tangent space of the domain D at Y*/
    matrix_t tangent_space_projection(const matrix_t &Y, const matrix_t &dotY) const;                                        

    /** Given a matrix Y in the domain D of the relaxation and a tangent vector
     * dotY in T_D(Y), this function returns the point Yplus in D obtained by
     * retracting along dotY */
    matrix_t retract(const matrix_t &Y, const matrix_t &dotY) const;

    matrix_t retract_newton(const matrix_t &V, const matrix_t &dotV) const;

    matrix_t round_solution(const matrix_t &Y) const;

    matrix_t project_essential(const matrix_t &E) const;

    /** Given a critical point Y of the rank-r relaxation, this function
     * constructs the certificate matrix S(Y), and returns a
     * boolean value indicating whether S(Y) is positive-semidefinite.  In the
     * event that S is *not* positive-semidefinite, it also computes directions
     * of negative curvature x of S, and its corresponding Rayleigh quotient
     * theta := x'*S*x < 0.  Here:
     *
     * - eta is a numerical tolerance for S(Y)'s positive-semidefiniteness: we
     *   test the positive semidefiniteness of the *regularized* certificate
     *   matrix S(Y) + eta * I.
    */
    
    bool verify_solution(const matrix_t &V, scalar_t eta, std::vector<scalar_t> &theta_vec,
                              std::vector<vector_t> &x_vec) const;

    /** Randomly samples a point in the domain for the rank-restricted
     * semidefinite relaxation */
    matrix_t random_sample() const;

    matrix_t data_matrix_initialization();

    scalar_t get_objective_min_bound() const;

    ~Problem() {}

private:

    size_t m_rank;
    matrix_t m_C; // objective function
    ReducedQuintessential m_FE; // underlying manifold
    scalar_t m_min_objective_bound;
};

} // end of namespace QME