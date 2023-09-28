
#include "reduced_quintessential.hpp"

#include <iostream>

//#include <cmath>
#include <math.h>

namespace QME 
{
    ReducedQuintessential::ReducedQuintessential() : m_rank(1)
    {
    }

    ReducedQuintessential::ReducedQuintessential(int r) : m_rank(r)
    {
    }

    ReducedQuintessential::~ReducedQuintessential(void)
    {
    }

    Variable ReducedQuintessential::Retract(const Vector& x, const Vector& v) const
    {
        return RetractNewton(x+v);
    }

    double ReducedQuintessential::Norm(const Vector& x, const Vector& v) const
    {
        return 1;
    }

    void ReducedQuintessential::Gradient(const Vector& x, const Vector& y, Vector& grad) const
    {

    }

    Vector ReducedQuintessential::Projection(const Variable x, const Vector v) const
    {
        Vector v_tilde = v - x * (v.transpose()*x + x.transpose()*v) * 0.5;
        
        if (m_rank==1)
        {
            return v_tilde;
        }

        double c, c_nom, c_denom = 0;

        std::vector<int> ind;
        for (size_t rdx = 0; rdx<m_rank; rdx++)
        {
            ind.push_back(rdx*4+3);
        }
        
        c_nom = x(ind, Eigen::all).cwiseProduct(v_tilde(ind, Eigen::all)).sum();
        c_denom = 1 - (x(ind, Eigen::all).transpose()*x(ind, Eigen::all)).squaredNorm();

        c = c_nom/c_denom;

        if (isnan(c) || isinf(c))
        {
            return v_tilde;
        }
        
        Vector xtgx = Vector::Zero(3,3);
        Vector gx   = Vector::Zero(4*m_rank,3);

        for (size_t idx = 0; idx<3; idx++)
        {
            for (size_t jdx = idx; jdx<3; jdx++)
            {
                for (size_t rdx = 0; rdx<m_rank; rdx++)
                {
                    double v1 = x(rdx*4+3,idx);
                    double v2 = x(rdx*4+3,jdx);
                    xtgx(idx, jdx) += v1*v2;
                }
                xtgx(jdx, idx) = xtgx(idx, jdx);
            }
        }
        gx(Eigen::seq(3,m_rank*4-1,4), Eigen::all) = x(Eigen::seq(3,m_rank*4-1,4), Eigen::all);
        
        

        Vector v_final = v_tilde - c*(gx-x*xtgx);

        return v_final;
    }

    Vector ReducedQuintessential::RandomManifold(void) const
    {
        Vector rand = Vector::Random(4*m_rank,3);

        return Retract(rand, Vector::Zero(rand.rows(), rand.cols()));
    }

    Vector ReducedQuintessential::EucGradToGrad(const Variable& x, const Vector& egf) const
    {
        return Projection(x, egf);
    }
    
    
    Vector ReducedQuintessential::EucHvToHv(const Variable& x, const Vector& egrad, const Vector& ehess, const Vector& x_dot) const
    {   
        Vector gx = Vector::Zero(4*m_rank, 3);
        Vector gx_dot = Vector::Zero(4*m_rank, 3);
        for (size_t idx = 0; idx<m_rank; idx++)
        {
            gx.row(idx*4+3) = x.row(idx*4+3);
            gx_dot.row(idx*4+3) = x_dot.row(idx*4+3);
        }

        Vector xteg = x.transpose() * egrad;
        Vector xteg_sym = 0.5 * (xteg + xteg.transpose());

        Vector mm = Vector::Zero(3, 3);
        for (size_t idx = 0; idx<3; idx++)
        {
            for (size_t jdx = idx; jdx<3; jdx++)
            {
                for (size_t rdx = 0; rdx<m_rank; rdx++)
                {
                    mm(idx, jdx) += x(4*rdx+3,idx) * x(4*rdx+3,jdx);
                }

                mm(jdx, idx) = mm(idx, jdx); // To symmetrize mm
            }
        }
        double c_nom = mm.cwiseProduct(xteg).sum();
        double c_denom = 1 - mm.squaredNorm();

        double c = c_nom/c_denom;
        if (isnan(c_denom) || isinf(c) || isnan(c))
        {
            c = 0;
        }

        std::vector<int> ind;
        for (size_t rdx = 0; rdx<m_rank; rdx++)
        {
            ind.push_back(4*rdx+3);
        }
        Vector xtgx_dot = x(ind, Eigen::all).transpose() * x_dot(ind, Eigen::all);

        double c_nom_dot = 2 * xtgx_dot.cwiseProduct(xteg_sym).sum() +  mm.cwiseProduct( x_dot.transpose()*egrad + x.transpose()*ehess ).sum();
        double c_denom_dot = -4 * mm.cwiseProduct(xtgx_dot).sum();
        
        double c_dot = 0;

        c_dot = c_nom_dot/c_denom - c_denom_dot*c_nom/pow(c_denom,2);
        if (isnan(c_dot) || isinf(c_dot))
        { 
            c_dot=0;
        }

        Vector to_project = ehess-x_dot*xteg_sym-c_dot*gx-c*(gx_dot-x_dot*mm);;
        return Projection(x, to_project);
    }

    Vector ReducedQuintessential::GetConstraintsVal(const Variable x) const
    {
        Vector constraints_val = Vector::Zero(7,1);

        constraints_val(0) = -1+x.col(0).squaredNorm();
        constraints_val(1) = -1+x.col(1).squaredNorm();
        constraints_val(2) = -1+x.col(2).squaredNorm();

        constraints_val(3) = x.col(0).dot(x.col(1));
        constraints_val(4) = x.col(0).dot(x.col(2));
        constraints_val(5) = x.col(1).dot(x.col(2));

        constraints_val(6) = -1+x(Eigen::seqN(3,m_rank,4), Eigen::all).squaredNorm();
        
        return constraints_val;
    }

    Vector ReducedQuintessential::GetGradConstraintsVal(const Variable x, const Variable g) const
    {
        Vector constraints_val = Vector::Zero(7,1);

        Vector gtx   = x.transpose() * g;

        constraints_val(0) = gtx(0,0);
        constraints_val(1) = gtx(1,1);
        constraints_val(2) = gtx(2,2);

        constraints_val(3) = gtx(1,0)+gtx(0,1);
        constraints_val(4) = gtx(2,0)+gtx(0,2);
        constraints_val(5) = gtx(2,1)+gtx(1,2);

        constraints_val(6) = 0;
        for (size_t idx=0; idx<m_rank; ++idx)
        {
            constraints_val(6) += x.row(idx*4+3).dot(g.row(idx*4+3));
        } //x(Eigen::seqN(3,m_rank,4), Eigen::all).dot(g(Eigen::seqN(3,m_rank,4), Eigen::all));

        return constraints_val;
    }
    
    Variable ReducedQuintessential::RetractHeuristic(const Vector U) const
    {
        
        Vector thinQ = U;

        scalar_t norm = thinQ(Eigen::seqN(3,m_rank,4), Eigen::all).norm();

        thinQ(Eigen::seqN(3,m_rank,4), Eigen::all) /= norm;

        Vector K_half = thinQ(Eigen::seqN(3,m_rank,4), Eigen::all);
        Vector K = Vector::Identity(3,3) - K_half.transpose() * K_half;
        Vector K_sqrt;
        
        if (K.determinant() > 1e-12)
        {
            K_sqrt = K.pow(0.5);
        }
        else
        {
            Eigen::JacobiSVD<Vector> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // Get the singular values and matrices U, V
            Vector singularValues = svd.singularValues();
            Vector U_ = svd.matrixU();
            Vector V_ = svd.matrixV();

            // Set small singular values to zero to handle numerical errors (e.g., due to zero eigenvalue)
            const double epsilon = 1e-12;
            for (int i = 0; i < singularValues.size(); ++i) {
                if (singularValues(i) < epsilon) {
                    singularValues(i) = 0.0;
                } else {
                    singularValues(i) = std::sqrt(singularValues(i));
                }
            }
            // Compute the square root of A as A_sqrt = U * S * V.transpose(), where S is a diagonal matrix of singular values
            K_sqrt = U_ * singularValues.asDiagonal() * V_.transpose();
        }
        
        std::vector<int> ind;
        for (size_t idx=0; idx<m_rank; ++idx)
        {
            ind.push_back(4*idx+0);
            ind.push_back(4*idx+1);
            ind.push_back(4*idx+2);
        }
        
        Vector M = K_sqrt * thinQ(ind, Eigen::all).transpose() * thinQ(ind, Eigen::all) * K_sqrt;
        Vector M_pinv_sqrt;
        if (M.determinant() > 1e-12)
        {
            M_pinv_sqrt = M.pow(-0.5);
        }
        else
        {
            Eigen::JacobiSVD<Vector> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // Get the singular values and matrices U, V
            Vector singularValues = svd.singularValues();
            Vector U_ = svd.matrixU();
            Vector V_ = svd.matrixV();
            // Set small singular values to zero to handle numerical errors (e.g., due to zero eigenvalue)
            const double epsilon = 1e-12;
            for (int i = 0; i < singularValues.size(); ++i) {
                if (singularValues(i) < epsilon) {
                    singularValues(i) = 0.0;
                } else {
                    singularValues(i) = 1.0/std::sqrt(singularValues(i));
                }
            }
            M_pinv_sqrt = U_ * singularValues.asDiagonal() * V_.transpose();
        }
        
        Vector M_sol = thinQ(ind, Eigen::all) * K_sqrt * M_pinv_sqrt * K_sqrt;
        
        thinQ(ind, Eigen::all) = M_sol;

        return thinQ;
    }

	Variable ReducedQuintessential::RetractNewton(const Vector X) const
    {
        Vector U = X;

        double residual_sq = 1.0;
        constexpr double residual_sq_limit = 1e-8;
        int counter = 0;
        while (residual_sq > residual_sq_limit)
        {
            auto J = GetConstraintsJacobian(U); // 7 by 12*m_rank
            auto f = GetConstraintsVal(U);

            Eigen::VectorXd y = (J * J.transpose()).llt().solve(f);

            Eigen::VectorXd delta = - J.transpose()*y;
            
            for (size_t idx=0; idx<3; ++idx)
            {
                U(Eigen::all, idx) += delta(Eigen::seq(idx*m_rank*4, (idx+1)*m_rank*4-1));
            }

            counter++;
            residual_sq = delta.squaredNorm();
        }
        
        return U;
    }

    Vector ReducedQuintessential::GetConstraintsJacobian(const Vector X) const
    {
        Vector J = Vector::Zero(7, 12*m_rank);

        for (size_t idx = 0; idx<3; ++idx)
        {
            J(idx, Eigen::seq(4*idx*m_rank, 4*(idx+1)*m_rank-1)) = 2 * X(Eigen::all, idx);

            for (size_t jdx = idx+1; jdx<3; ++jdx)
            {
                J(idx+jdx+2, Eigen::seq(4*idx*m_rank, 4*(idx+1)*m_rank-1)) = X(Eigen::all, jdx);
                J(idx+jdx+2, Eigen::seq(4*jdx*m_rank, 4*(jdx+1)*m_rank-1)) = X(Eigen::all, idx);
            }
        }

        for (int idx_r = 0; idx_r<m_rank; ++idx_r)
        {
            J(6, 3 + 4*idx_r) = 2 * X(3 + 4*idx_r, 0);
            J(6, 4*m_rank + 3 + 4*idx_r) = 2 * X(3 + 4*idx_r, 1);
            J(6, 8*m_rank + 3 + 4*idx_r) = 2 * X(3 + 4*idx_r, 2);
        }

        return J;
    }

    Variable ReducedQuintessential::to_stiefel(const Variable &Y) const
    {
        // Y is (12) by (m_rank)
        // V should be (4*m_rank) by 3
        Variable V;
        V.resize(4*m_rank, 3);

        V(Eigen::all, 0) = Y(Eigen::seqN(0,4), Eigen::all).reshaped();
        V(Eigen::all, 1) = Y(Eigen::seqN(4,4), Eigen::all).reshaped();
        V(Eigen::all, 2) = Y(Eigen::seqN(8,4), Eigen::all).reshaped();

        return V;
    }
    
    // This function acts regardless of m_rank
    Variable ReducedQuintessential::to_bm(const Variable &V) const
    {
        // V is (4*m_rank) by 3
        // Y should be (12) by (m_rank)
        size_t r = V.rows() / 4;

        Variable Y;
        Y.resize(12, r);

        Y(Eigen::seqN(0,4), Eigen::all) = V(Eigen::all, 0).reshaped(4, r);
        Y(Eigen::seqN(4,4), Eigen::all) = V(Eigen::all, 1).reshaped(4, r);
        Y(Eigen::seqN(8,4), Eigen::all) = V(Eigen::all, 2).reshaped(4, r);

        return Y;
    }

} // end of namespace QME