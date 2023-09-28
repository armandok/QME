#include "problem.hpp"

namespace QME 
{

Problem::Problem() : m_rank(1)
{
    m_C = matrix_t::Zero(12,12);
}

Problem::Problem(const measurements_t &measurements) : m_rank(1), m_min_objective_bound(0)
{
    m_C = matrix_t::Zero(12,12);
    scalar_t N = measurements.size();

    for (auto bearings: measurements)
    {
        bearing_t b1 = bearings.first;
        bearing_t b2 = bearings.second;

        matrix_t v;
        v.resize(12,1);
        v << b1(0)*b2, 0, b1(1)*b2, 0, b1(2)*b2, 0;
        
        m_C += v * v.transpose()/N;
    }
}

void Problem::set_relaxation_rank(size_t rank)
{
    m_rank = rank;
    m_FE.set_rank(rank);
}

matrix_t Problem::sdp_matrix(const matrix_t &V) const
{
    matrix_t Y = m_FE.to_bm(V);
    return Y * Y.transpose();
}

matrix_t Problem::data_matrix_product(const matrix_t &Y) const
{
    return m_C * Y;
}

matrix_t Problem::data_matrix_product_stiefel(const matrix_t &V) const
{
    // C is 12 by 12, Y is 12 by r
    // V is 4r by 3

    matrix_t C_times_V;
    C_times_V.resize(V.rows(), V.cols());

    size_t rank = V.rows() / 4;

    for (size_t idx=0; idx<rank; idx++)  // iterate over rows of CV
    {
        for (size_t col=0; col<3; col++) // iterate over cols of CV
        {
            C_times_V(Eigen::seq(idx*4+0,idx*4+3),col) =
                m_C(Eigen::seq(col*4+0,col*4+3), Eigen::seq(0,3))*V(Eigen::seq(idx*4+0,idx*4+3),0)
               +m_C(Eigen::seq(col*4+0,col*4+3), Eigen::seq(4,7))*V(Eigen::seq(idx*4+0,idx*4+3),1)
               +m_C(Eigen::seq(col*4+0,col*4+3), Eigen::seq(8,11))*V(Eigen::seq(idx*4+0,idx*4+3),2);
        }
    }
    return C_times_V;
}

scalar_t Problem::evaluate_objective(const matrix_t &V) const
{
    /* matrix_t Y = m_FE.to_bm(V);
    return (Y.transpose() * data_matrix_product(Y)).trace(); */
    matrix_t CV = data_matrix_product_stiefel(V);
    return CV.cwiseProduct(V).sum();
}

matrix_t Problem::euclidean_gradient(const matrix_t &V) const
{
    /* matrix_t Y = m_FE.to_bm(V);
    return 2 * m_FE.to_stiefel( data_matrix_product(Y) ); */
    return 2 * data_matrix_product_stiefel(V);
}

matrix_t Problem::riemannian_gradient(const matrix_t &V, const matrix_t &nablaF_V) const
{
    return m_FE.Projection(V, nablaF_V);
}

matrix_t Problem::riemannian_gradient(const matrix_t &V) const
{
    return m_FE.Projection(V, euclidean_gradient(V));
}

matrix_t Problem::riemannian_hessian_vector_product(const matrix_t &V,
                                            const matrix_t &nablaF_V,
                                            const matrix_t &dotV) const
{
    //matrix_t nabla2F_V = 2 * m_FE.to_stiefel( data_matrix_product(m_FE.to_bm(dotV)) );
    matrix_t nabla2F_V = 2 * data_matrix_product_stiefel(dotV);
    return m_FE.EucHvToHv(V, nablaF_V, nabla2F_V, dotV);
    // EucHvToHv(const Variable& x, const Vector& egrad, const Vector& ehess, const Vector& x_dot)
}

matrix_t Problem::riemannian_Hessian_vector_product(const matrix_t &V,
                                            const matrix_t &dotV) const
{
    // matrix_t nabla2F_V = 2 * m_FE.to_stiefel( data_matrix_product(m_FE.to_stiefel(dotV)) );
    matrix_t nabla2F_V = 2 * data_matrix_product_stiefel(dotV);
    return m_FE.EucHvToHv(V, euclidean_gradient(V), nabla2F_V, dotV);
    // EucHvToHv(const Variable& x, const Vector& egrad, const Vector& ehess, const Vector& x_dot)
}

matrix_t Problem::tangent_space_projection(const matrix_t &V, const matrix_t &dotV) const
{
    return m_FE.Projection(V, dotV);
}

matrix_t Problem::retract(const matrix_t &V, const matrix_t &dotV) const
{
    return m_FE.Retract(V, dotV);
}

matrix_t Problem::retract_newton(const matrix_t &V, const matrix_t &dotV) const
{
    return m_FE.RetractNewton(V+dotV);
}

matrix_t Problem::round_solution(const matrix_t &V) const
{
    matrix_t Y = m_FE.to_bm(V);
    // rows of Y that contain entries of the Essential matrix
    std::vector<size_t> ind = {0,1,2,4,5,6,8,9,10};
    matrix_t Y_E = Y(ind, Eigen::all);
    matrix_t X_E = Y_E * Y_E.transpose();
    
    // Compute eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<matrix_t> solver(X_E);

    // Get the largest eigenvalue
    scalar_t largestEigenvalue = solver.eigenvalues()(8); // The eigenvalues are in ascending order

    // Get the corresponding (largest) eigenvector
    vector_t eig = solver.eigenvectors().col(8);

    // Normalize the eigenvector and scale to have 
    // the same Frobeneius norm as an Essential matrix
    eig = (eig * (std::sqrt(2)/eig.norm())).eval();

    
    matrix_t E;
    E.resize(3,3);
    E << eig(0), eig(3), eig(6),
         eig(1), eig(4), eig(7),
         eig(2), eig(5), eig(8);

    return project_essential(E); // TODO
}

 matrix_t Problem::project_essential(const matrix_t &E) const
 {
    Eigen::JacobiSVD<matrix_t> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Get the left and right singular vectors
    vector_t singularValues = svd.singularValues();
    matrix_t leftSingularVectors = svd.matrixU();
    matrix_t rightSingularVectors = svd.matrixV();

    matrix_t S = matrix_t::Identity(3,3);
    S(2,2) = 0;

    matrix_t V_proj = matrix_t::Zero(4,3);
    V_proj.block<3,3>(0,0) = leftSingularVectors * S * rightSingularVectors.transpose();
    V_proj.row(3) = rightSingularVectors.col(2).transpose();
    //std::cout << "Identity? :\n " << V_proj.transpose()*V_proj << std::endl;
    return V_proj;
 }

bool Problem::verify_solution(const matrix_t &V, scalar_t eta, std::vector<scalar_t> &theta_vec,
                              std::vector<vector_t> &x_vec) const
{
    matrix_t Y = m_FE.to_bm(V);
    matrix_t CX = m_C * Y * Y.transpose();

    matrix_t S = m_C;
    scalar_t s = 0;
    for (size_t idx=0; idx<3; idx++)
    {
        s = CX(Eigen::seq(idx*4,idx*4+3),Eigen::seq(idx*4,idx*4+3)).trace();
        S(Eigen::seq(idx*4,idx*4+3),Eigen::seq(idx*4,idx*4+3)) -= s * matrix_t::Identity(4,4);
        for (size_t jdx=idx+1; jdx<3; jdx++)
        {
            s = 0.5*( CX(Eigen::seq(idx*4,idx*4+3),Eigen::seq(jdx*4,jdx*4+3)).trace() + 
                      CX(Eigen::seq(jdx*4,jdx*4+3),Eigen::seq(idx*4,idx*4+3)).trace());
            S(Eigen::seq(idx*4,idx*4+3),Eigen::seq(jdx*4,jdx*4+3)) -= s * matrix_t::Identity(4,4);
            S(Eigen::seq(jdx*4,jdx*4+3),Eigen::seq(idx*4,idx*4+3)) -= s * matrix_t::Identity(4,4);
        }
    }

    S += eta * matrix_t::Identity(12,12);

    bool is_PSD = true;

    // Compute eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<matrix_t> solver(S);

    // Get the eigenvalues
    vector_t eigenvalues = solver.eigenvalues();

    // Find the negative eigenvalues and their corresponding eigenvectors
    for (size_t idx = 0; idx < eigenvalues.size(); ++idx)
    {
        if (eigenvalues(idx) < 0)
        {
            is_PSD = false;
            
            theta_vec.push_back(eigenvalues(idx));
            x_vec.push_back(solver.eigenvectors().col(idx));
        }
    }
    return is_PSD;
}

matrix_t Problem::random_sample() const
{
    return m_FE.RandomManifold();
}

matrix_t Problem::data_matrix_initialization()
{
    if (m_C.size() == 0)
    {
        std::cout << "WARNING: Cannot perform data mat initialization since data matrix is not set" << std::endl;
        return m_FE.RandomManifold();
    }

    matrix_t C = matrix_t::Zero(9,9);
    std::vector<size_t> ind = {0,1,2,4,5,6,8,9,10};
    
    C = m_C(ind, ind);
    
    // Create a SelfAdjointEigenSolver object.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C);

    // Get the minimum eigenvalue and eigenvector.
    double min_eigenvalue = solver.eigenvalues()[0];
    Eigen::VectorXd min_eigenvector = solver.eigenvectors().col(0);

    m_min_objective_bound = min_eigenvalue*2;

    scalar_t scale = std::sqrt(2) / min_eigenvector.norm();
    min_eigenvector *= scale;

    matrix_t E;
    E.resize(3,3);
    E << min_eigenvector(0), min_eigenvector(3), min_eigenvector(6),
         min_eigenvector(1), min_eigenvector(4), min_eigenvector(7),
         min_eigenvector(2), min_eigenvector(5), min_eigenvector(8);

    matrix_t V = matrix_t::Zero(4*m_rank,3);
    V.topRows(4) = project_essential(E);
    return V;
}

scalar_t Problem::get_objective_min_bound() const
{
    return m_min_objective_bound;
}

} // end of namespace QME