#include "random_bearing_generator.hpp"

namespace QME 
{

RandomBearingGenerator::RandomBearingGenerator(const size_t N,
                                               const scalar_t pixel_noise,
                                               BearingGenParams params)
                                                : m_N(N), m_sigma(pixel_noise), m_params(params)
{
    m_t12 = vector3_t::Random().normalized() * m_params.cam_distance;
    m_t1 = -0.5 * m_t12;
    m_t2 =  0.5 * m_t12;

    m_R1 = matrix_t::Identity(3,3);
    m_R2 = generate_random_rotation();
    m_R12 = m_R1.transpose() * m_R2;

    calculate_essential_matrix();

    m_points.resize(3,N);
    m_bearings.reserve(N);
    m_bearings_noisy.reserve(N);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 generator(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<scalar_t> distribution(m_params.min_depth, m_params.max_depth);

    for (int idx=0; idx<N; idx++)
    {
        vector3_t p;
        
        auto dir = vector3_t::Random().normalized();
        p = distribution(generator) * dir;
        bearing_t v1, v2;

        v1 = m_R1.transpose() * (p-m_t1).normalized();
        v2 = m_R2.transpose() * (p-m_t2).normalized();

        bearing_t v1_n = add_bearing_noise(v1, pixel_noise);
        bearing_t v2_n = add_bearing_noise(v2, pixel_noise);
        
        m_points.col(idx) = p;
        m_bearings.push_back(std::make_pair(v1, v2));
        m_bearings_noisy.push_back(std::make_pair(v1_n, v2_n));
    }
}

measurements_t RandomBearingGenerator::get_bearings() const
{
    return m_bearings;
}

measurements_t RandomBearingGenerator::get_bearings_noisy() const
{
    return m_bearings_noisy;
}

rotation_t RandomBearingGenerator::get_relative_rotation() const
{
    return m_R12;
}

rotation_t RandomBearingGenerator::get_first_rotation() const
{
    return m_R1;
}

rotation_t RandomBearingGenerator::get_second_rotation() const
{
    return m_R2;
}

vector3_t RandomBearingGenerator::get_first_position() const
{
    return m_t1;
}

vector3_t RandomBearingGenerator::get_second_position() const
{
    return m_t2;
}

bearing_t RandomBearingGenerator::get_relative_bearing() const
{
    return m_t12;
}
matrix_t RandomBearingGenerator::get_essential_matrix() const
{
    return m_E;
}

void RandomBearingGenerator::calculate_essential_matrix()
{
    matrix_t tx = matrix_t::Zero(3,3);
    tx <<         0, -m_t12(2),  m_t12(1),
           m_t12(2),         0, -m_t12(0),
          -m_t12(1),  m_t12(0),         0;
    m_E = m_R1.transpose() * tx * m_R2; 
}

matrix_t RandomBearingGenerator::get_points() const
{
    return m_points;
}

std::pair<vector3_t,vector3_t> RandomBearingGenerator::get_camera_positions() const
{
    return std::make_pair(m_t1, m_t2);
}

std::pair<rotation_t,rotation_t> RandomBearingGenerator::get_camera_rotations() const
{
    return std::make_pair(m_R1, m_R2);
}

rotation_t RandomBearingGenerator::generate_random_rotation() const
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 generator(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<scalar_t> distribution(0, 0.5);
    Eigen::AngleAxisd rot(distribution(generator)*3.1415926535897932384, vector3_t::Random().normalized());
    return rot.matrix();
}

rotation_t RandomBearingGenerator::generate_random_rotation_uniform() const
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 generator(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<scalar_t> distribution(m_params.min_rotation, m_params.max_rotation);
    Eigen::AngleAxisd rot(distribution(generator), vector3_t::Random().normalized());
    return rot.matrix();
}

bearing_t RandomBearingGenerator::add_bearing_noise(bearing_t b_gt, scalar_t pixel_noise) const
{
    std::random_device rd; 
    std::mt19937 generator(rd());
    std::uniform_real_distribution<scalar_t> distribution(0.0, 1.0);

    bearing_t b_noisy;

    b_gt.normalize();

    // find good vector in normal plane based on good conditioning
    vector3_t inPlaneVector1, inPlaneVector2;
    inPlaneVector1.setZero();

    int idx_max = 0;
    scalar_t max_value = b_gt(0);
    if (b_gt(1)>max_value){
        idx_max = 1;
        max_value = b_gt(1);
    }
    if (b_gt(2)>max_value){
        idx_max = 2;
        max_value = b_gt(2);
    }

    if (idx_max == 0){
        inPlaneVector1 << -b_gt(1)/b_gt(0), 1.0, 0.0;
    }
    if (idx_max == 1) {
        inPlaneVector1 << 0.0, -b_gt(2)/b_gt(1), 1.0;
    }
    if (idx_max == 2) {
        inPlaneVector1 << 1.0, 0.0, -b_gt(0)/b_gt(2);
    }

    // normalize the in-plane vector
    inPlaneVector1.normalize();
    inPlaneVector2 = b_gt.cross(inPlaneVector1);

    double noiseX = pixel_noise * distribution(generator);
    double noiseY = pixel_noise * distribution(generator);

    b_noisy = m_params.focal_length * b_gt + noiseX * inPlaneVector1 + noiseY * inPlaneVector2;
    b_noisy.normalize();

    return b_noisy;
}

} // end of namespace QME