#pragma once

#include "defs.hpp"
#include <random>

namespace QME 
{

struct BearingGenResult
{
    size_t N;
    vector3_t t1;
    vector3_t t2;
    bearing_t t12;
    rotation_t R1;
    rotation_t R2;
    rotation_t R12;
    matrix3_t E;
    matrix_t points;
    measurements_t bearings;
    measurements_t bearings_noisy;
    scalar_t sigma;
};

struct BearingGenParams
{
    scalar_t min_depth = 2.0;
    scalar_t max_depth = 8.0;

    scalar_t min_rotation = 0.0;
    scalar_t max_rotation = 0.5 * M_PI;

    scalar_t cam_distance = 1.0;

    scalar_t focal_length = 800;
};

class RandomBearingGenerator  {
public:
    RandomBearingGenerator(const size_t N, const scalar_t pixel_noise = 0.0, BearingGenParams params = BearingGenParams());

    measurements_t get_bearings() const;
    measurements_t get_bearings_noisy() const;
    rotation_t get_relative_rotation() const;
    bearing_t get_relative_bearing() const;
    rotation_t get_first_rotation() const;
    rotation_t get_second_rotation() const;
    vector3_t get_first_position() const;
    vector3_t get_second_position() const;
    matrix_t get_essential_matrix() const;
    matrix_t get_points() const; // Returns a 3 * N matrix containing feature point 3D positions
    std::pair<vector3_t,vector3_t> get_camera_positions() const;
    std::pair<rotation_t,rotation_t> get_camera_rotations() const;
    scalar_t get_sigma() const {return m_sigma;};

    BearingGenResult get_result() const
    {
        BearingGenResult result;
        result.N = m_N;
        result.t1 = m_t1;
        result.t2 = m_t2;
        result.t12 = m_t12;
        result.R1 = m_R1;
        result.R2 = m_R2;
        result.R12 = m_R12;
        result.E = m_E;
        result.points = m_points;
        result.bearings = m_bearings;
        result.bearings_noisy = m_bearings_noisy;
        result.sigma = m_sigma;
        return result;
    };

    ~RandomBearingGenerator() {}

private:
    rotation_t generate_random_rotation() const;
    rotation_t generate_random_rotation_uniform() const;
    bearing_t add_bearing_noise(bearing_t b_gt, scalar_t pixel_noise) const;
    void calculate_essential_matrix();
    size_t m_N;
    vector3_t m_t1;
    vector3_t m_t2;
    bearing_t m_t12;
    rotation_t m_R1;
    rotation_t m_R2;
    rotation_t m_R12;
    matrix3_t m_E;
    matrix_t m_points;
    measurements_t m_bearings;
    measurements_t m_bearings_noisy;
    scalar_t m_sigma;

    BearingGenParams m_params;
};



} // end of namespace QME