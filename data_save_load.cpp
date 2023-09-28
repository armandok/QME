#include "data_save_load.hpp"
#include <filesystem>
#include <iostream>
#include <fstream>

namespace QME 
{

DataSaveLoad::DataSaveLoad(std::string file_name): m_file_exists(std::filesystem::exists(file_name))
{
    m_filename = file_name;
}

bool DataSaveLoad::file_exists()
{
    return m_file_exists;
}

BearingGenResult DataSaveLoad::read_measurements()
{
    BearingGenResult result;

    std::ifstream ifs(m_filename);
    std::string line;

    // Helper function to read a block of measurements
    auto read_measurements_block = [&ifs, &line, this]() 
    {
        measurements_t measurements;
        while (std::getline(ifs, line) && !std::isalpha(line[0]))
        {
            if (line == "")
            {
                continue;
            }
            bearing_t b1 = line_to_bearing(line).normalized();
            std::getline(ifs, line);
            bearing_t b2 = line_to_bearing(line).normalized();
            measurements.push_back(std::make_pair(b1, b2));
        }
        return measurements;
    };

    // Find and read noisy bearings
    while (std::getline(ifs, line) && line != "Noisy bearings: ");
    result.bearings_noisy = read_measurements_block();

    // Find and read actual bearings
    assert(line == "Actual bearings: ");
    result.bearings = read_measurements_block();

    // Read other data
    assert(line == "E: ");
    result.E = read_3_by_N_mat(ifs, 3);
    assert(std::getline(ifs, line) && line == "R1: ");
    result.R1 = read_3_by_N_mat(ifs, 3);
    assert(std::getline(ifs, line) && line == "R2: ");
    result.R2 = read_3_by_N_mat(ifs, 3);
    assert(std::getline(ifs, line) && line == "t1: ");
    result.t1 = read_3_by_N_mat(ifs, 1);
    assert(std::getline(ifs, line) && line == "t2: ");
    result.t2 = read_3_by_N_mat(ifs, 1);
    assert(std::getline(ifs, line) && line == "Points: ");
    result.points = read_3_by_N_mat(ifs, result.N);

    // Read Sigma
    assert(std::getline(ifs, line) && line == "Sigma: ");
    assert(ifs >> result.sigma);
    std::getline(ifs, line);
    std::getline(ifs, line);
    
    // Read N
    assert(std::getline(ifs, line) && line == "N: ");
    size_t N;
    assert(ifs >> N && N == result.bearings.size());
    result.N = N;

    // Calculate t12 and R12
    result.t12 = (result.t2 - result.t1).normalized();
    result.R12 = result.R1.transpose() * result.R2;

    return result;
}


bool DataSaveLoad::write_measurements(const RandomBearingGenerator &rbg)
{
    // RandomBearingGenerator
    measurements_t measurements_noisy = rbg.get_bearings_noisy();
    std::ofstream of(m_filename);
    if(of.is_open())
    {
        of<<"Noisy bearings: "<<std::endl;
        for (auto measurement: measurements_noisy)
        {
            auto b1 = measurement.first;
            auto b2 = measurement.second;
            of<<b1.transpose()<<std::endl;
            of<<b2.transpose()<<std::endl<<std::endl;
        }

        measurements_t measurements = rbg.get_bearings();
        of<<"Actual bearings: "<<std::endl;
        for (auto measurement: measurements)
        {
            auto b1 = measurement.first;
            auto b2 = measurement.second;
            of<<b1.transpose()<<std::endl;
            of<<b2.transpose()<<std::endl<<std::endl;
        }

        of<<"E: "<<std::endl << rbg.get_essential_matrix()<<std::endl<<std::endl;
        of<<"R1: "<<std::endl << rbg.get_first_rotation()<<std::endl<<std::endl;
        of<<"R2: "<<std::endl << rbg.get_second_rotation()<<std::endl<<std::endl;
        of<<"t1: "<<std::endl << rbg.get_first_position()<<std::endl<<std::endl;
        of<<"t2: "<<std::endl << rbg.get_second_position()<<std::endl<<std::endl;
        of<<"Points: "<<std::endl << rbg.get_points()<<std::endl<<std::endl;
        of<< "Sigma: "<<std::endl << rbg.get_sigma()<<std::endl<<std::endl;
        of<< "N: "<<std::endl << measurements.size()<<std::endl<<std::endl;

        of.flush();
        of.close();
    }
    else
    {
        std::cerr<<"Failed to open file : " << m_filename <<std::endl;
        return false;
    }
    return true;
}

bool DataSaveLoad::write_results(
    const std::string file_name,
    const QMEResult result,
    const std::pair<scalar_t,scalar_t> rot_tra_err,
    const std::pair<scalar_t,scalar_t> rot_tra_err_lcl) const
{
    if (std::filesystem::exists(file_name))
    {
        return false;
    }

    std::ofstream of(file_name);
    if(of.is_open())
    {
        of << "Rot error (deg): " <<std::endl<< rot_tra_err.first <<std::endl<<std::endl;
        of << "Translation error (deg): " <<std::endl<< rot_tra_err.second <<std::endl<<std::endl;
        of << "Rot error local (deg): " <<std::endl<< rot_tra_err_lcl.first <<std::endl<<std::endl;
        of << "Translation error local (deg): " <<std::endl<< rot_tra_err_lcl.second <<std::endl<<std::endl;
        
        of << "Optimal Quintessential matrix: " <<std::endl<<result.Vhat <<std::endl<<std::endl;
        of << "Optimal objective value: " <<std::endl<<result.f_Vhat <<std::endl<<std::endl;
        of << "Computation time (us): " <<std::endl<<result.total_computation_time_ms <<std::endl<<std::endl;

        of << "Optimal+Local Quintessential matrix: " <<std::endl<<result.Vhat_local <<std::endl<<std::endl;
        of << "Optimal+Local objective value: " <<std::endl<<result.f_Vhat_local <<std::endl<<std::endl;
        of << "Computation+Local time (us): " <<std::endl<<result.local_optimization_time_ms <<std::endl<<std::endl;
        
        of << "Optimal Staircase matrix: " <<std::endl<<result.Vopt <<std::endl<<std::endl;

        of << "Objective lower bound: " <<std::endl<<result.f_min_bound <<std::endl<<std::endl;
        of << "Objective upper bound: " <<std::endl<<result.f_max_bound <<std::endl<<std::endl;

        of << "Number of stairs: " <<std::endl<<result.staircase_times_ms.size() <<std::endl<<std::endl;

        of << "Optimization time per stair (ms): " <<std::endl;
        for (auto time: result.staircase_times_ms)
        {
            of << time <<std::endl;
        }
        of << std::endl;

        of << "Optimization termination status: " << std::endl << result.status<<std::endl<<std::endl;

        of.flush();
        of.close();
    }
    else
    {
        std::cerr<<"Failed to open file : " << file_name <<std::endl;
        return false;
    }
    return true;
}

bool DataSaveLoad::write_results_sdp(const std::string file_name,
    const matrix3_t E,
    const matrix_t X,
    const std::pair<scalar_t, scalar_t> errors,
    const scalar_t alg_error,
    const unsigned int time) const
{
    if (std::filesystem::exists(file_name))
    {
        return false;
    }

    std::ofstream of(file_name);
    if(of.is_open())
    {
        of << "Rot error (deg): " <<std::endl<< errors.first <<std::endl<<std::endl;
        of << "Translation error (deg): " <<std::endl<< errors.second <<std::endl<<std::endl;
                
        of << "Optimal Essential matrix: " <<std::endl<< E <<std::endl<<std::endl;
        of << "Optimal objective value: " <<std::endl<< alg_error <<std::endl<<std::endl;
        
        of << "Computation time (us): " <<std::endl<< time <<std::endl<<std::endl;

        of << "Optimal PSD matrix: " <<std::endl<< X <<std::endl<<std::endl;

        of.flush();
        of.close();
    }
    else
    {
        std::cerr<<"Failed to open file : " << file_name <<std::endl;
        return false;
    }
    return true;
}


bearing_t DataSaveLoad::line_to_bearing(std::string s)
{
    scalar_t a,b,c;

    std::istringstream iss(s);
    bearing_t bearing;
    iss >> a >> b >> c;
    bearing[0] = a;
    bearing[1] = b;
    bearing[2] = c;
    return bearing;
}

matrix_t DataSaveLoad::read_3_by_N_mat(std::ifstream &ifs, size_t N)
{
    matrix_t mat = matrix_t::Zero(3, N);
    std::string line;
    for (size_t idx=0; idx<3; idx++)
    {
        std::getline( ifs, line );
        std::istringstream iss(line);
        for (size_t jdx=0; jdx<N; jdx++)
        {
            scalar_t val;
            iss >> val;
            mat(idx,jdx) = val;
        }
    }
    std::getline( ifs, line );
    assert(line.length() == 0);

    return mat;
}

} // end of namespace QME