#pragma once

#include <iostream>
#include <string>
#include "defs.hpp"
#include "random_bearing_generator.hpp"
#include "qme.hpp"

namespace QME 
{

class DataSaveLoad  {
public:
    // file_name should follow the convention N_noisePixel_rep
    DataSaveLoad(std::string file_name); // read from file
    bool file_exists();
    BearingGenResult read_measurements();
    bool write_measurements(const RandomBearingGenerator &rbg);

    bool write_results(
        const std::string file_name,
        const QMEResult result,
        const std::pair<scalar_t,scalar_t> rot_tra_err,
        const std::pair<scalar_t,scalar_t> rot_tra_err_lcl) const;

    bool write_results_sdp(
        const std::string file_name,
        const matrix3_t E,
        const matrix_t X_sol,
        const std::pair<scalar_t, scalar_t> errors,
        const scalar_t alg_error, 
        const unsigned int time) const;


private:
    bearing_t line_to_bearing(std::string);
    // matrix3_t read_3_by_3_mat(std::ifstream &ifs);
    matrix_t read_3_by_N_mat(std::ifstream &ifs, size_t N);
    bool m_file_exists;
    std::string m_filename;
};

} // end of namespace QME