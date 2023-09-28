#include "problem.hpp"
#include "qme.hpp"
#include "random_bearing_generator.hpp"
#include "data_save_load.hpp"
#include <iostream>
#include <random>
#include "utils.hpp"
#include <filesystem>

void optimize_and_save_results(int N, int sigma, int rep, 
        std::string data_dir, std::string out_dir, bool random_init = false)
{
    std::string file_name = data_dir + 
                            std::to_string(N) +
                            "_" + 
                            std::to_string(sigma) +
                            "_" + 
                            std::to_string(rep)+".txt";

    QME::DataSaveLoad dsl(file_name);

    QME::BearingGenResult bg_result;

    if (dsl.file_exists())
    {
        // read them and print them here!
        bg_result = dsl.read_measurements();
    }
    else
    {
        QME::RandomBearingGenerator rbg(N, sigma);
        bg_result = rbg.get_result();
        dsl.write_measurements(rbg);
    }

    QME::measurements_t m = bg_result.bearings_noisy;

    std::string result_extention;

    QME::QMEOpts options;
    options.r0 = 2;
    options.verbose = false;
    options.grad_norm_tol = 1e-7;
    options.rel_func_decrease_tol = 1e-7;
    options.stepsize_tol = 5e-6;
    options.max_iterations = 1000;
    options.max_tCG_iterations = 1000;
    options.STPCG_kappa = 0.1;
    options.STPCG_theta = .5;
    options.min_eig_num_tol = 5e-4;

    if (random_init)
    {
        options.initialization = QME::Initialization::Random;
        result_extention = "random";
    }
    else
    {
        options.initialization = QME::Initialization::Eigen;
        result_extention = "eigen";
    }
    options.do_local_optimization = true;

    bool write_results = true;
    
    auto result = QME::QME(m, options);

    if (result.status == QME::SaddlePoint){
    std::cout << "Duration: " << result.total_computation_time_ms << " + "
                              << result.local_optimization_time_ms << " = "
                              << result.total_computation_time_ms+
                              result.local_optimization_time_ms << " |"
                              << " N: " << std::to_string(N)
                              << ", sigma: " << std::to_string(sigma)
                              << ", rep: " << std::to_string(rep) << std::endl;}
    
    /* std::cout << "Staircase solution objective value: " << std::endl << "     " << result.f_V << std::endl
              << "Staircase rounded solution objective value: " << std::endl << "     " << result.f_Vhat << std::endl
              << "Local optimization objective value: " << std::endl << "     " << result.f_Vhat_local << std::endl
              << "Lower bound on objective: "  << std::endl << "     " << result.f_min_bound << std::endl
              << "Number of staircase levels: " << result.staircase_times_ms.size() << std::endl; */
    if (options.do_local_optimization)
    {
        QME::matrix3_t essential_estimate = result.Vhat.topRows(3).transpose();
        auto errors = QME::get_rotation_and_translation_error(essential_estimate, bg_result.R12, bg_result.t12);

        QME::matrix3_t essential_estimate_local = result.Vhat_local.topRows(3).transpose();
        auto errors_local = QME::get_rotation_and_translation_error(essential_estimate_local, bg_result.R12, bg_result.t12);

        std::string result_file_name = out_dir + 
                                std::to_string(N) +
                                "_" + 
                                std::to_string(sigma) +
                                "_" + 
                                std::to_string(rep) + 
                                "_" + 
                                result_extention + ".txt";
        
        if (write_results)
        {
            dsl.write_results(result_file_name, result, errors, errors_local);
        }
    }
}

int main()
{
    // Create the data and result directories if they do not exist, in the source dir
    std::string data_dir = "../data/";
    std::string out_dir = "../result/";

    if (!std::filesystem::exists(data_dir))
    {
        std::filesystem::create_directories(data_dir);
    }

    if (!std::filesystem::exists(out_dir))
    {
        std::filesystem::create_directories(out_dir);
    }

    std::vector<int> N_vec = {10};
    std::vector<int> sigma_vec = {1};
    int number_of_repetetion = 5;

    bool random_initialization = false;
    
    for (auto N: N_vec)
    {
        for (auto sigma: sigma_vec)
        {
            for (size_t rep=0; rep<number_of_repetetion; rep++)
            {
                optimize_and_save_results(N, sigma, rep, data_dir, out_dir, random_initialization);
            }
            
        }
    }
    
}