cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)

project(qme)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 3.4 REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add the Eigen library
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${EIGEN3_INCLUDE_DIR})

# Find Optimization library
set(OPTIMIZATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Optimization/ CACHE PATH "Path to top-level directory of Optimization library (the one containing CMakeLists.txt)")

# Run the Optimization package's CMake file
add_subdirectory(${OPTIMIZATION_DIR})

set(QME_HEADERS
    manifold.hpp
    defs.hpp
    reduced_quintessential.hpp
    random_bearing_generator.hpp
    problem.hpp
    qme.hpp
    stopwatch.hpp
    data_save_load.hpp
    utils.hpp
)

set(QME_SOURCES
    manifold.cpp
    reduced_quintessential.cpp
    random_bearing_generator.cpp
    problem.cpp
    qme.cpp
    data_save_load.cpp
)
add_library(${PROJECT_NAME} ${QME_HEADERS} ${QME_SOURCES})

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} Optimization eigen)

# Link the executable against the qme and eigen libraries
add_executable(main main.cpp)
target_link_libraries (main PRIVATE ${PROJECT_NAME})