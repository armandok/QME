# QME - ICCV 2023
The Quintessential Matrix Estimation (QME) repository solves the non-minimal Essential matrix estimation problem as described in the paper [Essential Matrix Estimation using Convex Relaxations in Orthogonal Space](https://openaccess.thecvf.com/content/ICCV2023/html/Karimian_Essential_Matrix_Estimation_using_Convex_Relaxations_in_Orthogonal_Space_ICCV_2023_paper.html).

## C++ code
The C++ version of QME uses the Optimization library by David Rosen.
To run the code, first, clone the repository with its submodule. Then, run the following commands in the terminal.
```
git clone --recurse-submodules https://github.com/armandok/QME.git
cd QME
mkdir build && cd build
cmake ..
make
```

Then run the main application by ``` ./main ``` and check the *data* and *result* folders.

## MATLAB code
To run the MATLAB code, you must have [Manopt](https://www.manopt.org/) installed and added to the path. Then simply run *main.m* with MATLAB.
For the comparison of the Riemmanian staircase method with an SDP solver, you need to have [YALMIP](https://yalmip.github.io/) installed with an SDP solver like [SDPT3](https://www.math.cmu.edu/~reha/sdpt3.html).
