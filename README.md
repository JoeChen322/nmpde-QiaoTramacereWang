# Commands for compiling and running the code for this project

## Group members

We select the Project 2: Solving the wave equation. We implement both time and space convergence tests, as well as dissipation and dispersion studies for the wave equation solver using the finite element method and the theta method, with a lot adjustable parameters.

- Yanlong Wang
- Zhaochen Qiao
- Luca Tramacere

## Overview of the project structure

The project is organized as follows:

```PDEProject/nmpde-QiaoTramacereWang/
├── CMakeLists.txt
├── README.md
├── include              # Header files
│   ├── Wave.h
│   └── IOUtils.hpp
├── src                  # Source code files      
│   ├── TimeConvergence.cpp
│   ├── SpaceConvergence.cpp
│   ├── DissipationStudy.cpp
│   ├── DispersionOmegaDtStudy.cpp
│   ├── DispersionSpatialStudy.cpp
│   ├── Main.cpp
│   └── Wave.cpp
├── meshes               # Folder to store and load mesh files
├── results              # Folder to store results
│   ├── convergence      # Subfolder for convergence test results
│   ├── dissipation      # Subfolder for dissipation study results      
│   ├── dispersion       # Subfolder for dispersion study results 
│   └── plots            # Subfolder for plots
├── scripts              # Folder for scripts to generate meshes and plot results
│   ├── generate_square_meshes_by_N.py
│   ├── plot_time_convergence.py
│   ├── plot_space_convergence.py
│   ├── plot_dissipation.py
│   ├── plot_dispersion_time.py
│   ├── plot_dispersion_space.py 
│   └── ...
│── build                # Build folder (created after compiling)    
```

## Compiling

To build the executable, make sure you have loaded the needed modules with

```bash
module load gcc-glibc dealii
```

Then run the following commands within the `nmpde-QiaoTramacereWang` directory:

```bash
mkdir build
cd build
cmake ..
make
```

The executable will be created into `build`, and can be executed through

```bash
mpirun -np 4 ./executable-name (optional parameters)
```

## Commands to run various tests and scripts

- `generate_square_meshes_by_N.py`: To use it we need to input the value N which is the number of divisions on each side of the square. It will generate a mesh file named `mesh-square-N.msh` into the `meshes` folder. To run it, use the following command:

  ```bash
  python3 generate_square_meshes_by_N.py --out-dir ../meshes --Ns 8 16 32 64 128
  ```

- `TimeConvergence.cpp`: To run the time convergence test, use the following command:

  ```bash
  mpirun -np 8 ./TimeConvergence ../meshes/mesh-square-400.msh 1 1.0 0.5 0.1 5
  ## mpirun -np 8 ./TimeConvergence (with default args)
  ```

  The input arguments are: `mesh_file degree T theta initial_deltat num_refinements`. The output `time_convergence.csv` is written to the `results/convergence` folder.

- `SpaceConvergence.cpp`: To run the space convergence test, use the following command:

  ```bash
  mpirun -np 8 ./SpaceConvergence ../meshes mesh-square 1 0.75 0.5 0.001 4 8 16 32 64 128
  ## mpirun -np 8 ./SpaceConvergence (with default args)
  ```

  The input arguments are: `mesh_dir mesh_base_name degree T theta deltat Ns...`. The output `space_convergence.csv` is written to the `results/convergence` folder.

- `DissipationStudy.cpp`: To run the dissipation study, use the following command:

  ```bash
  mpirun -np 8 ./DissipationStudy ../meshes/mesh-square-64.msh 1 10.0 10
  ## mpirun -np 8 ./DissipationStudy (with default args)
  ```

  The input arguments are: `mesh_file degree T stride`. The output `dissipation_summary.csv` is written to the `results/dissipation` folder.

- `DispersionOmegaDtStudy.cpp`: To run the dispersion study, use the following command:

  ```bash
  mpirun -np 8 ./DispersionOmegaDtStudy ../meshes/mesh-square-64.msh 1 10.0 10
  ## mpirun -np 8 ./DispersionOmegaDtStudy (with default args)
  ```

  The input arguments are: `mesh_file degree T stride`. The output `dispersion_summary.csv` is written to the `results/dispersion` folder.
  
- `DispersionSpatialStudy.cpp`: To run the dissipation study, use the following command:

  ```bash
  mpirun -np 8 ./DispersionSpatialStudy ../meshes mesh-square 1 10.0 0.001 1 1 8 16 32 64 
  ## mpirun -np 8 ./DispersionSpatialStudy (with default args)
  ```

  The input arguments are: `mesh_dir mesh_base_name degree T deltat m n Ns...`. The output `dispersion_spatial.csv` is written to the `results/dispersion` folder.

- `plot_time_convergence.py`: To plot the time convergence results, use the following command:

  ```bash
  python3 plot_time_convergence.py --csv ../results/convergence/time_convergence.csv --out ../results/plots/time_convergence.png
  ```

- `plot_space_convergence.py`: To plot the space convergence results, use the following command:

  ```bash
  python3 plot_space_convergence.py --csv ../results/convergence/space_convergence.csv --out ../results/plots/space_convergence.png
  ```

- `plot_dissipation.py`: To plot the dissipation results, use the following command:

  ```bash
  python3 plot_dissipation.py
  ```

  This will read the summary CSV file from `results/dissipation/dispersion_summary.csv` and save the plots to `results/plots`.

- `plot_dispersion_time.py`: To plot the dispersion results, use the following command:

  ```bash
  python3 plot_dispersion_time.py
  ```

  This will read the summary CSV file from `results/dispersion/dispersion_summary.csv` and save the plots to `results/plots`.

- `plot_dispersion_space.py`: To plot the dispersion spatial results, use the following command:

  ```bash
  python3 plot_dispersion_space.py
  ```

  This will read the summary CSV file from `results/dispersion/dispersion_spatial.csv` and save the plots to `results/plots`.

## Notes

- Some input and output folders locations are written inside the files (e.g. loading meshes), so make sure to keep the structure same as shown in this README when running the code.

- Some default setting is to avoid too long running time. User can modify the input arguments to increase the accuracy as needed.

- Because we put all the executables into one make file, when you run `make`, all the executables will be built. So that when you compile them for the first time, it may take a bit longer. But subsequent `make` commands will be faster as only changed files will be recompiled.

- The codes for evaluation is at this main branch. We keep the other branches for development history.