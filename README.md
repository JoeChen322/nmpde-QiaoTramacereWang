# Simple README

## Compiling

To build the executable, make sure you have loaded the needed modules with

```bash
module load gcc-glibc dealii
```

Then run the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

The executable will be created into `build`, and can be executed through

```bash
./executable-name
```

## Result Verification Through Paraview

Here is a sample visualization of the results using Paraview:

- We set the parameters as follows:
  - `f = 0.0`
  - `g = 0.0`
  - `mu = 1.0`
  - `theta = 0.5`
  - `initial condition: u(x,0) = sin(pi*x)*sin(pi*y)*sin(pi*z); U'(x,0) = 0`
  
![Paraview](./src/OtherImplementation/Visual.png)

## More tests results and commands to run

The commands to run the various tests and scripts are as follows:

- `generate_square_meshes_by_N.py`: To use it we need to input the value N which is the number of divisions on each side of the square. It will generate a mesh file named `mesh-square-N.msh` in the `mesh` folder. To run it, use the following command:

  ```bash
  python3 generate_square_meshes_by_N.py --out-dir ../mesh --Ns 8 16 32 64 128
  ```

- `TimeConvergence.cpp`: To run the time convergence test, use the following command:

  ```bash
  mpirun -np 8 ./TimeConvergence ../mesh/mesh-square-400.msh 1 1.0 0.5 0.1 5
  ```

  The input arguments are: `mesh_file degree T theta initial_deltat num_refinements` and have default values so that they are not mandatory.

- `SpaceConvergence.cpp`: To run the space convergence test, use the following command:

  ```bash
  mpirun -np 8 ./SpaceConvergence ../mesh mesh-square 1 0.75 0.5 0.001 4 8 16 32 64 128
  ```

  The input arguments are: `mesh_dir mesh_base_name degree T theta deltat Ns...` and have default values so that they are not mandatory.

- `DissipationStudy.cpp`: To run the dissipation study, use the following command:

  ```bash
  mpirun -np 8 ./DissipationStudy ../mesh/mesh-square-64.msh 1 10.0 0.75 0.05 1 1
  ```

  The input arguments are: `mesh_file degree T theta deltat mode_x mode_y` and have default values so that they are not mandatory.

- `plot_time_convergence.py`: To plot the time convergence results, use the following command:

  ```bash
  python3 plot_time_convergence.py --csv ../build/time_convergence.csv --out ./Plots/time_convergence.png
  ```

- `plot_space_convergence.py`: To plot the space convergence results, use the following command:

  ```bash
  python3 plot_space_convergence.py --csv ../build/space_convergence.csv --out ./Plots/space_convergence.png
  ```

- `plot_dissipation.py`: To plot the dissipation results, use the following command:

  ```bash
  python3 plot_dissipation.py
  ```

  This will read the summary CSV file and generate plots in the specified directory.

## Notes
