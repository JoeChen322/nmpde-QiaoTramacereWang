# AI Coding Agent Instructions for Wave Equation Solver Project

## Project Overview

This is a **numerical PDE solver** implementing the **wave equation** in 1D and 2D using the **deal.II finite element library** (v9.5.1) with a **Continuous Galerkin** discretization scheme. The project demonstrates convergence testing via Method of Manufactured Solutions (MMS) and supports both serial and MPI-parallel computation.

---

## Mathematical Formulation

### Strong Form (Problem Statement)

Find $u : \Omega \times [0,T] \rightarrow \mathbb{R}$ such that:

$$
\begin{cases}
\displaystyle \frac{\partial^2 u}{\partial t^2} - \Delta u = f & \text{in } \Omega \times (0,T), \\[6pt]
u = 0 & \text{on } \partial\Omega \times (0,T), \\[4pt]
u(\cdot,0) = u_0, \quad \displaystyle \frac{\partial u}{\partial t}(\cdot,0) = u_1 & \text{in } \Omega.
\end{cases}
$$

### Weak Formulation

Let $V = H_0^1(\Omega)$. For all $t \in (0,T)$, find $u(t) \in V$ such that $u(0) = u_0$, $\frac{\partial u}{\partial t}(0) = u_1$, and:

$$
\int_\Omega \frac{\partial^2 u}{\partial t^2} v \, dx + \underbrace{\int_\Omega \nabla u \cdot \nabla v \, dx}_{a(u,v)} = \underbrace{\int_\Omega f v \, dx}_{F(v)} \quad \forall v \in V
$$

### Semi-Discrete FEM Formulation

Introduce mesh $\mathcal{T}_h$ over $\Omega$ and finite element space $V_h = V \cap X_h^r(\Omega)$ of degree $r$. The semi-discrete problem seeks $u_h(t) \in V_h$:

$$
\int_\Omega \frac{\partial^2 u_h}{\partial t^2} v_h \, dx + a(u_h(t), v_h) = F(v_h) \quad \forall v_h \in V_h
$$

Using basis functions $\{\varphi_i\}_{i=1}^{N_h}$, this becomes the ODE system:

$$
M \frac{d^2 \mathbf{u}}{dt^2}(t) + A \mathbf{u}(t) = \mathbf{f}(t)
$$

where:
- $\mathbf{u}(t) = (U_1(t), \ldots, U_{N_h}(t))^T$ — vector of nodal coefficients
- $M_{ij} = \int_\Omega \varphi_i \varphi_j \, dx$ — **mass matrix**
- $A_{ij} = \int_\Omega \nabla \varphi_i \cdot \nabla \varphi_j \, dx$ — **stiffness matrix**
- $f_i(t) = \int_\Omega f(x,t) \varphi_i(x) \, dx$ — **load vector**

### Fully Discrete θ-Scheme

Partition $[0,T]$ into $N_T$ sub-intervals of width $\Delta t$. Introduce velocity $\mathbf{w}^n \approx \frac{d\mathbf{u}}{dt}(t_n)$. The θ-scheme reads:

$$
\begin{cases}
\dfrac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\Delta t} = (1-\theta)\mathbf{w}^n + \theta\mathbf{w}^{n+1}, \\[10pt]
M \dfrac{\mathbf{w}^{n+1} - \mathbf{w}^n}{\Delta t} + \theta A \mathbf{u}^{n+1} + (1-\theta) A \mathbf{u}^n = \theta \mathbf{f}^{n+1} + (1-\theta)\mathbf{f}^n
\end{cases}
$$

**Reduced to single linear system** for $\mathbf{u}^{n+1}$:

$$
\bigl(M + \theta^2 \Delta t^2 A\bigr) \mathbf{u}^{n+1} = \bigl(M - \theta(1-\theta)\Delta t^2 A\bigr) \mathbf{u}^n + \Delta t M \mathbf{w}^n + \theta^2 \Delta t^2 \mathbf{f}^{n+1} + \theta(1-\theta) \Delta t^2 \mathbf{f}^n
$$

**Velocity update**:

$$
\mathbf{w}^{n+1} = \frac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\theta \Delta t} - \frac{1-\theta}{\theta} \mathbf{w}^n
$$

### Special Cases of θ-Scheme

| Method | θ | System |
|--------|---|--------|
| **Forward Euler** (explicit) | 0 | $\mathbf{u}^{n+1} = \mathbf{u}^n + \Delta t \mathbf{w}^n$ |
| **Backward Euler** (implicit) | 1 | $(M + \Delta t^2 A)\mathbf{u}^{n+1} = M\mathbf{u}^n + \Delta t M \mathbf{w}^n + \Delta t^2 \mathbf{f}^{n+1}$ |
| **Crank-Nicolson** | ½ | $(M + \frac{\Delta t^2}{4} A)\mathbf{u}^{n+1} = (M - \frac{\Delta t^2}{4} A)\mathbf{u}^n + \Delta t M \mathbf{w}^n + \frac{\Delta t^2}{4}(\mathbf{f}^{n+1} + \mathbf{f}^n)$ |

### Energy Analysis

**Continuous energy** (conserved for $f \equiv 0$):

$$
E(t) = \frac{1}{2} \left\|\frac{\partial u}{\partial t}\right\|_{L^2}^2 + \frac{1}{2} \|\nabla u\|_{L^2}^2
$$

**Discrete energy**:

$$
E^n = \frac{1}{2} (\mathbf{w}^n)^\top M \mathbf{w}^n + \frac{1}{2} (\mathbf{u}^n)^\top A \mathbf{u}^n
$$

**Energy identity for θ-scheme** (homogeneous case $f \equiv 0$):

$$
E^{n+1} - E^n = \frac{\Delta t}{2}(2\theta - 1) \bigl[ (\mathbf{u}^n)^\top A \mathbf{w}^{n+1} - (\mathbf{u}^{n+1})^\top A \mathbf{w}^n \bigr]
$$

**Key properties**:
- **θ = ½ (Crank-Nicolson)**: Exactly conserves discrete energy ($E^{n+1} = E^n$)
- **θ > ½**: Energy-dissipative (numerical damping)
- **θ < ½**: Can be unstable without very small $\Delta t$

---

## Architecture & Design Patterns

### Core Component Structure

The codebase uses a **unified MPI-parallel Wave solver** class:

1. **Wave** ([include/Wave.hpp](include/Wave.hpp), [src/Wave.cpp](src/Wave.cpp))
   - Main solver class for 2D wave equation (dim=2)
   - Uses `FE_SimplexP<dim>` (triangular elements) on simplex meshes
   - MPI-parallel via Trilinos linear algebra (TrilinosWrappers::MPI::Vector, etc.)
   - Implements **θ-scheme time stepping** with configurable θ parameter
   - Assembles system matrix $(M + \theta^2 \Delta t^2 A)$ for implicit solve
   - Built-in energy computation and VTK output

2. **Inner Function Classes** (defined in Wave.hpp):
   - `FunctionMu` - Wave speed coefficient μ(x)
   - `ForcingTerm` - Source term f(x,t)
   - `InitialValuesU` / `InitialValuesV` - Initial displacement u₀(x) and velocity u₁(x)
   - `BoundaryValuesU` / `BoundaryValuesV` - Dirichlet boundary data
   - `ExactSolutionU` / `ExactSolutionV` - Analytical solution for convergence tests

### Data Flow Pattern

```
Wave class (defines problem, mesh, solver)
    ├→ setup() [load mesh, assemble M, A matrices]
    ├→ solve() [θ-scheme time-stepping loop]
    │    ├→ assemble_rhs(time)
    │    ├→ solve_time_step() - solve (M + θ²Δt²A)u^{n+1} = RHS
    │    └→ output(step) [write VTK files]
    └→ compute_*_error() [L² and H¹ error norms]
```

## Build & Test Workflow

### Build System (CMake)
- **Required**: MPI, Boost, deal.II (≥9.3.1)
- **Configuration**: `cmake-common.cmake` handles MPI/Boost/deal.II setup

### Quick Build
```bash
./build.sh  # Clean rebuild with parallel compilation (-j)
# OR manually:
mkdir build && cd build
cmake ..
make -j
```

### Executables
All built in `build/` directory:
- **WaveSolver** - Main 2D wave simulation
- **TimeConvergence** - Temporal convergence study (Δt refinement)
- **SpaceConvergence** - Spatial convergence study (h refinement)
- **DissipationStudy** - θ-scheme energy dissipation analysis

### Running Simulations
```bash
cd build
mpirun -np 4 ./WaveSolver                    # Main solver with 4 MPI processes
mpirun -np 4 ./TimeConvergence               # Time convergence study
mpirun -np 4 ./SpaceConvergence ../meshes mesh-square 1 1.0 0.5 0.001 20 40 80
mpirun -np 4 ./DissipationStudy              # Dissipation study
```

## Critical Patterns & Conventions

### Mesh Generation
Generate meshes using the Python script:
```bash
cd scripts
python generate_square_meshes_by_N.py
# Creates meshes/mesh-square-N.msh for various N values
```

### Configuration Pattern
Wave class constructor and setters for parametrization:
```cpp
Wave problem(mesh_file, degree, T, deltat, theta);
problem.set_output_interval(10);      // Output every 10 steps
problem.set_output_directory("./");   // Output directory
problem.set_mode(m, n);               // Eigenmode (m,n) for initial condition
```

### Output & Analysis
- **VTK output**: Paraview-compatible `.vtu`/`.pvtu` files with displacement and energy fields
- **Energy calculation**: Built-in `compute_total_energy()` for stability verification
- **Error computation**: `compute_L2_error()` and `compute_H1_error()` for convergence tests
- **Scripts** ([scripts/](scripts/)): Python plotting for convergence and dissipation analysis

## Key Developer Workflows

### Modifying Solver Behavior
1. Core logic: `Wave::solve_time_step()` (θ-scheme time integration)
2. Matrix assembly: `assemble_matrices()` - builds mass and stiffness matrices
3. RHS assembly: `assemble_rhs(time)` - builds forcing and solution-dependent terms
4. System matrix: $(M + \theta^2 \Delta t^2 A)$ assembled once, reused each step
5. Initial/boundary conditions: Modify inner classes in Wave.hpp

### Adding New Features
- **Different initial conditions**: Modify `InitialValuesU`/`InitialValuesV` classes
- **Non-zero forcing**: Implement `ForcingTerm::value()` method
- **Different FE spaces**: Change `fe` initialization (currently `FE_SimplexP`)
- **Time stepping schemes**: Modify θ parameter in constructor

### Debugging Convergence Issues
1. Run `SpaceConvergence` with mesh sequence (e.g., N = 20, 40, 80, 160)
2. Verify error decay: $\mathcal{O}(h^{r+1})$ in L² norm for degree-r elements
3. Run `TimeConvergence` with Δt sequence
4. Check energy stability via output (should be conserved for θ=0.5)
5. For θ≠0.5, expect numerical dissipation (θ>0.5) or potential instability (θ<0.5)

## Repository Structure
```
nmpde-QiaoTramacereWang/
├── CMakeLists.txt          # Main build configuration
├── cmake-common.cmake      # MPI/Boost/deal.II setup
├── build.sh                # Build script
├── ROADMAP.md              # Development roadmap
│
├── include/
│   └── Wave.hpp            # Main solver header
│
├── src/
│   ├── Wave.cpp            # Solver implementation
│   ├── Main.cpp            # Main wave simulation
│   ├── TimeConvergence.cpp # Temporal convergence study
│   ├── SpaceConvergence.cpp# Spatial convergence study
│   └── DissipationStudy.cpp# θ-scheme dissipation analysis
│
├── meshes/
│   ├── mesh-square-*.msh   # Generated simplex meshes
│   └── square_level_*.msh  # Legacy refinement meshes
│
├── scripts/
│   ├── plot_dissipation.py          # Dissipation analysis plots
│   ├── plot_space_convergence.py    # Spatial convergence plots
│   ├── plot_time_convergence.py     # Temporal convergence plots
│   ├── generate_square_meshes_by_N.py  # Mesh generation
│   ├── generate_2D_meshes.py        # Alternative mesh generation
│   ├── visualize_meshes.py          # Mesh visualization
│   └── derive_manufactured_solution*.py  # MMS derivation
│
├── results/
│   └── plots/              # Generated analysis plots
│
├── archive/                # Old implementation (reference)
│   ├── old_src/            # Previous source files
│   ├── old_include/        # Previous headers
│   └── old_scripts/        # Previous scripts
│
└── build/                  # Generated by CMake (gitignored)
```

## External Dependencies & Quirks
- **deal.II** (≥9.3.1): Found via `find_package` in cmake-common.cmake
- **MPI**: Required for parallel execution
- **Boost** (≥1.72.0): filesystem, iostreams, serialization
- **Trilinos**: Used for parallel linear algebra
- **Output files**: VTK files generated in current directory by default
- **Python scripts**: Require NumPy, Matplotlib, SymPy

---

## Numerical Properties (Dissipation & Dispersion)

### θ-Scheme Characteristics
- **θ = 0 (Forward Euler)**: Explicit, conditionally stable, requires CFL condition $\Delta t \leq h/c$
- **θ = ½ (Crank-Nicolson)**: Unconditionally stable, energy-conserving, no numerical dissipation, second-order accurate in time
- **θ = 1 (Backward Euler)**: Unconditionally stable, first-order accurate, introduces numerical dissipation (damping)

### Dispersion Analysis
For wave propagation, the numerical phase velocity $c_h$ differs from exact $c$:
- Crank-Nicolson (θ=½) minimizes phase error but may exhibit spurious oscillations for discontinuous data
- Backward Euler (θ=1) smooths out oscillations but over-damps high-frequency modes
- Optimal choice depends on problem requirements (accuracy vs. stability vs. smoothness)

### References
- Quarteroni, A. (2017). *Numerical Models for Differential Problems*
- Salsa, S., Verzini, G. (2022). *Partial Differential Equations in Action*

---

**Last Updated**: 2025-12-22  
**Project Type**: Numerical PDE Solver (Wave Equation, Finite Elements, deal.II)
