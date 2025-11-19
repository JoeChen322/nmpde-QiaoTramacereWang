#!/usr/bin/env python3
"""
Derive manufactured solution for 1D wave equation with Neumann BCs

Wave equation: u_tt = c^2 * u_xx + f(x,t)
Domain: [-1, 1]
Boundary conditions: u_x(-1,t) = 0, u_x(1,t) = 0 (Neumann/reflecting)

We need to find u_exact(x,t) that satisfies the Neumann BCs
and compute the corresponding source term f(x,t).
"""

import sympy as sp
from sympy import sin, cos, pi, diff, simplify, symbols

# Define symbolic variables
x, t, c = symbols('x t c', real=True)

print("="*70)
print("Manufactured Solution for 1D Wave Equation with Neumann BCs")
print("="*70)
print()
print("Wave equation: u_tt = c^2 * u_xx + f(x,t)")
print("Domain: x ∈ [-1, 1]")
print("Boundary conditions: u_x(-1,t) = 0, u_x(1,t) = 0 (Neumann)")
print()

# For Neumann BCs (u_x = 0 at boundaries), we need cos(n*pi*x/2) modes
# The derivative of cos(n*pi*x/2) is -n*pi/2 * sin(n*pi*x/2)
# At x = ±1: sin(±n*pi/2) = 0 when n is even, ±1 when n is odd
# So we need cos((2k)*pi*x/2) = cos(k*pi*x) where k = 1,2,3,...

# Try different manufactured solutions
solutions = []

# Solution 1: Cosine in space, cosine in time
print("Solution 1: Cosine-Cosine")
print("-" * 70)
u1 = cos(pi * x) * cos(pi * c * t)
print(f"u(x,t) = cos(π*x) * cos(π*c*t)")
print()

# Check derivatives
u1_x = diff(u1, x)
u1_xx = diff(u1, x, 2)
u1_t = diff(u1, t)
u1_tt = diff(u1, t, 2)

print("Spatial derivatives:")
print(f"u_x  = {u1_x}")
print(f"u_xx = {u1_xx}")
print()

# Check BCs
u1_x_at_minus1 = u1_x.subs(x, -1)
u1_x_at_plus1 = u1_x.subs(x, 1)
print("Boundary conditions:")
print(f"u_x(-1,t) = {u1_x_at_minus1}")
print(f"u_x(+1,t) = {u1_x_at_plus1}")
print("✓ Both are zero - Neumann BCs satisfied!" if u1_x_at_minus1 == 0 and u1_x_at_plus1 == 0 else "✗ BCs not satisfied")
print()

# Compute source term
f1 = simplify(u1_tt - c**2 * u1_xx)
print("Source term f(x,t) = u_tt - c^2 * u_xx:")
print(f"f = {f1}")
print()

solutions.append(("Cosine-Cosine (homogeneous)", u1, f1))

# Solution 2: Cosine in space, sine in time
print("Solution 2: Cosine-Sine")
print("-" * 70)
u2 = cos(pi * x) * sin(pi * t)
print(f"u(x,t) = cos(π*x) * sin(π*t)")
print()

u2_x = diff(u2, x)
u2_xx = diff(u2, x, 2)
u2_tt = diff(u2, t, 2)

# Check BCs
u2_x_at_minus1 = u2_x.subs(x, -1)
u2_x_at_plus1 = u2_x.subs(x, 1)
print("Boundary conditions:")
print(f"u_x(-1,t) = {u2_x_at_minus1}")
print(f"u_x(+1,t) = {u2_x_at_plus1}")
print("✓ Both are zero - Neumann BCs satisfied!" if u2_x_at_minus1 == 0 and u2_x_at_plus1 == 0 else "✗ BCs not satisfied")
print()

f2 = simplify(u2_tt - c**2 * u2_xx)
print("Source term f(x,t) = u_tt - c^2 * u_xx:")
print(f"f = {simplify(f2)}")
print()

solutions.append(("Cosine-Sine (with source)", u2, f2))

# Solution 3: Polynomial solution (also satisfies Neumann)
print("Solution 3: Polynomial")
print("-" * 70)
u3 = (1 - x**2) * cos(t)
print(f"u(x,t) = (1 - x^2) * cos(t)")
print()

u3_x = diff(u3, x)
u3_xx = diff(u3, x, 2)
u3_tt = diff(u3, t, 2)

u3_x_at_minus1 = u3_x.subs(x, -1)
u3_x_at_plus1 = u3_x.subs(x, 1)
print("Boundary conditions:")
print(f"u_x(-1,t) = {u3_x_at_minus1}")
print(f"u_x(+1,t) = {u3_x_at_plus1}")
print("✓ Both are zero - Neumann BCs satisfied!" if u3_x_at_minus1 == 0 and u3_x_at_plus1 == 0 else "✗ BCs not satisfied")
print()

f3 = simplify(u3_tt - c**2 * u3_xx)
print("Source term f(x,t) = u_tt - c^2 * u_xx:")
print(f"f = {f3}")
print()

solutions.append(("Polynomial", u3, f3))

# Print summary
print("="*70)
print("SUMMARY - Recommended Solutions")
print("="*70)
print()

for name, u, f in solutions:
    print(f"{name}:")
    print(f"  u(x,t) = {u}")
    print(f"  f(x,t) = {f}")
    
    # Initial conditions
    u_t0 = u.subs(t, 0)
    u_t_t0 = diff(u, t).subs(t, 0)
    print(f"  u(x,0) = {u_t0}")
    print(f"  u_t(x,0) = {u_t_t0}")
    print()

print("="*70)
print("RECOMMENDATION:")
print("="*70)
print()
print("Use Solution 1 (Cosine-Cosine) for c = 1:")
print("  u(x,t) = cos(π*x) * cos(π*t)")
print("  f(x,t) = 0  (homogeneous - exact solution!)")
print()
print("This is perfect because:")
print("  1. Satisfies Neumann BCs at x = ±1")
print("  2. Zero source term (f = 0)")
print("  3. Smooth everywhere")
print("  4. Simple to implement")
print()
print("Initial conditions:")
print("  u(x,0) = cos(π*x)")
print("  u_t(x,0) = 0")
print()

# Generate C++ code
print("="*70)
print("C++ IMPLEMENTATION")
print("="*70)
print()

print("For c = 1:")
print("""
// Exact solution: u(x,t) = cos(π*x) * cos(π*t)
double exact_solution(const dealii::Point<dim> &p, double t) const
{
    const double x = p[0];
    return std::cos(M_PI * x) * std::cos(M_PI * t);
}

// Time derivative: u_t(x,t) = -π * cos(π*x) * sin(π*t)
double exact_velocity(const dealii::Point<dim> &p, double t) const
{
    const double x = p[0];
    return -M_PI * std::cos(M_PI * x) * std::sin(M_PI * t);
}

// Initial displacement: u(x,0) = cos(π*x)
virtual double initial_displacement(const dealii::Point<dim> &p) const override
{
    const double x = p[0];
    return std::cos(M_PI * x);
}

// Initial velocity: u_t(x,0) = 0
virtual double initial_velocity(const dealii::Point<dim> &p) const override
{
    return 0.0;
}

// Source term: f(x,t) = 0
virtual double source_term(const dealii::Point<dim> &p, double t) const override
{
    return 0.0;
}
""")

print()
print("For general c (wave speed):")
print("""
// Exact solution: u(x,t) = cos(π*x) * cos(π*c*t)
double exact_solution(const dealii::Point<dim> &p, double t) const
{
    const double x = p[0];
    return std::cos(M_PI * x) * std::cos(M_PI * wave_speed_ * t);
}

// Time derivative: u_t(x,t) = -π*c * cos(π*x) * sin(π*c*t)
double exact_velocity(const dealii::Point<dim> &p, double t) const
{
    const double x = p[0];
    return -M_PI * wave_speed_ * std::cos(M_PI * x) * std::sin(M_PI * wave_speed_ * t);
}

// Initial displacement: u(x,0) = cos(π*x)
virtual double initial_displacement(const dealii::Point<dim> &p) const override
{
    const double x = p[0];
    return std::cos(M_PI * x);
}

// Initial velocity: u_t(x,0) = 0
virtual double initial_velocity(const dealii::Point<dim> &p) const override
{
    return 0.0;
}

// Source term: f(x,t) = 0
virtual double source_term(const dealii::Point<dim> &p, double t) const override
{
    return 0.0;
}
""")
