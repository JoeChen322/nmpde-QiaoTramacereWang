#!/usr/bin/env python3
"""
Derive manufactured solution for 2D wave equation with Neumann BCs

Wave equation: u_tt = c^2 * (u_xx + u_yy) + f(x,y,t)
Domain: [-1, 1] × [-1, 1]
Boundary conditions: ∂u/∂n = 0 on all boundaries (Neumann/reflecting)
"""

import sympy as sp
from sympy import sin, cos, pi, diff, simplify, symbols

def derive_manufactured_solution_2d():
    """
    Derive manufactured solution for 2D wave equation with Neumann BCs.
    Returns the exact solution and its derivatives.
    """
    # Define symbolic variables
    x, y, t, c = symbols('x y t c', real=True)
    
    # For Neumann BCs (∂u/∂n = 0 at boundaries), use cosine modes
    # Mode (k=1, m=1) with natural frequency ω = π*c*√(k² + m²) = π*c*√2
    k = 1
    m = 1
    omega = c * pi * sp.sqrt(k**2 + m**2)
    
    # Exact solution
    u_exact = cos(k * pi * x) * cos(m * pi * y) * cos(omega * t)
    
    # Compute derivatives
    u_t = diff(u_exact, t)
    u_xx = diff(u_exact, x, 2)
    u_yy = diff(u_exact, y, 2)
    u_tt = diff(u_exact, t, 2)
    
    # Compute source term: f = u_tt - c²(u_xx + u_yy)
    laplacian = u_xx + u_yy
    source = simplify(u_tt - c**2 * laplacian)
    
    # Initial conditions
    u_initial = u_exact.subs(t, 0)
    v_initial = u_t.subs(t, 0)
    
    return {
        'u_exact': u_exact,
        'u_t': u_t,
        'u_initial': u_initial,
        'v_initial': v_initial,
        'source': source,
        'omega': omega,
        'k': k,
        'm': m
    }

if __name__ == "__main__":
    solution = derive_manufactured_solution_2d()
    
    print("="*70)
    print("2D Wave Equation Manufactured Solution (Neumann BCs)")
    print("="*70)
    print()
    print("Domain: [-1, 1] x [-1, 1]")
    print("BCs: ∂u/∂n = 0 (Neumann/reflecting)")
    print()
    print("Exact solution:")
    print(f"  u(x,y,t) = {solution['u_exact']}")
    print()
    print("Time derivative:")
    print(f"  u_t(x,y,t) = {solution['u_t']}")
    print()
    print("Initial conditions:")
    print(f"  u(x,y,0) = {solution['u_initial']}")
    print(f"  u_t(x,y,0) = {solution['v_initial']}")
    print()
    print("Source term:")
    print(f"  f(x,y,t) = {solution['source']}")