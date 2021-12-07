"""
Model based on:
    On a wildland re model with radiation
    M. I. Asensio and L. Ferragut
    INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING
Int. J. Numer. Meth. Engng 2002; 54:137–157 (DOI: 10.1002/nme.420)
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.420
"""

from firedrake import *
from constants import Kappa, epsilon, alpha, T_inf, T_pc

# For constants, initial conditions and size of domain see pages 152-153
q = 1  # H*Y0/(C*T_inf)
u_pc = (T_pc - T_inf) / (epsilon * T_inf)  # about 8.3

# Mesh
nx = 50
ny = 50
Lx = 90
Ly = 90
mesh = RectangleMesh(nx, ny, Lx, Ly)

# Primary function space:
V = FunctionSpace(mesh, "CG", 1)
# Mixed function space:
W = V * V
# For vector functions like the wind field:
V2 = VectorFunctionSpace(mesh, "CG", 2)

# State variables
# The first component will be u, the second beta
# This all gets a little ackward due to the way
# mixed function spaces can be assigned values for initialization
curr = Function(W)  # Current time step
prev = Function(W)  # Previous time step

# Test function space
nu, eta = TestFunction(W)

# Parameters
w = Function(V2, name="Wind")

# Initial conditions and parameter values
x, y = SpatialCoordinate(mesh)

# logistic equation to have a circular step in the center of the domain
r_center = sqrt((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2)  # radial distance from center of domain
# Note: In the paper on page 63 they mention that u_max is set to 63.
u_max = 63  # Stability is strongly related to this value, make dt smaller if this higher
radius = 4.0
smooth = 1.5
uic = u_max * (1 - 1 / (1 + exp(-smooth * (r_center - radius))))

curr.sub(0).assign(project(uic, V))  # initial value of u (temp)
curr.sub(1).assign(project(Constant(.1), V))  # initial value of beta (fuel)

prev.sub(0).assign(project(uic, V))
prev.sub(1).assign(project(Constant(.1), V))

# Wind field
w.assign(as_vector([300, 300]))


# Functions that appear in the PDE:
# The nonlinear diffusion coefficient:
def kappa(u):
    """
    The diffusivity is nonlinear to include the radiative transport of heat.
    See workup in the introduction of the paper.
    This non-dimensional form appears in eqns 5, 3
    """
    return Kappa * (1 + epsilon * u) ** 3 + 1


def s_plus(u):
    """
    Simple boolean function appearing in both
    f(u,beta) and
    g(u,beta)
    Activates reaction chemistry when temperature threshold is exceeded.
    see eqns 5, 6
    """
    return conditional(ge(u, u_pc), 1, 0)


def f(u, beta):
    """
    This is the reaction term in the PDE for u (non-dimensional temp)
    alpha parametrizes convection up into atmosphere.
    s_plus activates the term when the temperature of combustion (u_pc)
    is reached.
    See eqns 5, 4
    """
    return s_plus(u) * beta * exp(u / (1 + epsilon * u)) - alpha * u


def g(u, beta):
    """
    This is the right hand side of the ODE for beta (fuel)
    See eqns 5, 5
    """
    return -s_plus(u) * epsilon / q * beta * exp(u / (1 + epsilon * u))


# The timestep is set to produce an advective Courant number of
# around 1.
# Diverged after 0.002973 with dt = 1e-7
dt = 0.5e-8

# Need to recover individual functions to write the weak form.
u, beta = split(curr)
u_, beta_ = split(prev)

# Here we define the residual of the equation. For a nonlinear problem,
# there are no trial functions in the formulation. These will be created
# automatically when the residual is differentiated by the nonlinear solver:
# The relations here can be checked against eqns 5 1,2 in the paper.

F = ((u - u_) / dt * nu + inner(w, grad(u)) * nu
     + inner(kappa(u) * grad(u), grad(nu))
     - f(u, beta) * nu
     + (beta - beta_) / dt * eta
     - g(u_, beta_) * eta) * dx

# Note the equations are implicit in that u depends on u and not u_, this is good.
# however, g(u_,beta_) is called so that the rate of change in the ODE for beta
# depends on previous rates of change only - this is proving more stable than an
# implicit approach of g(u,beta)

out_T = File("output/temp.pvd")
out_beta = File("output/fuel.pvd")
u, beta = curr.split()
out_T.write(u)
out_beta.write(beta)

# Main loop over all times steps, outputing solutions as some interval
# specified as an integer number of timesteps.
t = 0.0
t_end = 10
output_frequency = 1000  # Output once after this many steps
count = 0
while t <= t_end:
    print("time: %f/%f" % (t, t_end))
    solve(F == 0, curr)
    u, beta = curr.split()
    prev.sub(0).assign(u)
    prev.sub(1).assign(beta)
    u_, beta_ = split(prev)
    t += dt
    if count % output_frequency == 0:
        out_T.write(u)
        out_beta.write(beta)
        print("output written at: %f" % t)
    count += 1
