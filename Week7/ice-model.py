import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import sparse
from scipy import interpolate
from scipy.integrate import solve_ivp

# Load in the data
data = np.array([[801., -1.4404231],
                 [741., -3.3368846],
                 [721., -4.3305769],
                 [701., -5.3550385],
                 [681., -6.4179615],
                 [661., -7.3731923],
                 [641., -8.3976538],
                 [621., -9.3298077],
                 [601., -9.9696538],
                 [581., -10.947962],
                 [561., -11.572423],
                 [541., -12.012269],
                 [521., -12.652115],
                 [501., -12.630423],
                 [481., -12.654885],
                 [461., -13.817808],
                 [441., -13.834577],
                 [421., -12.9975],
                 [401., -13.429654],
                 [381., -13.261808],
                 [361., -13.863192],
                 [341., -13.472269],
                 [321., -13.565962],
                 [301., -13.782731],
                 [281., -13.168731],
                 [261., -13.131654],
                 [241., -12.963808],
                 [221., -12.095962],
                 [201., -12.0435],
                 [181., -12.152577],
                 [161., -11.577038],
                 [141., -11.3015],
                 [121., -11.279808],
                 [101., -10.5735],
                 [81., -10.074885],
                 [61., -9.6993462],
                 [41., -9.5238077],
                 [21., -9.4867308],
                 [1., -9.6111923]])

data[:, 1] += 273.25

def C_to_K(x):
    """ Converts Celsius to Kelvin """
    return x + 273.15


def K_to_C(x):
    """ Converts Kelvin to Celsius """
    return x - 273.15


# Find a surface temperature by interpolating the provided data
interp = interpolate.interp1d(data[:, 0], data[:, 1], kind='quadratic', fill_value='extrapolate')
theta_s = float(interp(0))

# Define Constants
z_s = 0  # m
z_b = -data[0, 0]  # m
theta_b = data[0, 1]
g = 9.81  # m/s^2
spy = 31556926  # s/a
rho = 911  # kg/m^3
C = 2009  # J/kg/K
beta = 9.8e-8  # K/Pa
k = 2.1  # W/m/K
u_s = (1 / spy) * 90  # m/s
u_b = (1 / spy) * 1e-12  # m/s
a_dot = (1 / spy) * -1  # m/s
dzs_dx = np.radians(0.7)
# lamda = C_to_K(7e-3)
lamda = 7e-3
Q_geo = 32e-3

# Constants that depend on constants above
theta_pmp = -beta * rho * g * (z_s - z_b)
dtheta_dx = lamda * dzs_dx


# Define dependent variables
def sigma(z_i):
    """Normalized vertical coordinate"""
    numerator = z_s - z_i
    denominator = z_s - z_b
    return numerator / denominator


def u_func(z):
    """ Horizontal ice velocity at depth (m/s) """
    return u_s * (1 - sigma(z))


def w_func(z):
    """ Vertical ice velocity (m/s) """
    return -a_dot * (1 - sigma((5 / 4) - (sigma(z) / 4)))


def du_dz_func(z):
    """ Vertical Shear (1/s) """
    return 4 * (u_s - u_b) * sigma(z) ** 3 / (z_s - z_b)


def phi_func(z):
    """ Heat sources from deformation of ice """
    return rho * g * (z_s - z) * du_dz_func(z) * dzs_dx


# Create a vector of different heights to sample the temperature
time_bounds = [0, 50 * spy]
dt = spy/1000
Nt = int((time_bounds[1] - time_bounds[0]) / dt)
num = 100
z_vec = np.linspace(-data[0, 0], data[-1, 0], num)
dz = (data[0, 0] - data[-1, 0]) / num

# Apply finite difference to first term in the PDE (d^2 theta / d z^2)
constant = k / rho / C / dz ** 2
left = 1 * constant
center = -2 * constant
right = 1 * constant
A = sparse.diags([left, center, right], [-1, 0, 1], shape=[num, num])
A = sparse.csc_matrix(A)

# Apply finite different to second term in the PDE (d theta / d z)
w = w_func(z_vec)
constant = 1 / (2 * dz)
left = -1 * constant
right = 1 * constant
C = sparse.diags([left, right], [-1, 1], shape=[num, num])
C = w * sparse.csc_matrix(C)

AC = A + C

# Constants in the PDE
B = u_func(z_vec) * dtheta_dx
D = phi_func(z_vec) / rho / C

boundary_constant = data[0, 1] - z_b * Q_geo / k
def fix_boundary(y, t):
    y[0] = theta_s
    if y[-1] < theta_pmp:
        # y[-1] = bc_integrator.integrate(t)
        y[-1] += dz * Q_geo / k + boundary_constant
    else:
        y[-1] = theta_pmp
    return y


def neumann(y, t):
    return Q_geo / k


def f(t, y):
    y = y.squeeze()
    y = fix_boundary(y, t)
    # return A @ y + B + C @ y + D
    return AC @ y + B + D


initial_values = interp(z_vec)

# bc_integrator = ode(neumann)
# bc_integrator.set_integrator('dop853', atol=1e-12)
# bc_integrator.set_initial_value(initial_values[-1], time_bounds[0])

# integrator = ode(f)
# integrator.set_integrator('dop853', atol=1e-12)
# integrator.set_f_params(A, B, C, D)
# integrator.set_initial_value(initial_values, time_bounds[0])

# pbar_model = tqdm.tqdm(total=(Nt - 1), position=0)
# while integrator.t <= time_bounds[-1]:
#     theta = integrator.integrate(integrator.t + dt)
#     theta = fix_boundary(integrator.y, integrator.t)
#     pbar_model.update(1)

# Convert theta back to celcius
# theta = K_to_C(theta)

solution = solve_ivp(f, time_bounds, initial_values, method='RK45')
theta = solution.y[:, -1]

plt.plot(data[:, 1], -data[:, 0])
plt.plot(solution.y[:, 0], z_vec)
plt.show()
print(f'\nDone {theta.min()}, {theta.max()}')
