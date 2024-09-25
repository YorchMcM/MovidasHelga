# import os
import numpy as np
from numpy import pi as PI
from matplotlib import use, interactive
import matplotlib.pyplot as plt

interactive(True)
use('TkAgg')

TWOPI = 2.0*PI

plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')
# plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
plt.rcParams['lines.markersize'] = 6.0

x0 = 13.4
y0 = 6.89
vx0 = 0.1
vy0 = 56.8
g = 9.81
E = 0.2

m = 0.4
q = 2.61
N_params = 6


def x(t):
    return x0 + vx0*t + 0.5*q/m*E*t**2.0


def y(t):
    return y0 + vy0*t - 0.5*g*t**2


t_min = 0.0
t_max = 10.0
t_step = 0.05
t = np.arange(t_min, t_max+t_step, t_step)

t_step_obs = 0.5
t_obs = np.arange(t_min, t_max+t_step_obs, t_step_obs)

# noise_x = np.random.normal(size = len(t_obs))
# noise_y = np.random.normal(size = len(t_obs))
noise_x = np.array([ 0.17937325,  0.23838382, -0.30640050,  1.40191799, -1.37515145,
                     0.42984278, -0.35815166, -0.21502256,  0.39440315,  2.11916393,
                     1.30259821, -1.27045575,  0.43752113,  1.05424413,  0.79689396,
                    -0.64466918,  0.33161518, -0.06543112, -0.38316589, -0.03321996,
                    -0.68671885])
noise_y = np.array([ 1.03828050, -1.57251880,  1.54352467, -2.47723368,  0.24020734,
                     0.56415681, -1.54742110, -2.05221592, -0.69803908, -0.78984794,
                     3.47122296,  0.61474802,  0.55578097, -0.77050015,  0.06277035,
                     0.05959999, -0.99208181, -0.67464570,  1.69212532,  1.32702312,
                     0.22440582])

# Design matrix
H = np.zeros([2*len(t_obs), 6])
I = np.eye(2)
for idx, t_i in enumerate(t_obs):
    H[2*idx:2*idx+2,:2] = I
    H[2*idx:2*idx+2,2:4] = t_i*I
    H[2*idx:2*idx+2,4:] = 0.5*t_i**2.0*np.array([[0.0, q/m],
                                        [-1.0, 0.0]])

# Observaciones y  ruido
z = np.zeros(2*len(t_obs))
z[::2] = x0 + vx0*t_obs + 0.5*q/m*E*t_obs**2.0
z[1::2] = y0 + vy0*t_obs - 0.5*g*t_obs**2
noise = np.zeros(2*len(t_obs))
noise[::2] = noise_x
noise[1::2] = noise_y

# Observation covariance
S = np.zeros([len(z)])
S[::2] = 1.5
S[1::2] = 2.0
S = np.diag(S)

# Solución
p = np.linalg.solve(H.T @ H, H.T @ z) # Ideales
p_noise = np.linalg.solve(H.T @ H, H.T @ (z+noise)) # Con ruido
covariance = np.linalg.inv(H.T @ H) # Matriz de covarianza
correlation = np.zeros_like(covariance) # Matriz de correlación
for row in range(N_params):
    sigma_row = np.sqrt(covariance[row,row])
    for col in range(N_params):
        sigma_col = np.sqrt(covariance[col,col])
        correlation[row,col] = covariance[row,col] / sigma_row / sigma_col

covariance_S = np.linalg.inv(H.T @ np.linalg.inv(S) @ H) # Matriz de covarianza
correlation_S = np.zeros_like(covariance_S) # Matriz de correlación
for row in range(N_params):
    sigma_row = np.sqrt(covariance_S[row,row])
    for col in range(N_params):
        sigma_col = np.sqrt(covariance_S[col,col])
        correlation_S[row,col] = covariance_S[row,col] / sigma_row / sigma_col

residuos = H @ p_noise - z
residuos_x = residuos[::2]
residuos_y = residuos[1::2]

# Trayectoria y observaciones sin ruido
plt.figure()
plt.plot(x(t), y(t), c = 'b')
plt.scatter(x(t_obs), y(t_obs), s = 15, c = 'r', zorder = 2)
plt.grid()
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.title(r'Trayectoria')

# Trayectoria y observaciones con ruido
plt.figure()
plt.plot(x(t), y(t), c = 'b')
plt.scatter(x(t_obs), y(t_obs), marker = 'o', s = 20, edgecolors = 'k', zorder = 2, facecolors = 'none')
plt.scatter(x(t_obs)+noise_x, y(t_obs)+noise_y, s = 15, c = 'r', zorder = 2)
plt.grid()
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.title(r'Trayectoria')

# Residuos vs ruido
plt.figure()
plt.plot(t_obs, residuos_x, label = r'Residuo $x$')
plt.plot(t_obs, residuos_y, label = r'Residuo $y$')
plt.plot(t_obs, noise_x, label = r'Ruido $x$')
plt.plot(t_obs, noise_y, label = r'Ruido $y$')
plt.grid()
plt.legend()
plt.xlabel(r'Tiempo [s]')
plt.ylabel(r'Residuo [m]')

# plt.show()