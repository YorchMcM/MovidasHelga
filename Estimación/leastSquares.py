# import os
import numpy as np
# from matplotlib import use
import matplotlib.pyplot as plt
# use('TkAgg')

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

plt.figure()
plt.plot(x(t), y(t), c = 'b')
plt.scatter(x(t_obs), y(t_obs), s = 15, c = 'r', zorder = 2)
plt.grid()
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.title(r'Trayectoria')

# Design matrix
H = np.zeros([2*len(t_obs), 6])
I = np.eye(2)
for idx, t_i in enumerate(t_obs):
    H[2*idx:2*idx+2,:2] = I
    H[2*idx:2*idx+2,2:4] = t_i*I
    H[2*idx:2*idx+2,4:] = 0.5*t_i**2.0*np.array([[0.0, q/m],
                                        [-1.0, 0.0]])

# Observaciones
z = np.zeros(2*len(t_obs))
noise = np.zeros(2*len(t_obs))
for idx, t_i in enumerate(t_obs):
    z[2*idx] = x0 + vx0*t_i + 0.5*q/m*E*t_i**2.0
    z[2*idx+1] = y0 + vy0*t_i - 0.5*g*t_i**2
    noise[2*idx] = noise_x[idx]
    noise[2*idx+1] = noise_y[idx]


# Solución
p = np.linalg.solve(H.T @ H, H.T @ z) # Ideales
p_noise = np.linalg.solve(H.T @ H, H.T @ (z+noise)) # Con ruido
covariance = np.linalg.inv(H.T @ H)
correlation = np.zeros_like(covariance)
for row in range(N_params):
    sigma_row = np.sqrt(covariance[row,row])
    for col in range(N_params):
        sigma_col = np.sqrt(covariance[col,col])
        correlation[row,col] = covariance[row,col] / sigma_row / sigma_col

plt.show()
