import numpy as np
from numpy import pi as PI
from matplotlib import use, interactive
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from typing import Callable
import warnings

use('TkAgg')
interactive(True)
TWOPI = 2*PI


def derivative(x: np.ndarray, t: float)\
        -> np.ndarray:

    A = [ [0, 0,  1, 0],
          [0, 0,  0, 1],
          [0, 0,  0, t],
          [0, 0, -t, 0] ]

    dxdt = A @ x

    return dxdt


def rk_step(x: np.ndarray, t: float, dt: float)\
        -> np.ndarray:

    k1 = derivative(x, t)
    k2 = derivative(x + 0.5*k1*dt, t+0.5*dt)
    k3 = derivative(x + 0.5*k2*dt, t+0.5*dt)
    k4 = derivative(x + k3*dt, t + dt)

    x = x + (1.0/6.0) * dt * (k1 + 2.0*k2 + 2.0*k3 + k4)

    return x


def euler_step(x: np.ndarray, t: float, dt: float)\
        -> np.ndarray:

    x = x + derivative(x,t) * dt

    return x


def integrate_rk4(x0: np.ndarray, t0: float, dt: float, tf: float)\
        -> np.ndarray:

    time_array = np.round(np.arange(t0, tf+dt, dt), -int(np.log10(dt)))
    result = np.zeros([len(time_array), 1+len(x0)])
    result[:,0] = time_array
    result[0,1:] = x0

    for idx, time in enumerate(time_array[:-1]):

        result[idx+1,1:] = rk_step(result[idx,1:], time, dt)

    return result


def integrate_euler(x0: np.ndarray, t0: float, dt: float, tf: float)\
        -> np.ndarray:

    time_array = np.round(np.arange(t0, tf+dt, dt), -int(np.log10(dt)))
    result = np.zeros([len(time_array), 1+len(x0)])
    result[:,0] = time_array
    result[0,1:] = x0

    for idx, time in enumerate(time_array[:-1]):

        result[idx+1,1:] = euler_step(result[idx,1:], time, dt)

    return result


x0 = np.array([0,0,1,0])
t0, tf, dt = 0.0, 20.0, 0.01

trajectory = integrate_rk4(x0, t0, dt, tf)
trajectory_euler = integrate_euler(x0, t0, dt, tf)
trajectory_euler_fine = integrate_euler(x0, t0, 0.1*dt, 2.0*tf)
diffs = trajectory_euler - trajectory
# diffs_fine = trajectory_euler_fine - trajectory
diffs[:,0] = trajectory[:,0]
# diffs_fine[:,0] = trajectory[:,0]

plt.figure()
plt.plot(trajectory[:,0], trajectory[:,1], label = r'$x(t)$')
plt.plot(trajectory[:,0], trajectory[:,2], label = r'$y(t)$')
# plt.plot(trajectory_euler[:,0], trajectory_euler[:,1], label = r'$x(t)$ (Euler)')
# plt.plot(trajectory_euler[:,0], trajectory_euler[:,2], label = r'$y(t)$ (Euler)')
plt.grid()
plt.legend()
plt.xlabel('Tiempo [$1/\lambda$]')
plt.ylabel(r'Coordenadas [$v_o/\lambda$]')
plt.title(r'Trayectoria adimensional $(\lambda^2 = qc/m)$')

# plt.figure()
# plt.plot(trajectory_euler[:,0], trajectory_euler[:,1], label = r'$x(t)$')
# plt.plot(trajectory_euler[:,0], trajectory_euler[:,2], label = r'$y(t)$')
# plt.grid()
# plt.legend()
# plt.xlabel('Tiempo [$1/\lambda$]')
# plt.ylabel(r'Coordenadas [$v_o/\lambda$]')
# plt.title(r'Trayectoria adimensional (Euler) $(\lambda^2 = qc/m)$')

plt.figure()
plt.semilogy(diffs[:,0], abs(diffs[:,1]), label = r'$\Delta x(t)$')
plt.semilogy(diffs[:,0], abs(diffs[:,2]), label = r'$\Delta y(t)$')
# plt.semilogy(diffs[:,0], abs(diffs[:,1]), label = r'$\Delta x(t)$ ($\Delta t = 0.001$)')
# plt.semilogy(diffs[:,0], abs(diffs[:,2]), label = r'$\Delta y(t)$ ($\Delta t = 0.001$)')
plt.grid()
plt.legend()
plt.xlabel('Tiempo [$1/\lambda$]')
plt.ylabel(r'Diferencias [$v_o/\lambda$]')
plt.title(r'Diferencias entre Euler y RK4 $(\lambda^2 = qc/m)$')

plt.figure()
plt.plot(trajectory[:,1], trajectory[:,2])
plt.grid()
plt.axis('equal')
plt.xlabel('$x$  $[v_o/\lambda]$')
plt.ylabel('$y$  $[v_o/\lambda]$')
plt.title(r'Trayectoria')

plt.figure()
plt.plot(trajectory_euler[:,1], trajectory_euler[:,2])
plt.grid()
plt.axis('equal')
plt.xlabel('$x$  $[v_o/\lambda]$')
plt.ylabel('$y$  $[v_o/\lambda]$')
plt.title(r'Trayectoria (Euler, $\Delta t = 0.01$)')

plt.figure()
plt.plot(trajectory_euler_fine[:,1], trajectory_euler_fine[:,2])
plt.grid()
plt.axis('equal')
plt.xlabel('$x$  $[v_o/\lambda]$')
plt.ylabel('$y$  $[v_o/\lambda]$')
plt.title(r'Trayectoria (Euler, $\Delta t = 0.001$)')

# plt.show()