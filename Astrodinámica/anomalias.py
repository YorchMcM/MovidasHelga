import numpy as np
from numpy import pi as PI
from matplotlib import use

use('TkAgg')

import matplotlib.pyplot as plt
TWOPI = 2*PI

e = 0.6
true_anomaly = np.arange(0.0, 361.0, 1.0) * TWOPI / 360.0
eccentric_anomaly = np.arctan2(np.sqrt(1-e**2)*np.sin(true_anomaly), e + np.cos(true_anomaly))
eccentric_anomaly[eccentric_anomaly < 0] = eccentric_anomaly[eccentric_anomaly < 0] + TWOPI
mean_anomaly = eccentric_anomaly - e*np.sin(eccentric_anomaly)

theta_dot = np.sqrt(1-e*2) / (1-e*np.cos(eccentric_anomaly))*2
E_dot = 1 / (1-e*np.cos(eccentric_anomaly))
M_dot = np.ones_like(true_anomaly)

true_anomaly = true_anomaly * 360.0 / TWOPI
eccentric_anomaly = eccentric_anomaly * 360.0 / TWOPI
mean_anomaly = mean_anomaly * 360.0 / TWOPI

plt.figure()
plt.plot(true_anomaly, true_anomaly, label = r'Anomalía verdadera')
plt.plot(true_anomaly, eccentric_anomaly, label = r'Anomalía excéntrica')
plt.plot(true_anomaly, mean_anomaly, label = r'Anomalía media')
plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
plt.grid()
plt.legend()
plt.xlabel(r'Anomalía verdadera [º]')
plt.ylabel(r'[º]')

plt.figure()
plt.plot(true_anomaly, theta_dot, label = r'Anomalía verdadera')
plt.plot(true_anomaly, E_dot, label = r'Anomalía excéntrica')
plt.plot(true_anomaly, M_dot, label = r'Anomalía media')
plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
plt.grid()
plt.legend()
plt.xlabel(r'Anomalía verdadera [º]')
plt.ylabel(r'Velocidad normalizada con movimiento medio [-]')