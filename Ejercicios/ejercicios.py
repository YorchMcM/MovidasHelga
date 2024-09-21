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

gamma = -0.5*np.ones_like(x)
gamma[x < 0.0] = -(x[x<0.0]+1.0/(x[x<0.0]-0.5)**2.0/8.0)
gamma[x > 1.0] = -(1.0/x[x>1.0]**2 - 1.0/(x[x>1.0]-0.5)**2.0/8.0)
gamma[x < -1.0] = 1.0/x[x < -1.0]**2 - 1.0/8.0/(x[x<-1.0]-0.5)**2.0

plt.figure()
plt.plot(x,gamma)
plt.grid()
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\gamma_x$')
plt.axvline(0.0, ls = 'dashdot', c = 'k')
plt.axvline(0.5, ls = 'dashdot', c = 'k')
plt.axvline(1.0, ls = 'dashed', c = 'k')
plt.axvline(-1.0, ls = 'dashed', c = 'k')
plt.axhline(0.0, c = 'k')
plt.title(r'Componente $x$ del campo gravitatorio (adimensional)')

plt.show()
