import numpy as np
from numpy import pi as PI
from matplotlib import use
from typing import Callable
import warnings

use('TkAgg')

import matplotlib.pyplot as plt
TWOPI = 2*PI

mu = 0.8
step = 0.001
x = np.arange(-2.0, 2.0+step, step)
eta = x+1-mu

# f = mu - mu/(np.abs(x+1-mu)**3) - (mu/np.abs(x+1-mu)**3+(1-mu)/np.abs(x-mu)**3-1)*(x-mu)
f = x - np.sign(x+1-mu)*mu/(x+1-mu)**2 - np.sign(x-mu)*(1-mu)/(x-mu)**2

plt.figure()
# plt.scatter(x, f, s = 1)
plt.plot(x,f)
plt.axvline(-(1-mu), c = 'red')
plt.axvline(mu, c = 'red')
plt.grid()
plt.xlabel(r'x')
plt.ylabel(r'f(x)')
plt.ylim([-20, 20])


def f3(x: float) -> float:

    # To find L3 (left of the primary), alpha, beta < 0

    eta = x + 1 - mu

    to_return = x + mu/eta**2 + (1-mu)/(eta-1)**2

    return to_return


def f1(x: float) -> float:

    # To find L1 (in between primary and secondary), alpha > 0, beta < 1

    eta = x + 1 - mu

    to_return = x - mu / eta ** 2 + (1 - mu) / (eta - 1) ** 2

    return to_return


def f2(x: float) -> float:

    # To find L2 (right of secondary), alpha, beta > 0

    eta = x + 1 - mu

    to_return = x - mu / eta ** 2 - (1 - mu) / (eta - 1) ** 2

    return to_return


def derivative(x: float) -> float:

    # To find the derivative (always the same, independent of alpha and beta).

    eta = x + 1 - mu

    to_return = 1 + 2*mu/abs(eta)**3 + 2*(1-mu)/abs(eta-1)**3

    return to_return


def newton_raphson(fun: Callable[[float], float],
                   x0: float,
                   fun_deriv: Callable[[float], float],
                   tol: float = 1e-5,
                   max_iter: int = 15,
                   leash: float = 0.0) -> float:

    x_old = x0
    iter = 0

    converged = False
    while (not converged) and iter < max_iter:

        delta = fun(x_old)/fun_deriv(x_old)

        if 0.0 < leash < abs(delta):
            x_new = x_old - np.sign(delta)*leash
        else:
            x_new = x_old - delta

        converged = abs(x_new - x_old) < tol
        iter = iter + 1

        # print('x_old = ', x_old, ' f_old = ', fun(x_old), ' x_new = ', x_new, ' delta = ', delta, ' converged = ', converged)

        if iter >= max_iter:
            warnings.warn('(newton_raphson): Maximum number of iterations reached. Exiting algorithm.')
            break

        if not converged: x_old = x_new

    return x_new

# l1 = (newton_raphson(f1, 0.8*mu, derivative, max_iter = 100, leash = 0.1) + 1 - mu ) / (1 - mu)
# l2 = (newton_raphson(f2, 1.2*mu, derivative, max_iter = 100, leash = 0.1) + 1 - mu ) / (1 - mu)
# l3 = (newton_raphson(f3, -1.2*(1-mu), derivative, max_iter = 100, leash = 0.1) + 1 - mu ) / (1 - mu)
# l0 = (1+np.sqrt((1/mu)-1))**(-1)  / (1 - mu)
# l0 = l0 - 1 + mu
l1 = newton_raphson(f1, 0.8*mu, derivative, max_iter = 100, leash = 0.1)
l2 = newton_raphson(f2, 1.2*mu, derivative, max_iter = 100, leash = 0.1)
l3 = newton_raphson(f3, -1.2*(1-mu), derivative, max_iter = 100, leash = 0.1)
print('L1', l1)
print('L2', l2)
print('L3', l3)