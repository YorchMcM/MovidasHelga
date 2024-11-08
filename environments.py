import numpy as np


class Separator:
    # Properties of the separator
    Q = 0.0004076  # Volume flow inside the separator [m³/s] #0.0002241 0.0004076
    omega = 50  # Angular velocity of centrifuge [rad/s] (for omega^2 * R = 5000)
    phi = 65  # Plate inclination angle [º]
    n = 34  # Number of plates
    H = 0.35e-3  # Inter-plate distance [m] (R.Plat paper)
    Rt = 0.19  # Radius of outer case [m] (R.Plat paper) # 0.170000 #0.08
    R = 0.1785824  # Radius of plates [m] (R.Plat paper) # 0.169755 #0.107
    L = 0.189  # Plate length [m] # 0.24 #0.07  #m
    h = H * np.sin(np.radians(phi))
    D_l = 0.377  # 0.089 #0.07
    D_h = 0.004  # 0.0099 #0.02

    # Material properties of the light phase
    mu_l = 1.53e-3  # Dynamic viscosity of light phase (HVO) [Ns/m²]
    rho_l = 765  # Density of light phase (HVO) [kg/m³]
    sigma_l = 0.0245  # Surface tension of light phase (HVO) [N/m]
    ni_l = mu_l / rho_l  # Kinematic viscosity of light phase (HVO) [m²/s]

    # Properties of the mixture
    X_l = 0.9  # Mass fraction of light phase [-] # ?
    X_h = 0.1  # Mass fraction of heavy phase [-] # ?

    def compute_couette_velocity(self, position: np.ndarray):
        x, y = position

        x = x / self.L
        y = y / self.h

        return (3 * self.Q * self.L) / (np.pi * self.n * self.h) * (y / x) * (1 - y)


class Particle:
    # Material properties of heavy phase
    mu = 461.39e-6  # Dynamic viscosity of heavy phase (methanol) [Ns/m²]
    rho_h = 795.5  # Density of heavy phase (methanol) [kg/m³]

    def __init__(self, particle_diameter: float):
        # Physical properties of the particle
        self.Dp = particle_diameter  # Particle diameter [m]

        # Derived properties
        self.volume = (4.0 / 3.0) * np.pi * (self.Dp / 2.0) ** 3.0
        self.mass = self.volume * self.rho_h


def compute_equilirbium_velocity(
    separator: Separator, particle: Particle, position: np.ndarray
):
    x, y = position

    density_difference = particle.rho_h - separator.rho_l
    premultiplying_factor = 16.0 * separator.mu_l / 3.0 / separator.rho_l / particle.Dp
    numerator = (
        particle.Dp**2
        * density_difference
        * separator.omega**2
        * x
        * np.sin(np.radians(separator.phi))
    )
    denominator = 18.0 * separator.mu_l
    equilibrium_velocity = separator.compute_couette_velocity(
        position
    ) + premultiplying_factor * (-1.0 + numerator / denominator)

    return equilibrium_velocity
