# The following import includes numpy, environments.py and numericalMethods.py.
from numericalMethods import *

from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

unique_separator = Separator()
rk_integrator = RK_integrator(time_step=1e-5)

initial_time = 0.0
termination_conditions = [
    TimeTermination(termination_time=310),
    XLimTermination(limits=[0.0, unique_separator.L]),
]
initial_position = np.array([unique_separator.L, 0.0])
for i in range(5):
    test_particle = Particle(particle_diameter=10e-6 + 10e-6 * i)
    test_problem = MathematicalModel(unique_separator, test_particle)
    test_simulator = Simulator(test_problem, rk_integrator)

    initial_velocity = np.array([compute_equilirbium_velocity(unique_separator, test_particle, initial_position), 0.0])
    initial_state = np.concatenate((initial_position, initial_velocity))
    print(f"Simulating trajectory particle {i+1}/5...")
    trajectory_of_test_particle = test_simulator.simulate_trajectory(initial_time,
                                                                     initial_state,
                                                                     termination_conditions)

    # small_particle = Particle(particle_diameter = 20e-3)
    # medium_particle = Particle(particle_diameter = 40e-3)
    # big_particle = Particle(particle_diameter = 60e-3)
    #
    # small_problem = MathematicalModel(unique_separator, small_particle)
    # medium_problem = MathematicalModel(unique_separator, medium_particle)
    # big_problem = MathematicalModel(unique_separator, big_particle)
    #
    # small_simulator = Simulator(small_problem, rk_integrator)
    # medium_simulator = Simulator(medium_problem, rk_integrator)
    # big_simulator = Simulator(big_problem, rk_integrator)

    # initial_velocity = np.array([compute_equilirbium_velocity(unique_separator, small_particle, initial_position), 0.0])
    # initial_state = np.concatenate((initial_position, initial_velocity))
    # print('Simulating trajectory of SMALL particle...')
    # trajectory_of_small_particle = small_simulator.simulate_trajectory(initial_time, initial_state, termination_conditions)

    # initial_velocity = np.array([compute_equilirbium_velocity(unique_separator, medium_particle, initial_position), 0.0])
    # initial_state = np.concatenate((initial_position, initial_velocity))
    # print('Simulating trajectory of MEDIUM particle...')
    # trajectory_of_medium_particle = medium_simulator.simulate_trajectory(initial_time, initial_state, termination_conditions)
    #
    # initial_velocity = np.array([compute_equilirbium_velocity(unique_separator, big_particle, initial_position), 0.0])
    # initial_state = np.concatenate((initial_position, initial_velocity))
    # print('Simulating trajectory of BIG particle...')
    # trajectory_of_big_particle = big_simulator.simulate_trajectory(initial_time, initial_state, termination_conditions)

    plt.figure(1)
    plt.plot((trajectory_of_test_particle[:, 0] - trajectory_of_test_particle[0, 0]) * 1e3,
             trajectory_of_test_particle[:, 2] / unique_separator.h,
             label=f"$D_p = {np.round(test_particle.Dp * 1e6 * 100)/100} \mu m$")
    plt.grid()
    plt.legend()
    plt.xlabel(r"Time [ms]")
    plt.ylabel(r"$y / h$ [-]")
    plt.title(r"$Q = " + str(unique_separator.Q) + r"\frac{m^3}{s}$")

    plt.figure(2)
    plt.plot((trajectory_of_test_particle[:, 0] - trajectory_of_test_particle[0, 0]) * 1e3,
             1.0 - trajectory_of_test_particle[:, 1] / unique_separator.L,
             label=f"$D_p = {np.round(test_particle.Dp * 1e6 * 100)/100} \mu m$")
    plt.grid()
    plt.legend()
    plt.xlabel(r"Time [ms]")
    plt.ylabel(r"1.0 - $x / L$ [-]")
    plt.title(r"$Q = " + str(unique_separator.Q) + r"\frac{m^3}{s}$")

    plt.figure(3)
    plt.axhline(0.0, c="k")
    plt.axhline(1.0, c="k")
    plt.axvline(0.0, c="k")
    plt.axvline(1.0, c="k")
    plt.axvline(test_problem.R / (unique_separator.L * np.sin(np.deg2rad(unique_separator.phi))),
                c="k",
                linestyle="--")
    plt.axvline(unique_separator.Rt / (unique_separator.L * np.sin(np.deg2rad(unique_separator.phi))),
                c="k",
                linestyle="-.")
    plt.plot(trajectory_of_test_particle[:, 1] / unique_separator.L,
             trajectory_of_test_particle[:, 2] / unique_separator.h,
             label=f"$D_p = {np.round(test_particle.Dp * 1e6 * 100)/100} \mu m$")
    plt.grid()
    plt.xlabel(r"$x / L$ [-]")
    plt.ylabel(r"$y / h$ [-]")
    plt.title(r"$Q = " + str(unique_separator.Q) + r"\frac{m^3}{s}$")

plt.legend()
plt.show()
