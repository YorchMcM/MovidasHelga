from environments import *


class TerminationCondition:
    def assess_termination(self, time: float, state: np.ndarray) -> bool:
        return False


class TimeTermination(TerminationCondition):
    def __init__(self, termination_time: float):
        self.termination_time = termination_time
        return

    def assess_termination(self, time: float, state: np.ndarray) -> bool:
        return time >= self.termination_time


class YLimTermination(TerminationCondition):
    def __init__(self, limits: list[float]):
        self.limits = limits

        return

    def assess_termination(self, time, state) -> bool:
        return state[1] < self.limits[0] or state[1] >= self.limits[1]


class XLimTermination(TerminationCondition):
    def __init__(self, limits: list[float]):
        self.limits = limits

        return

    def assess_termination(self, time, state) -> bool:
        return state[0] <= self.limits[0] or state[0] > self.limits[1]


class MathematicalModel:
    def __init__(self, separator: Separator, particle: Particle):
        self.separator = separator
        self.particle = particle

        self.S_h = np.pi * separator.D_h**2 / 4
        self.S_l = np.pi * separator.D_l**2 / 4

        self.Q_h = separator.X_h * separator.Q
        self.Q_l = separator.X_l * separator.Q

        self.R = np.sqrt(
            separator.Rt**2
            + (
                (separator.rho_l * self.Q_l**2)
                / (particle.rho_h * self.S_l**2 * separator.omega**2)
            )
            - (self.Q_h**2 / (self.S_h**2 * separator.omega**2))
        )
        return

    def compute_reynolds_number(self, hydrodynamic_velocity: float) -> float:
        return (
            self.separator.rho_l
            * self.particle.Dp
            * hydrodynamic_velocity
            / self.separator.mu_l
        )

    def compute_derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, u, v = state
        z = (self.particle.rho_h - self.separator.rho_l) / self.particle.rho_h
        hydrodynamic_velocity = u + self.separator.compute_couette_velocity(state[:2])
        reynolds_number = self.compute_reynolds_number(abs(hydrodynamic_velocity))

        # Here we decide between using K, N1 or N2
        if reynolds_number < 1.0:
            premultiplicative_constant = (-18.0 * self.separator.mu_l) / (
                self.separator.rho_l * self.particle.Dp**2.0
            )

        elif 1 <= reynolds_number < 5:
            premultiplicative_constant = (
                (-3.0 * np.pi * self.separator.mu_l * self.particle.Dp)
                / abs(hydrodynamic_velocity)
                - 9 * np.pi * self.separator.rho_l * self.particle.Dp**2 / 16
            ) * np.sqrt(hydrodynamic_velocity**2 + v**2)
            premultiplicative_constant = premultiplicative_constant / self.particle.mass

        else:
            premultiplicative_constant = (
                (
                    1.85
                    * (
                        self.separator.rho_l
                        * self.particle.Dp
                        * abs(hydrodynamic_velocity)
                        / self.separator.mu_l
                    )
                    ** (-0.6)
                )
                * np.pi
                * self.particle.Dp**2.0
                * self.separator.rho_l
                / 8.0
            )

            premultiplicative_constant = (
                -1.0 * premultiplicative_constant / self.particle.mass
            )

        x_dot_dot = (
            z
            * (
                x * np.sin(np.deg2rad(self.separator.phi))
                + y * np.cos(np.deg2rad(self.separator.phi))
            )
            * self.separator.omega**2
            * np.sin(np.deg2rad(self.separator.phi))
        )
        # x_dot_dot = x_dot_dot + premultiplicative_constant*hydrodynamic_velocity
        x_dot_dot = x_dot_dot + premultiplicative_constant * hydrodynamic_velocity

        y_dot_dot = (
            z
            * (
                x * np.sin(np.deg2rad(self.separator.phi))
                + y * np.cos(np.deg2rad(self.separator.phi))
            )
            * self.separator.omega**2
            * np.cos(np.deg2rad(self.separator.phi))
        )
        y_dot_dot = y_dot_dot + premultiplicative_constant * v

        if y <= 0.0 and y_dot_dot < 0.0:
            # In this case, the flow is squeezing the particle against the lower plate.
            y_dot_dot = 0.0
        elif y >= self.separator.h and y_dot_dot > 0.0:
            # In this case, the flow is squeezing the particle against the upper plate.
            y_dot_dot = 0.0
        else:
            pass

        return np.array([u, v, x_dot_dot, y_dot_dot])


class RK_integrator:
    def __init__(self, time_step: float):
        self.time_step = time_step
        return

    def advance_step(self, model: MathematicalModel, state: np.ndarray) -> np.ndarray:
        k1 = model.compute_derivatives(state)
        k2 = model.compute_derivatives(state + k1 * self.time_step / 2)
        k3 = model.compute_derivatives(state + k2 * self.time_step / 2)
        k4 = model.compute_derivatives(state + k3 * self.time_step)

        return state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * self.time_step / 6.0


class Simulator:
    def __init__(self, model: MathematicalModel, integrator: RK_integrator):
        self.model = model
        self.integrator = integrator
        return

    def bring_back_to_plate(
        self, state_inside_plates: np.ndarray, state_outside_plates: np.ndarray
    ) -> np.ndarray:
        y_in, y_out = state_inside_plates[1], state_outside_plates[1]

        if y_out > self.model.separator.h:
            y_critical = self.model.separator.h
        else:
            y_critical = 0.0

        fraction = (y_critical - y_in) / (y_out - y_in)
        collided_state = state_inside_plates + fraction * (
            state_outside_plates - state_inside_plates
        )
        # Here we just say that the velocity in the y direction is 0, because the droplet just collided with the plate.
        collided_state[3] = 0.0

        return collided_state

    def simulate_trajectory(
        self,
        initial_time: float,
        initial_state: np.ndarray,
        termination_conditions: list[TerminationCondition],
    ) -> np.ndarray:
        there_is_time_termination = False
        for condition in termination_conditions:
            if type(condition) is TimeTermination:
                final_time = condition.termination_time
                there_is_time_termination = True

        if not there_is_time_termination:
            raise TypeError(
                "(Simulator.simulate_trajectory): No time termination encountered. Simulation is currently "
                "only possible in the presence of a time termination condition."
            )

        t = np.arange(initial_time, final_time, self.integrator.time_step)
        history = np.full([len(t), 5], np.nan)
        history[:, 0] = t
        history[0, 1:] = initial_state

        is_terminated = False

        for idx in range(1, len(t)):
            history[idx, 1:] = self.integrator.advance_step(
                self.model, history[idx - 1, 1:]
            )

            if history[idx, 2] > self.model.separator.h or history[idx, 2] < 0.0:
                history[idx, 1:] = self.bring_back_to_plate(
                    history[idx - 1, 1:], history[idx, 1:]
                )

            for condition in termination_conditions:
                if condition.assess_termination(history[idx, 0], history[idx, 1:]):
                    is_terminated = True

            if is_terminated:
                break

        return self.remove_nans(history)

    def remove_nans(self, history: np.ndarray) -> np.ndarray:
        while not sum(history[-1, :] == history[-1, :]) == 5:
            history = history[:-1, :]

        return history
