from robot_planning.environment.dynamics.bicycle_dynamics import BicycleDynamics
import numpy as np


class RcCarDynamics(BicycleDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None, cog_pos=0.4, car_length=0.09):
        BicycleDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.dynamics_type = dynamics_type
        self.data_type = data_type
        self.delta_t = delta_t
        self.cog_pos = 0.4
        self.car_length = 0.09
        self.set_parameters(self.car_length, self.cog_pos)

    def initialize_from_config(self, config_data, section_name):
        BicycleDynamics.initialize_from_config(self, config_data, section_name)
        self.cog_pos = config_data.getfloat(section_name, 'cog_pos')
        self.car_length = config_data.getfloat(section_name, 'car_length')
        self.set_parameters(self.car_length, self.cog_pos)

    def set_parameters(self, car_length, cog_pos):
        self.lf = car_length * cog_pos
        self.lr = (1 - car_length) * cog_pos

    def propagate(self, state, action, dt=None):
        lf = self.lf
        lr = self.lr
        x0 = state
        u0 = action
        psi = x0[3]
        V = x0[2]
        throttle = u0[0]
        steering = u0[1]
        beta = np.arctan(np.tan(steering)*lr/(lr+lf))
        dX = V*np.cos(psi + beta) * dt
        dY = V*np.sin(psi + beta) * dt
        dV = throttle * dt
        dheading = V/lr*np.sin(beta) * dt

        x0[0] += dX
        x0[1] += dY
        x0[2] += dV
        x0[3] += dheading
        state_next = x0
        return state_next

    def get_state_dim(self):
        return (4,)

    def get_action_dim(self):
        return (2,)

    def get_max_action(self):
        pass

    def get_delta_t(self):
        return 0.02
