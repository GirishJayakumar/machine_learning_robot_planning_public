import numpy as np
import scipy
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import ast


class DynamicsLinearizer():
    def __init__(self, dynamics=None, delta_x=None, delta_u=None):
        self.dynamics = dynamics
        self.delta_x = delta_x
        self.delta_u = delta_u

    def initialize_from_config(self, config_data, section_name):
        pass

    def linearize_dynamics(self, state, controls):
        raise NotImplementedError

    def set_dynamics(self, dynamics):
        self.dynamics = dynamics
        self.n = self.dynamics.get_state_dim()[0]
        self.m = self.dynamics.get_action_dim()[0]
        self.l = self.n  # TODO: Ji:what is self.l?

    def set_delta_x(self, delta_x):
        self.delta_x = delta_x

    def set_delta_u(self, delta_u):
        self.delta_u = delta_u


class NumpyDynamicsLinearizer(DynamicsLinearizer):
    def __init__(self, dynamics=None, delta_x=None, delta_u=None):
        DynamicsLinearizer.__init__(self, dynamics, delta_x, delta_u)

    def initialize_from_config(self, config_data, section_name):
        DynamicsLinearizer.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.dynamics = factory_from_config(dynamics_factory_base, config_data, dynamics_section_name)
        self.n = self.dynamics.get_state_dim()[0]
        self.m = self.dynamics.get_action_dim()[0]
        self.l = self.n  # TODO: Ji:what is self.l?
        if config_data.has_option(section_name, 'delta_x'):
            self.delta_x = np.asarray(ast.literal_eval(config_data.get(section_name, 'delta_x')), dtype=np.float64).reshape((1, self.n))
        if config_data.has_option(section_name, 'delta_u'):
            self.delta_u = np.asarray(ast.literal_eval(config_data.get(section_name, 'delta_u')), dtype=np.float64).reshape((1, self.m))

    def linearize_dynamics(self, states, controls, dt=None):
        N = controls.shape[1]
        if dt is None:
            dt = self.dynamics.get_delta_t()
        if self.delta_x is None:
            delta_x = 0.01*np.ones(self.dynamics.get_state_dim())
        else:
            delta_x = self.delta_x
        if self.delta_u is None:
            delta_u = 0.01*np.ones(self.dynamics.get_action_dim())
        else:
            delta_u = self.delta_u

        delta_x_flat = np.tile(delta_x, (1, N))
        delta_u_flat = np.tile(delta_u, (1, N))
        # delta_x_flat = (states / 100.0).reshape((1, -1))
        # delta_u_flat = (controls / 100.0).reshape((1, -1))

        delta_x_final = np.multiply(np.tile(np.eye(self.n), (1, N)), delta_x_flat)
        delta_u_final = np.multiply(np.tile(np.eye(self.m), (1, N)), delta_u_flat)
        xx = np.tile(states, (self.n, 1)).reshape((self.n, self.n * N), order='F')
        # print(delta_x_final, xx)
        ux = np.tile(controls, (self.n, 1)).reshape((self.m, self.n*N), order='F')
        x_plus = xx + delta_x_final
        # print(x_plus, ux)
        x_minus = xx - delta_x_final
        fx_plus = self.dynamics.propagate(x_plus, ux, dt)
        # print(fx_plus)
        fx_minus = self.dynamics.propagate(x_minus, ux, dt)
        A = (fx_plus - fx_minus) / (2 * delta_x_flat)

        xu = np.tile(states, (self.m, 1)).reshape((self.n, self.m * N), order='F')
        uu = np.tile(controls, (self.m, 1)).reshape((self.m, self.m * N), order='F')
        u_plus = uu + delta_u_final
        # print(xu)
        u_minus = uu - delta_u_final
        fu_plus = self.dynamics.propagate(xu, u_plus, dt)
        # print(fu_plus)
        fu_minus = self.dynamics.propagate(xu, u_minus, dt)
        B = (fu_plus - fu_minus) / (2 * delta_u_flat)

        state_row = scipy.sparse.block_diag(list(states.T.reshape((N, self.n, 1)))).toarray()
        input_row = scipy.sparse.block_diag(list(controls.T.reshape((N, self.m, 1)))).toarray()
        d = self.dynamics.propagate(states, controls, dt) - np.dot(A, state_row) - np.dot(B, input_row)

        return A, B, d

    def form_long_matrices_LTI(self, A, B, D):
        N = int(A.shape[1] / A.shape[0])

        AA = np.zeros((self.n*N, self.n))
        BB = np.zeros((self.n*N, self.m * N))
        DD = np.zeros((self.n, self.n * N))
        B_i_row = np.zeros((self.n, 0))
        # D_i_bar = zeros((nx, nx))
        for ii in np.arange(0, N):
            AA[ii*self.n:(ii+1)*self.n, :] = np.linalg.matrix_power(A, ii+1)

            B_i_cell = np.dot(np.linalg.matrix_power(A, ii), B)
            B_i_row = np.hstack((B_i_cell, B_i_row))
            BB[ii*self.n:(ii+1)*self.n, :(ii+1)*self.m] = B_i_row

            # D_i_bar = np.hstack((np.dot(np.linalg.matrix_power(A, ii - 1), D), D_i_bar))
            # temp = np.hstack((D_i_bar, np.zeros((nx, max(0, nx * N - D_i_bar.shape[1])))))
            # DD = np.vstack((DD, temp[:, 0: nx * N]))

        return AA, BB, DD

    def form_long_matrices_LTV(self, A, B, d, D):
        N = A.shape[2]

        AA = np.zeros((self.n * N, self.n))
        BB = np.zeros((self.n * N, self.m * N))
        DD = np.zeros((self.n * N, self.l * N))
        dd = np.zeros((self.n * N, 1))
        AA_i_row = np.eye(self.n)
        dd_i_row = np.zeros((self.n, 1))
        # B_i_row = zeros((nx, 0))
        # D_i_bar = zeros((nx, nx))
        for ii in np.arange(0, N):
            AA_i_row = np.dot(A[:, :, ii], AA_i_row)
            AA[ii*self.n:(ii+1) * self.n, :] = AA_i_row

            B_i_row = B[:, :, ii]
            D_i_row = D[:, :, ii]
            for jj in np.arange(ii-1, -1, -1):
                B_i_cell = np.dot(A[:, :, ii], BB[(ii-1)*self.n:ii*self.n, jj*self.m:(jj+1)*self.m])
                B_i_row = np.hstack((B_i_cell, B_i_row))
                D_i_cell = np.dot(A[:, :, ii], DD[(ii-1)*self.n:ii*self.n, jj*self.l:(jj+1)*self.l])
                D_i_row = np.hstack((D_i_cell, D_i_row))
            BB[ii*self.n:(ii+1)*self.n, :(ii+1)*self.m] = B_i_row
            DD[ii*self.n:(ii+1)*self.n, :(ii+1)*self.l] = D_i_row

            dd_i_row = np.dot(A[:, :, ii], dd_i_row) + d[:, :, ii]
            dd[ii*self.n:(ii+1)*self.n, :] = dd_i_row

        return AA, BB, dd, DD