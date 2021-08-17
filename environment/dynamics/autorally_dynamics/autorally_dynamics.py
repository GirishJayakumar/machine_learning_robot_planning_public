import numpy as np
import scipy.sparse
import torch
from robot_planning.environment.dynamics.simulated_dynamics import NumpySimulatedDynamics
from robot_planning.environment.dynamics.autorally_dynamics import throttle_model
from robot_planning.environment.dynamics.autorally_dynamics import map_coords
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


class LinearizableDynamics(NumpySimulatedDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        NumpySimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.n = self.get_state_dim()[0]
        self.m = self.get_action_dim()[0]
        self.l = self.n

    def linearize_dynamics(self, states, controls, dt=None):
        N = controls.shape[1]
        if dt is None:
            dt = self.get_delta_t()

        delta_x = np.array([0.01, 0.001, 0.01, 0.1, 0.1, 0.05, 0.1, 0.2])
        delta_u = np.array([0.001, 0.01])
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
        fx_plus = self.propagate(x_plus, ux, dt)
        # print(fx_plus)
        fx_minus = self.propagate(x_minus, ux, dt)
        A = (fx_plus - fx_minus) / (2 * delta_x_flat)

        xu = np.tile(states, (self.m, 1)).reshape((self.n, self.m * N), order='F')
        uu = np.tile(controls, (self.m, 1)).reshape((self.m, self.m * N), order='F')
        u_plus = uu + delta_u_final
        # print(xu)
        u_minus = uu - delta_u_final
        fu_plus = self.propagate(xu, u_plus, dt)
        # print(fu_plus)
        fu_minus = self.propagate(xu, u_minus, dt)
        B = (fu_plus - fu_minus) / (2 * delta_u_flat)

        state_row = scipy.sparse.block_diag(list(states.T.reshape((N, self.n, 1)))).toarray()
        input_row = scipy.sparse.block_diag(list(controls.T.reshape((N, self.m, 1)))).toarray()
        d = self.propagate(states, controls, dt) - np.dot(A, state_row) - np.dot(B, input_row)

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


class AutoRallyDynamics(LinearizableDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        LinearizableDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.mass = 0.0
        self.Iz = 0.0
        self.lF = 0.0
        self.lFR = 0.0
        self.IwF = 0.0
        self.IwR = 0.0
        self.rF = 0.0
        self.rR = 0.0
        self.height = 0.0
        self.tire_B = 0.0
        self.tire_C = 0.0
        self.tire_D = 0.0
        self.kSteering = 0.0
        self.cSteering = 0.0
        self.throttle_factor = 0.0
        self.friction_nn = None
        self.throttle_nn = None
        self.track = None

    def initialize_from_config(self, config_data, section_name):
        LinearizableDynamics.initialize_from_config(self, config_data, section_name)
        self.mass = config_data.getfloat(section_name, 'm')
        self.Iz = config_data.getfloat(section_name, 'Iz')
        self.lF = config_data.getfloat(section_name, 'lF')
        self.lFR = config_data.getfloat(section_name, 'lFR')
        self.IwF = config_data.getfloat(section_name, 'IwF')
        self.IwR = config_data.getfloat(section_name, 'IwR')
        self.rF = config_data.getfloat(section_name, 'rF')
        self.rR = config_data.getfloat(section_name, 'rR')
        self.height = config_data.getfloat(section_name, 'h')
        self.tire_B = config_data.getfloat(section_name, 'tire_B')
        self.tire_C = config_data.getfloat(section_name, 'tire_C')
        self.tire_D = config_data.getfloat(section_name, 'tire_D')
        self.kSteering = config_data.getfloat(section_name, 'kSteering')
        self.cSteering = config_data.getfloat(section_name, 'cSteering')
        self.throttle_factor = config_data.getfloat(section_name, 'throttle_factor')
        throttle_nn_path = config_data.get(section_name, 'throttle_nn_path')
        if throttle_nn_path:
            self.throttle_nn = throttle_model.Net()
            self.throttle_nn.load_state_dict(torch.load(throttle_nn_path))
        self.track = map_coords.MapCA(config_data.get(section_name, 'track_path'))

    def propagate(self, state, control, delta_t=None):
        state = state.copy().T
        input = control.copy().T
        m_Vehicle_m = self.mass
        m_Vehicle_Iz = self.Iz
        m_Vehicle_lF = self.lF
        lFR = self.lFR
        m_Vehicle_lR = lFR - m_Vehicle_lF
        m_Vehicle_IwF = self.IwF
        m_Vehicle_IwR = self.IwR
        m_Vehicle_rF = self.rF
        m_Vehicle_rR = self.rR
        m_Vehicle_h = self.height
        m_g = 9.80665

        tire_B = self.tire_B
        tire_C = self.tire_C
        tire_D = self.tire_D

        m_Vehicle_kSteering = self.kSteering
        m_Vehicle_cSteering = self.cSteering
        throttle_factor = self.throttle_factor

        if state.ndim == 1:
            state = state.reshape((1, -1))
            output_flat = True
        else:
            output_flat = False
        if input.ndim == 1:
            input = input.reshape((1, -1))

        vx = state[:, 0]
        vy = state[:, 1]
        wz = state[:, 2]
        wF = state[:, 3]
        wR = state[:, 4]
        e_psi = state[:, 5]
        e_y = state[:, 6]
        s = state[:, 7]

        steering = input[:, 0]
        delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering
        T = np.maximum(input[:, 1], 0)

        deltaT = 0.01
        if delta_t is None:
            dt = self.get_delta_t()
        else:
            dt = delta_t
        t = 0

        while t < dt:
            beta = np.arctan2(vy, vx)

            V = np.sqrt(vx * vx + vy * vy)
            vFx = V * np.cos(beta - delta) + wz * m_Vehicle_lF * np.sin(delta)
            vFy = V * np.sin(beta - delta) + wz * m_Vehicle_lF * np.cos(delta)
            vRx = vx
            vRy = vy - wz * m_Vehicle_lR

            # sEF = -(vFx - wF * m_Vehicle_rF) / (vFx) + tire_Sh
            # muFx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            # sEF = -(vRx - wR * m_Vehicle_rR) / (vRx) + tire_Sh
            # muRx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            #
            # sEF = atan(vFy / abs(vFx)) + tire_Sh
            # alpha = -sEF
            # muFy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            # sEF = atan(vRy / abs(vRx)) + tire_Sh
            # alphaR = -sEF
            # muRy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv

            sFx = np.where(wF > 0, (vFx - wF * m_Vehicle_rF) / (wF * m_Vehicle_rF), 0)
            sRx = np.where(wR > 0, (vRx - wR * m_Vehicle_rR) / (wR * m_Vehicle_rR), 0)
            # sFy = np.where(vFx > 0, (1 + sFx) * vFy / vFx, 0)
            # sRy = np.where(vRx > 0, (1 + sRx) * vRy / vRx, 0)
            sFy = np.where(vFx > 0, vFy / (wF * m_Vehicle_rF), 0)
            sRy = np.where(vRx > 0, vRy / (wR * m_Vehicle_rR), 0)

            sF = np.sqrt(sFx * sFx + sFy * sFy) + 1e-2
            sR = np.sqrt(sRx * sRx + sRy * sRy) + 1e-2

            muF = tire_D * np.sin(tire_C * np.arctan(tire_B * sF))
            muR = tire_D * np.sin(tire_C * np.arctan(tire_B * sR))

            muFx = -sFx / sF * muF
            muFy = -sFy / sF * muF
            muRx = -sRx / sR * muR
            muRy = -sRy / sR * muR

            fFz = m_Vehicle_m * m_g * (m_Vehicle_lR - m_Vehicle_h * muRx) / (
                    m_Vehicle_lF + m_Vehicle_lR + m_Vehicle_h * (muFx * np.cos(delta) - muFy * np.sin(delta) - muRx))
            # fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / 0.57)
            fRz = m_Vehicle_m * m_g - fFz

            fFx = fFz * muFx
            fRx = fRz * muRx
            fFy = fFz * muFy
            fRy = fRz * muRy

            ax = ((fFx * np.cos(delta) - fFy * np.sin(delta) + fRx) / m_Vehicle_m + vy * wz)

            next_state = np.zeros_like(state)
            if self.friction_nn:
                input_tensor = torch.from_numpy(np.vstack((steering, vx, vy, wz, ax, wF, wR)).T).float()
                # input_tensor = torch.from_numpy(input).float()
                forces = self.friction_nn(input_tensor).detach().numpy()
                fafy = forces[:, 0]
                fary = forces[:, 1]
                fafx = forces[0, 2]
                farx = forces[0, 3]

                next_state[:, 0] = vx + deltaT * ((fafx + farx) / m_Vehicle_m + vy * wz)
                next_state[:, 1] = vy + deltaT * ((fafy + fary) / m_Vehicle_m - vx * wz)
                next_state[:, 2] = wz + deltaT * ((fafy) * m_Vehicle_lF - fary * m_Vehicle_lR) / m_Vehicle_Iz
            else:
                next_state[:, 0] = vx + deltaT * (
                            (fFx * np.cos(delta) - fFy * np.sin(delta) + fRx) / m_Vehicle_m + vy * wz)
                next_state[:, 1] = vy + deltaT * ((fFx * np.sin(delta) + fFy * np.cos(delta) + fRy) / m_Vehicle_m - vx * wz)
                next_state[:, 2] = wz + deltaT * (
                        (fFy * np.cos(delta) + fFx * np.sin(delta)) * m_Vehicle_lF - fRy * m_Vehicle_lR) / m_Vehicle_Iz
            next_state[:, 3] = wF - deltaT * m_Vehicle_rF / m_Vehicle_IwF * fFx
            if self.throttle_nn:
                input_tensor = torch.from_numpy(
                    np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))).float()
                next_state[:, 4] = wR + deltaT * self.throttle_nn(input_tensor).detach().numpy().flatten()
            else:
                next_state[:, 4] = T  # wR + deltaT * (m_Vehicle_kTorque * (T-wR) - m_Vehicle_rR * fRx) / m_Vehicle_IwR
            rho = self.track.get_cur_reg_from_s(s)[4].flatten()
            rho = np.zeros_like(s).flatten()
            next_state[:, 5] = e_psi + deltaT * (wz - (vx * np.cos(e_psi) - vy * np.sin(e_psi)) / (1 - rho * e_y) * rho)
            next_state[:, 6] = e_y + deltaT * (vx * np.sin(e_psi) + vy * np.cos(e_psi))
            next_state[:, 7] = s + deltaT * (vx * np.cos(e_psi) - vy * np.sin(e_psi)) / (1 - rho * e_y)

            # if len(cartesian) > 0:
            #     cartesian[0, :] += deltaT * wz
            #     cartesian[1, :] += deltaT * (cos(cartesian[0, :]) * vx - sin(cartesian[0, :]) * vy)
            #     cartesian[2, :] += deltaT * (sin(cartesian[0, :]) * vx + cos(cartesian[0, :]) * vy)

            t += deltaT
            vx = next_state[:, 0]
            vy = next_state[:, 1]
            wz = next_state[:, 2]
            wF = next_state[:, 3]
            wR = next_state[:, 4]
            e_psi = next_state[:, 5]
            e_y = next_state[:, 6]
            s = next_state[:, 7]

        # if len(cartesian) > 0:
        #     return next_state.T, cartesian
        # else:
        if output_flat:
            next_state = next_state.flatten()
        return next_state.T

    def get_state_dim(self):
        return (8,)

    def get_action_dim(self):
        return (2,)

    def get_max_action(self):
        return np.array([1,1])
