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


class CCMPPINumpyDynamicsLinearizer(DynamicsLinearizer):
    def __init__(self, dynamics=None, delta_x=None, delta_u=None, state_dim=None, control_dim=None, sigma_epsilon=None):
        DynamicsLinearizer.__init__(self, dynamics, delta_x, delta_u)
        self.n = state_dim
        self.m = control_dim
        self.l = self.n
        self.sigma_epsilon = sigma_epsilon

    def initialize_from_config(self, config_data, section_name):
        DynamicsLinearizer.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.dynamics = factory_from_config(dynamics_factory_base, config_data, dynamics_section_name)
        self.n = self.dynamics.get_state_dim()[0]
        self.m = self.dynamics.get_action_dim()[0]
        self.l = self.n
        if config_data.has_option(section_name, 'delta_x'):
            self.delta_x = np.asarray(ast.literal_eval(config_data.get(section_name, 'delta_x')), dtype=np.float64).reshape((1, self.n))
        if config_data.has_option(section_name, 'delta_u'):
            self.delta_u = np.asarray(ast.literal_eval(config_data.get(section_name, 'delta_u')), dtype=np.float64).reshape((1, self.m))

    def linearize_dynamics(self, ref_state_vec, ref_ctrl_vec, dt=None):
        if dt is None:
            dt = self.dynamics.get_delta_t()
        N = ref_ctrl_vec.shape[0]
        As = []
        Bs = []
        ds = []
        for i in range(N):
            # NOTE this gives discretized dynamics
            A, B, d = self.linearize(ref_state_vec[i, :], ref_ctrl_vec[i, :], dt)
            As.append(A)
            Bs.append(B)
            ds.append(d)
        As = np.dstack(As)
        Bs = np.dstack(Bs)
        ds = np.dstack(ds).reshape((self.n, 1, N))
        return As, Bs, ds

    def make_batch_dynamics(self, As, Bs, ds, Sigma_epsilon = np.asarray([[0.49, 0.0], [0.0, 0.1218]])):
        n = self.n
        l = self.l
        m = self.m
        N = As.shape[2]
        assert (Sigma_epsilon.shape == (m, m))
        # A: (N+1)n x n
        I = np.eye(n)
        A = [I]
        # row 1 to row N+1
        # i -> row i+1
        for i in range(N):
            A.append(As[:,:,i] @ A[-1])
        A = np.vstack(A)

        # B (N+1)n x Nm
        row0 = np.zeros((n,m*N))
        B = [row0]
        # row 1 to row N
        for i in range(1,N+1):
            row_i = B[-1].copy()
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*m:i*m] = Bs[:,:,i-1]
            B.append(row_i)
        B = np.vstack(B)

        # C: n(N+1) x nN
        row0 = np.zeros((n, n*N))
        C = [row0]
        for i in range(1,N+1):
            row_i = C[-1].copy()
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*n:i*n] = np.eye(n)
            C.append(row_i)
        C = np.vstack(C)

        # d
        d = ds[:,0,:].T.flatten()

        Sigma_epsilon_half = self.nearest_spd_cholesky(Sigma_epsilon)
        row0 = np.zeros((n,m*N))
        D = [row0]
        # row 1 to row N
        for i in range(1,N+1):
            row_i = D[-1].copy()
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*m:i*m] = Bs[:,:,i-1] @ Sigma_epsilon_half
            D.append(row_i)
        D = np.vstack(D)
        return A, B, C, d, D

    def linearize(self, nominal_state, nominal_ctrl, dt):
        nominal_state = np.array(nominal_state).copy()
        nominal_ctrl = np.array(nominal_ctrl).copy()
        epsilon = 1e-2

        # A = df/dx
        A = np.zeros((self.n, self.n), dtype=np.float)
        # find A
        for i in range(self.n):
            # d x / d x_i, ith row in A
            x_l = nominal_state.copy()
            x_l[i] -= epsilon

            x_post_l = self.dynamics.propagate(x_l, nominal_ctrl, dt)

            x_r = nominal_state.copy()
            x_r[i] += epsilon
            x_post_r = self.dynamics.propagate(x_l, nominal_ctrl, dt)

            A[:, i] += (x_post_r.flatten() - x_post_l.flatten()) / (2 * epsilon)

        # B = df/du
        B = np.zeros((self.n, self.m), dtype=np.float)
        # find B
        for i in range(self.m):
            # d x / d u_i, ith row in B
            x0 = nominal_state.copy()
            u_l = nominal_ctrl.copy()
            u_l[i] -= epsilon
            x_post_l = self.dynamics.propagate(x0, u_l, dt)
            x_post_l = x_post_l.copy()

            x0 = nominal_state.copy()
            u_r = nominal_ctrl.copy()
            u_r[i] += epsilon
            x_post_r = self.dynamics.propagate(x0, u_r, dt)
            x_post_r = x_post_r.copy()

            B[:, i] += (x_post_r.flatten() - x_post_l.flatten()) / (2 * epsilon)

        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        x_post = self.dynamics.propagate(x0, u0, dt)
        # d = x_k+1 - Ak*x_k - Bk*u_k
        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        d = x_post.flatten() - A @ x0 - B @ u0
        return A, B, d

    def nearest_spd_cholesky(self,A):
        B = (A + A.T)/2
        U, Sigma, V = np.linalg.svd(B)
        H = np.dot(np.dot(V.T, np.diag(Sigma)), V)
        Ahat = (B+H)/2
        Ahat = (Ahat + Ahat.T)/2
        p = 1
        k = 0
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        while p != 0:
            k += 1
            try:
                R = np.linalg.cholesky(Ahat)
                p = 0
            except np.linalg.LinAlgError:
                eig = np.linalg.eigvals(Ahat)
                mineig = np.min(np.real(eig))
                Ahat = Ahat + I * (-mineig * k**2 + spacing)
        R_old = R.copy()
        R[np.abs(R) < 1e-5] = 1e-5
        np.tril(R)
        return R