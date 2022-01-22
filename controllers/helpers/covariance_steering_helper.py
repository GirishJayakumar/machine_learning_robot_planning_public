import numpy as np
from robot_planning.factory.factories import dynamics_linearizer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import cvxpy as cp
import ast


class CvxpyCovarianceSteeringHelper():
    def __init__(self, state_dim=None, control_dim=None, dynamics_linearizer=None):
        self.n = state_dim
        self.m = control_dim
        self.dynamics_linearizer = dynamics_linearizer

    def initialize_from_config(self, config_data, section_name):
        dynamics_linearizer_section_name = config_data.get(section_name, 'dynamics_linearizer')
        self.dynamics_linearizer = factory_from_config(dynamics_linearizer_factory_base, config_data, dynamics_linearizer_section_name)
        self.set_dynamics_linearizer(self.dynamics_linearizer)
        if config_data.has_option(section_name, 'Qf_val'):
            self.Qf_val = config_data.getfloat(section_name, 'Qf_val')
        if config_data.has_option(section_name, 'R_val'):
            self.R_val = config_data.getfloat(section_name, 'R_val')
    def set_dynamics_linearizer(self, dynamics_linearizer):
        self.dynamics_linearizer = dynamics_linearizer
        self.n = self.dynamics_linearizer.n
        self.m = self.dynamics_linearizer.m

    def covariance_control(self, state, ref_state_vec, ref_ctrl_vec, return_sx=False, Sigma_epsilon=np.asarray([[0.49, 0.0], [0.0, 0.1218]])):
        n = ref_state_vec.shape[1]
        m = ref_ctrl_vec.shape[1]
        N = ref_ctrl_vec.shape[0]

        As, Bs, ds = self.dynamics_linearizer.linearize_dynamics(ref_state_vec, ref_ctrl_vec)
        A, B, C, d, D = self.dynamics_linearizer.make_batch_dynamics(As, Bs, ds, Sigma_epsilon=Sigma_epsilon)

        # cost matrix
        # soft constraint Q matrix
        Q_bar = np.zeros([(N + 1) * self.n, (N + 1) * self.n])
        Q_bar[-self.n:, -self.n:] = np.eye(self.n) * self.Qf_val

        R = np.eye(m) * self.R_val
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal covariance constrain
        # not needed with soft constraint
        # sigma_f = self.sigma_f

        # setup cvxpy
        I = np.eye(n * (N + 1))
        E_N = np.zeros((n, n * (N + 1)))
        E_N[:, n * (N):] = np.eye(n)

        # assemble K as a diagonal block matrix with K_0..K_N-1 as var
        Ks = [cp.Variable((m, n)) for i in range(N)]
        # K dim: mN x n(N+1)
        K = cp.hstack([Ks[0], np.zeros((m, (N) * n))])
        for i in range(1, N):
            line = cp.hstack([np.zeros((m, n * i)), Ks[i], np.zeros((m, (N - i) * n))])
            K = cp.vstack([K, line])

        objective = cp.Minimize(cp.norm(cp.vec(R_bar_sqrt @ K @ D)) + cp.norm(cp.vec(Q_bar_sqrt @ (I + B @ K) @ D)))

        # TODO verify with Ji
        sigma_y_sqrt = self.nearest_spd_cholesky(D @ D.T)
        # hard constraint, cvxpy doesn't respect this for some reasons
        # constraints = [cp.bmat([[sigma_f, E_N @(I+B@K)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K).T@E_N.T, I ]]) >= 0]
        constraints = []
        prob = cp.Problem(objective, constraints)
        J = prob.solve()
        Ks = np.array([val.value for val in Ks])
        self.Ks = Ks

        As = np.swapaxes(As, 0, 2)
        As = np.swapaxes(As, 1, 2)

        Bs = np.swapaxes(Bs, 0, 2)
        Bs = np.swapaxes(Bs, 1, 2)

        ds = np.swapaxes(ds, 0, 2)
        ds = np.swapaxes(ds, 1, 2)

        # return terminal covariance, theoretical values with and without cc
        if (return_sx):
            reconstruct_K = np.hstack([Ks[0], np.zeros((m, (N) * n))])
            for i in range(1, N):
                line = np.hstack([np.zeros((m, n * i)), Ks[i], np.zeros((m, (N - i) * n))])
                reconstruct_K = np.vstack([reconstruct_K, line])
            Sigma_0 = np.zeros([n, n])
            # Sx_cc = (I + B@K.value ) @ (A @ Sigma_0 @ A.T + D @ D.T ) @ (I + B@K.value ).T
            Sx_cc = (I + B @ reconstruct_K) @ (A @ Sigma_0 @ A.T + D @ D.T) @ (I + B @ reconstruct_K).T
            Sx_nocc = (A @ Sigma_0 @ A.T + D @ D.T)
            return Ks, As, Bs, ds, Sx_cc, Sx_nocc
        else:
            return Ks, As, Bs, ds

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