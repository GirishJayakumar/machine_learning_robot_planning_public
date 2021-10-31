from robot_planning.controllers.controller import MpcController
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import stochastic_trajectories_sampler_factory_base
from robot_planning.factory.factories import dynamics_linearizer_factory_base
from robot_planning.controllers.CSSMPC import cs_solver
import ast
import numpy as np


class CSSMPC(MpcController):
    def __init__(self, control_horizon=None, dynamics=None, cost_evaluator=None, control_dim=None, inverse_temperature=None, initial_control_sequence=None, stochastic_trajectories_sampler=None, renderer=None):
        MpcController.__init__(self, control_horizon, dynamics, cost_evaluator, control_dim, renderer)
        self.initial_control_sequence = initial_control_sequence
        self.solver = None
        self.N = 0
        self.n = 0
        self.m = 0
        self.l = 0
        self.dt = 0
        self.target_speed = 0.0
        self.reference_traj_to_track = np.empty(())
        self.u_range = np.empty(())
        self.slew_rate = np.empty(())
        self.prob_lvl = 0.0
        self.load_k = 0
        self.track_w = 0.0
        self.Q_bar = np.empty(())
        self.R_bar = np.empty(())

    def initialize_from_config(self, config_data, section_name):
        MpcController.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, 'initial_control_sequence'):
            self.initial_control_sequence = np.asarray(ast.literal_eval(config_data.get(section_name, 'initial_control_sequence')), dtype=np.float64).reshape((self.get_control_dim(), self.get_control_horizon()))
            if not (self.initial_control_sequence.shape[0] is self.get_control_dim() and self.initial_control_sequence.shape[1] is self.get_control_horizon()):
                raise ValueError('The initial control sequence does not match control dimensions and control horizon')
        else:
            self.initial_control_sequence = np.zeros((self.get_control_dim(), self.get_control_horizon()))
        dynamics_linearizer_section_name = config_data.get(section_name, 'dynamics_linearizer')
        self.dynamics_linearizer =factory_from_config(dynamics_linearizer_factory_base, config_data, dynamics_linearizer_section_name)
        self.N = self.get_control_horizon()
        self.n = self.dynamics.get_state_dim()[0]
        self.m = self.get_control_dim()
        self.l = self.n
        self.dt = self.dynamics.get_delta_t()
        self.reference_traj_to_track = np.asarray(ast.literal_eval((config_data.get(section_name, 'reference_traj_to_track'))))
        self.u_range = np.asarray(ast.literal_eval((config_data.get(section_name, 'ctrl_range'))))
        self.slew_rate = np.asarray(ast.literal_eval((config_data.get(section_name, 'ctrl_slew_rate'))))
        self.prob_lvl = float(config_data.get(section_name, 'prob_lvl'))
        self.load_k = int(config_data.get(section_name, 'load_k'))
        self.track_w = float(config_data.get(section_name, 'track_w'))
        Q = np.asarray(ast.literal_eval((config_data.get(section_name, 'Q'))))
        Q = np.diag(Q)
        QN = np.asarray(ast.literal_eval((config_data.get(section_name, 'QN'))))
        QN = np.diag(QN)
        self.Q_bar = np.kron(np.eye(self.N, dtype=int), Q)
        self.Q_bar[-self.n:, -self.n:] = QN
        R = np.asarray(ast.literal_eval((config_data.get(section_name, 'R'))))
        R = np.diag(R)
        self.R_bar = np.kron(np.eye(self.N, dtype=int), R)
        self.solver = cs_solver.CSSolver(self.n, self.m, self.l, self.N, self.u_range, self.slew_rate, (False, ),
                                         mean_only=True, k_form=1, prob_lvl=self.prob_lvl, chance_const_N=self.N)

    def plan(self, state_cur):
        state_cur = state_cur.reshape((-1, 1))
        us = self.initial_control_sequence.copy()
        xs = self.roll_out(state_cur, us, self.dt)
        A, B, d = self.dynamics_linearizer.linearize_dynamics(xs[:, :-1], us, dt=self.dt)
        A = A.reshape((self.n, self.n, self.N), order='F')
        B = B.reshape((self.n, self.m, self.N), order='F')
        d = d.reshape((self.n, 1, self.N), order='F')
        D = np.zeros((self.n, self.l))
        D = np.tile(D.reshape((self.n, self.l, 1)), (1, 1, self.N))
        A, B, d, D = self.dynamics_linearizer.form_long_matrices_LTV(A, B, d, D)
        sigma_0 = np.zeros((self.n, self.n))
        mu_N = np.ones((self.n, 1)) * 9999
        self.solver.populate_params(A, B, d, D, xs[:, 0], sigma_0, sigma_0, self.Q_bar, self.R_bar, us[:, 0],
                                    self.reference_traj_to_track.reshape((-1, 1)), mu_N, self.track_w, K=np.zeros((self.m*self.N, self.n*self.N)))
        V, K = self.solver.solve()
        K = K.reshape((self.m * self.N, self.n * self.N))
        us = V.reshape((self.m, self.N), order='F')
        X_bar = np.dot(A, xs[:, 0]) + np.dot(B, V) + d.flatten()
        u = V[:self.m]  # + np.dot(K[jj*m:(jj+1)*m, jj*n:(jj+1)*n], y)
        u = np.where(u > self.u_range[:, 1], self.u_range[:, 1], u)
        u = np.where(u < self.u_range[:, 0], self.u_range[:, 0], u)
        xs = X_bar.reshape((self.n, self.N), order='F')
        if self.renderer is not None:
            # self.renderer.render_trajectories(trajectories, **{'color': "b"})
            self.renderer.render_trajectories([xs], **{'color': "r"})
        # us = np.delete(us, 0, 1)
        # us = np.hstack((us, us[:, -1].reshape(us.shape[0], 1)))
        self.set_initial_control_sequence(us)
        return u

    def set_initial_control_sequence(self, initial_control_sequence):
        self.initial_control_sequence = initial_control_sequence

    def roll_out(self, state_cur, us, dt):
        dynamics = self.get_dynamics()
        trajectory = np.zeros((dynamics.get_state_dim()[0], us.shape[1]+1))
        trajectory[:, 0] = state_cur.flatten()
        for ii in range(us.shape[1]):
            state_cur = dynamics.propagate(state_cur, us[:, ii:ii+1], dt)
            trajectory[:, ii+1] = state_cur.flatten()
        return trajectory
