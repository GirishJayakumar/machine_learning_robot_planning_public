from robot_planning.controllers.controller import MpcController
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
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
        self.goal_traj = np.empty(())
        self.reference_traj_to_track = np.empty(())
        self.u_range = np.empty(())
        self.slew_rate = np.empty(())
        self.prob_lvl = 0.0
        self.load_k = 0
        self.track_w = 0.0
        self.Q_bar = np.empty(())
        self.R_bar = np.empty(())
        self.X_bar = np.empty(())

    def initialize_from_config(self, config_data, section_name):
        MpcController.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.dynamics = factory_from_config(dynamics_factory_base, config_data, dynamics_section_name)
        dynamics_linearizer_section_name = config_data.get(section_name, 'dynamics_linearizer')
        self.dynamics_linearizer =factory_from_config(dynamics_linearizer_factory_base, config_data, dynamics_linearizer_section_name)
        self.N = self.get_control_horizon()
        self.n = self.dynamics.get_state_dim()[0]
        self.m = self.get_control_dim()
        self.l = self.n
        self.dt = self.dynamics.get_delta_t()
        init_ctrl_seq = np.asarray(ast.literal_eval(config_data.get(section_name, 'initial_control_sequence')), dtype=np.float64)
        if init_ctrl_seq.shape[0] == self.N * self.m:
            self.initial_control_sequence = init_ctrl_seq.reshape((self.get_control_dim(), self.get_control_horizon()))
        else:
            self.initial_control_sequence = np.tile(init_ctrl_seq.reshape((-1, 1)), (1, self.N))
        self.goal_traj = np.tile(np.asarray(ast.literal_eval(config_data.get(section_name, 'goal_state')), dtype=np.float64).reshape((-1, 1)), (1, self.N))
        self.goal_traj[:, -1] = np.asarray(ast.literal_eval(config_data.get(section_name, 'goal_terminal_state')), dtype=np.float64)
        self.u_range = np.asarray(ast.literal_eval((config_data.get(section_name, 'ctrl_range'))))
        self.slew_rate = np.asarray(ast.literal_eval((config_data.get(section_name, 'ctrl_slew_rate'))))
        self.prob_lvl = float(config_data.get(section_name, 'prob_lvl'))
        self.load_k = int(config_data.get(section_name, 'load_k'))
        self.track_w = float(config_data.get(section_name, 'track_w'))
        self.Q_bar = np.kron(np.eye(self.N, dtype=int), self.cost_evaluator.Q)
        self.R_bar = np.kron(np.eye(self.N, dtype=int), self.cost_evaluator.R)
        self.X_bar = np.zeros((self.n*self.N, 1)).flatten()
        self.solver = cs_solver.CSSolver(self.n, self.m, self.l, self.N, self.u_range, self.slew_rate, (False, ),
                                         mean_only=False, k_form=1, prob_lvl=self.prob_lvl, chance_const_N=self.N)

    def plan(self, state_cur):
        state_cur = state_cur.reshape((-1, 1))
        e_psi, e_y, s = self.dynamics.track.localize(np.array((state_cur[-2, :], state_cur[-1, :])), state_cur[-3, :])
        state_cur_map = state_cur.copy()
        state_cur_map[5:, :] = np.vstack((e_psi, e_y, s))
        us = self.initial_control_sequence.copy()
        xs = self.roll_out(state_cur_map, us, self.dt)
        A, B, d = self.dynamics_linearizer.linearize_dynamics(xs[:, :-1], us, dt=self.dt)
        A = A.reshape((self.n, self.n, self.N), order='F')
        B = B.reshape((self.n, self.m, self.N), order='F')
        d = d.reshape((self.n, 1, self.N), order='F')
        y = state_cur_map.flatten() - self.X_bar[0:self.n]
        print('y:', y)
        # D = np.zeros((self.n, self.l))
        D = np.diag(y)
        D = np.tile(D.reshape((self.n, self.l, 1)), (1, 1, self.N))
        A, B, d, D = self.dynamics_linearizer.form_long_matrices_LTV(A, B, d, D)
        sigma_0 = np.zeros((self.n, self.n))
        mu_N = np.ones((self.n, 1)) * 9999
        self.solver.populate_params(A, B, d, D, xs[:, 0], sigma_0, sigma_0, self.Q_bar, self.R_bar, us[:, 0],
                                    self.goal_traj.reshape((-1, 1), order='F'), mu_N, self.track_w, K=np.zeros((self.m*self.N, self.n*self.N)))
        V, K = self.solver.solve()
        K = K.reshape((self.m * self.N, self.n * self.N))
        us = V.reshape((self.m, self.N), order='F')
        sigma_y = np.dot(np.dot(A, sigma_0), A.T) + np.dot(D, D.T)
        sigma_X = np.dot(np.dot(np.eye(self.n*self.N) + np.dot(B, K), sigma_y), (np.eye(self.n*self.N) + np.dot(B, K)).T)
        self.X_bar = np.dot(A, xs[:, 0]) + np.dot(B, V) + d.flatten()
        u = V[:self.m]  # + np.dot(K[jj*m:(jj+1)*m, jj*n:(jj+1)*n], y)
        u = np.where(u > self.u_range[:, 1], self.u_range[:, 1], u)
        u = np.where(u < self.u_range[:, 0], self.u_range[:, 0], u)
        xs = self.X_bar.reshape((self.n, self.N), order='F')
        self.set_initial_control_sequence(us)

        dynamics = self.get_dynamics()
        optimal_trajectory = np.zeros((self.n, self.N))
        optimal_trajectory[:, 0:1] = dynamics.propagate(state_cur, us[:, 0:0 + 1], self.dt, cartesian=True)
        for ii in np.arange(1, self.N):
            optimal_trajectory[:, ii:ii+1] = dynamics.propagate(optimal_trajectory[:, ii-1:ii], us[:, ii:ii+1], self.dt, cartesian=True)
        if self.renderer is not None:
            # self.renderer.render_trajectories(trajectories, **{'color': "b"})
            self.renderer.render_trajectories([optimal_trajectory], **{'color': "r"})
        return u

    def set_initial_control_sequence(self, initial_control_sequence):
        self.initial_control_sequence = initial_control_sequence.copy()

    def roll_out(self, state_cur, us, dt):
        dynamics = self.get_dynamics()
        trajectory = np.zeros((dynamics.get_state_dim()[0], us.shape[1]+1))
        trajectory[:, 0] = state_cur.flatten()
        for ii in range(us.shape[1]):
            state_cur = dynamics.propagate(state_cur, us[:, ii:ii+1], dt)
            trajectory[:, ii+1] = state_cur.flatten()
        return trajectory
