from mosek.fusion import *
import numpy as np
import scipy.linalg
import scipy.stats
import time


class CSSolver:
    def __init__(self, n, m, l, N, v_range, slew_rate, obstacles, mean_only=False, k_form=0, prob_lvl=0.95, chance_const_N=-1):
        try:
            M = Model()
            self.n = n
            self.m = m
            self.l = l
            self.N = N
            if chance_const_N < 0:
                chance_const_N = N

            v_min = v_range[:, 0]
            v_max = v_range[:, 1]
            vmax = np.tile(v_max.reshape((-1, 1)), (N, 1)).flatten().astype('double')
            vmin = np.tile(v_min.reshape((-1, 1)), (N, 1)).flatten().astype('double')

            V = M.variable("V", m*N, Domain.inRange(vmin, vmax))
            k = None
            if not mean_only:
                if k_form == 2:
                    k = M.variable([m, n])
                    k_rep = Expr.repeat(k, N, 0)
                    k_rep2 = Expr.repeat(k_rep, N, 1)
                    identity_k = np.kron(np.eye(N), np.ones((m, n)))
                    K_dot = Matrix.sparse(identity_k)
                    K = Expr.mulElm(K_dot, k_rep2)
                elif k_form == 1:
                    pattern = []
                    for ii in range(N):
                        for jj in range(m):
                            for ll in range(n):
                                row = ii*m + jj
                                col = ii*n + ll
                                pattern.append([row, col])
                    K = M.variable([m*N, n*N], Domain.sparse(Domain.unbounded(), pattern))
                else:
                    pattern = []
                    for ii in range(m * N):
                        for jj in range(n * N):
                            row = ii
                            col = jj
                            pattern.append([row, col])
                            if jj >= n * (ii + 1):
                                break
                    K = M.variable([m * N, n * N], Domain.sparse(Domain.unbounded(), pattern))
            else:
                if k_form == 1:
                    pattern = []
                    for ii in range(N):
                        for jj in range(m):
                            for ll in range(n):
                                row = ii * m + jj
                                col = ii * n + ll
                                pattern.append([row, col])
                    K = M.parameter([m*N, n*N], pattern)
                else:
                    pattern = []
                    for ii in range(m * N):
                        for jj in range(n * N):
                            row = ii
                            col = jj
                            pattern.append([row, col])
                            if jj >= n * (ii + 1):
                                break
                    K = M.parameter([m * N, n * N], pattern)

            # linear variables with quadratic cone constraints
            w = M.variable("w", 1, Domain.unbounded())
            x = M.variable("x", 1, Domain.unbounded())
            # y = M.variable("y", 1, Domain.unbounded())
            # z = M.variable("z", 1, Domain.unbounded())
            y1 = M.variable("y1", 1, Domain.unbounded())
            z1 = M.variable("z1", 1, Domain.unbounded())
            y2 = M.variable("y2", 1, Domain.unbounded())
            z2 = M.variable("z2", 1, Domain.unbounded())

            mu_0_T_A_T_Q_bar_B = M.parameter([1, m*N])
            vec_T_sigma_y_Q_bar_B = M.parameter([1, n*N*m*N])
            pattern = []
            for ii in range(n*N):
                for jj in range(m*N):
                    if jj <= ii:
                        pattern.append([ii, jj])
            pattern = []
            for ii in range(N):
                for jj in range(n):
                    for kk in range(m * (ii + 1)):
                        pattern.append([jj + n * ii, kk])
            Q_bar_half_B = M.parameter([n*N, m*N], pattern)
            pattern = []
            for ii in range(m*N):
                pattern.append([ii, ii])
            R_bar_half = M.parameter([m*N, m*N], pattern)
            pattern = []
            for ii in range(n * N):
                pattern.append([ii, ii])
            Q_bar_half = M.parameter([n*N, n*N], pattern)
            # sigma_y_half = M.parameter([n*N, n*N])
            A_sigma_0_half = M.parameter([n*N, n])
            pattern = []
            for ii in range(n * N):
                for jj in range(l * N):
                    row = ii
                    col = jj
                    pattern.append([row, col])
                    if jj >= l * (ii + 1):
                        break
            D = M.parameter([n*N, l*N])
            A_mu_0 = M.parameter([n*N, 1])
            pattern = []
            for ii in range(n * N):
                for jj in range(m * N):
                    row = ii
                    col = jj
                    pattern.append([row, col])
                    if jj >= m * (ii + 1):
                        break
            B = M.parameter([n*N, m*N])
            d = M.parameter([n*N, 1])
            sigma_N_inv = M.parameter([n, n])
            neg_x_0_T_Q_B = M.parameter([1, m*N])
            d_T_Q_B = M.parameter([1, m*N])
            if mean_only:
                covariance_chance_constraints = M.parameter([N, 2])

            I = Matrix.eye(n*N)
            inv_prob = scipy.stats.norm.ppf(prob_lvl)

            # convert to linear objective with quadratic cone constraints
            u = Expr.mul(mu_0_T_A_T_Q_bar_B, V)
            # coordinate shift, check to make sure T = *2
            q = Expr.mul(neg_x_0_T_Q_B, V)
            r = Expr.mul(d_T_Q_B, V)
            # v = Expr.mul(vec_T_sigma_y_Q_bar_B, Expr.flatten(K))
            zz = M.variable("zz", 1, Domain.unbounded())
            M.constraint(Expr.vstack(0.5, zz, Expr.mul(1, Expr.sub(V.slice(2, N*m), V.slice(0, N*m-2)))), Domain.inRotatedQCone())
            if not mean_only:
                M.objective(ObjectiveSense.Minimize, Expr.add([q, r, u, w, x, y1, y2, z1, z2, zz]))
                M.constraint(Expr.vstack(0.5, w, Expr.mul(Q_bar_half_B, V)), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, x, Expr.mul(R_bar_half, V)), Domain.inRotatedQCone())

                M.constraint(Expr.vstack(0.5, y1, Expr.flatten(Expr.mul(Q_bar_half, Expr.mul(Expr.add(I, Expr.mul(B, K)), A_sigma_0_half)))), Domain.inRotatedQCone())
                # check BKD multiplicaion, or maybe with I? Something seems wrong here b/c sparsity pattern is invalid
                M.constraint(Expr.vstack(0.5, y2, Expr.flatten(Expr.mul(Q_bar_half, Expr.mul(Expr.add(I, Expr.mul(B, K)), D)))), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, z1, Expr.flatten(Expr.mul(R_bar_half, Expr.mul(K, A_sigma_0_half)))), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, z2, Expr.flatten(Expr.mul(R_bar_half, Expr.mul(K, D)))), Domain.inRotatedQCone())
            else:
                M.objective(ObjectiveSense.Minimize, Expr.add([q, r, u, w, x, zz]))
                M.constraint(Expr.vstack(0.5, w, Expr.mul(Q_bar_half_B, V)), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, x, Expr.mul(R_bar_half, V)), Domain.inRotatedQCone())

            # slew-rate constraints
            # M.constraint(Expr.sub(V.slice(2, N*m), V.slice(0, N*m-2)), Domain.inRange(np.tile(slew_rate[:, 0], N-1), np.tile(slew_rate[:,  1], N-1)))
            self.u_steering = M.parameter()
            self.u_throttle = M.parameter()
            M.constraint(Expr.sub(self.u_throttle, V.index(1)), Domain.inRange(slew_rate[1, 0], slew_rate[1, 1]))
            M.constraint(Expr.sub(self.u_steering, V.index(0)), Domain.inRange(slew_rate[0, 0], slew_rate[0, 1]))

            # terminal mean constraint
            # mu_N = np.zeros((n, 1))
            # mu_N = np.array([7.5, 2., 2.5, 100., 100., 0.5, 1.0, 1000.]).reshape((8, 1))
            mu_N = M.parameter([n, 1])
            e_n = np.zeros((n, n))
            e_n[4, 4] = 1
            e_n[6, 6] = 1
            e_n = np.eye(n)
            E_N = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), e_n)))
            E_N_T = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), np.eye(n))).T)
            M.constraint(Expr.sub(Expr.mul(E_N, Expr.add(Expr.add(A_mu_0, Expr.mul(B, V)), d)), mu_N), Domain.lessThan(0))
            e_n = -1 * e_n
            E_N = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), e_n)))
            E_N_T = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), np.eye(n))).T)
            M.constraint(Expr.sub(Expr.mul(E_N, Expr.add(Expr.add(A_mu_0, Expr.mul(B, V)), d)), mu_N), Domain.lessThan(0))

            if obstacles[0]:
                self.obs = M.parameter(obstacles[1])
                num_obs = int(obstacles[1] / N / 4)
                z = M.variable("z", [4, N, num_obs], Domain.integral(Domain.inRange(0, 1)))
                M.constraint(Expr.sum(z.asExpr(), 0), Domain.equalsTo(3))
                # alpha = Matrix.sparse(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]))
                X = Expr.add(Expr.add(A_mu_0, Expr.mul(B, V)), d)
                x_matrix = np.zeros((N, n * N))
                for ii in range(N):
                    x_matrix[ii, ii * n + 6] = 1
                y_matrix = np.zeros((N, n * N))
                for ii in range(N):
                    y_matrix[ii, ii * n + 7] = 1
                xs = Expr.mul(Matrix.sparse(x_matrix), X)
                ys = Expr.mul(Matrix.sparse(y_matrix), X)
                gamma = 100
                # print(Expr.flatten(z.slice([0, 0], [1, N])).getShape(), xs.getShape())

                for ii in range(num_obs):
                    x_min = self.obs.slice([ii*4*N], [ii*4*N+N])
                    x_max = self.obs.slice([ii*4*N+N], [ii*4*N+2*N])
                    y_min = self.obs.slice([ii*4*N+2*N], [ii*4*N+3*N])
                    y_max = self.obs.slice([ii*4*N+3*N], [ii*4*N+4*N])

                    if mean_only:
                        x_cov = covariance_chance_constraints.slice([0, 0], [N, 1])
                        y_cov = covariance_chance_constraints.slice([0, 1], [N, 2])
                        M.constraint(
                            Expr.sub(Expr.flatten(Expr.add(Expr.mul(Expr.flatten(z.slice([0, 0, ii], [1, N, ii+1])), -gamma), Expr.add(Expr.flatten(xs), Expr.mul(x_cov, Expr.constTerm(inv_prob))))),
                                     Expr.flatten(x_min)),
                            Domain.lessThan(0))
                        M.constraint(
                            Expr.sub(Expr.flatten(Expr.add(Expr.mul(Expr.flatten(z.slice([1, 0, ii], [2, N, ii+1])), gamma), Expr.sub(Expr.flatten(xs), Expr.mul(x_cov, Expr.constTerm(inv_prob))))),
                                     Expr.flatten(x_max)),
                            Domain.greaterThan(0))
                        M.constraint(
                            Expr.sub(Expr.flatten(Expr.add(Expr.mul(Expr.flatten(z.slice([2, 0, ii], [3, N, ii+1])), -gamma), Expr.add(Expr.flatten(ys), Expr.mul(y_cov, Expr.constTerm(inv_prob))))),
                                     Expr.flatten(y_min)),
                            Domain.lessThan(0))
                        M.constraint(
                            Expr.sub(Expr.flatten(Expr.add(Expr.mul(Expr.flatten(z.slice([3, 0, ii], [4, N, ii+1])), gamma), Expr.sub(Expr.flatten(ys), Expr.mul(y_cov, Expr.constTerm(inv_prob))))),
                                     Expr.flatten(y_max)),
                            Domain.greaterThan(0))

                    else:
                        M.constraint(
                            Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([0, 0, 0], [1, N, 1])), -gamma), Expr.flatten(xs)),
                                     x_min),
                            Domain.lessThan(0))
                        M.constraint(
                            Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([1, 0], [2, N])), gamma), Expr.flatten(xs)), x_max),
                            Domain.greaterThan(0))
                        M.constraint(
                            Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([2, 0], [3, N])), -gamma), Expr.flatten(ys)),
                                     y_min),
                            Domain.lessThan(0))
                        M.constraint(
                            Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([3, 0], [4, N])), gamma), Expr.flatten(ys)), y_max),
                            Domain.greaterThan(0))
            else:
                pass
                # terminal covariance constraint
                # sigma_N_inv = Expr.constTerm(np.ones((n, n)) * 1)
                # EN = scipy.sparse.csr_matrix(np.hstack((np.zeros((n, (N - 1) * n)), np.eye(n))))
                # BNbar = EN.dot(B.copy())
                # Z = Expr.add(Matrix.sparse(EN.dot(D)), Expr.mul(Expr.mul(Matrix.sparse(BNbar), K), Matrix.dense(D)))
                # X_sym = Expr.vstack(Expr.hstack(sigma_N_inv, Z),
                #                     Expr.hstack(Expr.transpose(Z), Expr.constTerm(np.eye(l * N))))
                # M.constraint(X_sym, Domain.inPSDCone())

            # chance constraint
            beta = M.parameter()
            for ii in range(chance_const_N):
                alpha = np.zeros((n, 1))
                alpha[6, 0] = 1
                alpha_T = Matrix.sparse(alpha.T)
                e_k = np.eye(n)
                E_k = Matrix.sparse(np.hstack((np.zeros((n, (ii) * n)), e_k, np.zeros((n, (N - ii - 1) * n)))))
                mean_part = Expr.mul(alpha_T, Expr.mul(E_k, Expr.add(Expr.add(A_mu_0, Expr.mul(B, V)), d)))
                if not mean_only:
                    sigma_0_part = M.variable()
                    M.constraint(Expr.vstack(sigma_0_part, Expr.flatten(
                        Expr.mul(alpha_T, Expr.mul(E_k, Expr.mul(Expr.add(I, Expr.mul(B, K)), A_sigma_0_half))))),
                                 Domain.inQCone())
                    D_part = M.variable()
                    M.constraint(Expr.vstack(D_part, Expr.flatten(
                        Expr.mul(alpha_T, Expr.mul(E_k, Expr.mul(Expr.add(I, Expr.mul(B, K)), D)))).slice(0, (
                                ii + 1) * l)), Domain.inQCone())
                    cov_part = M.variable()
                    M.constraint(Expr.vstack(cov_part, sigma_0_part, D_part), Domain.inQCone())
                    M.constraint(Expr.sub(Expr.add(mean_part, Expr.mul(cov_part, inv_prob)), beta),
                                 Domain.lessThan(0))
                    M.constraint(Expr.add(Expr.sub(mean_part, Expr.mul(cov_part, inv_prob)), beta),
                                 Domain.greaterThan(0))
                else:
                    cov_part = covariance_chance_constraints.index([ii, 0])
                    M.constraint(Expr.sub(Expr.add(mean_part, Expr.mul(cov_part, Expr.constTerm(inv_prob))), beta), Domain.lessThan(0))
                    M.constraint(Expr.add(Expr.sub(mean_part, Expr.mul(cov_part, Expr.constTerm(inv_prob))), beta),
                                 Domain.greaterThan(0))

            # M.setLogHandler(sys.stdout)

            self.M = M
            self.V = V
            self.k = k
            self.K = K
            self.mu_0_T_A_T_Q_bar_B = mu_0_T_A_T_Q_bar_B
            self.vec_T_sigma_y_Q_bar_B = vec_T_sigma_y_Q_bar_B
            self.Q_bar_half_B = Q_bar_half_B
            self.R_bar_half = R_bar_half
            self.Q_bar_half = Q_bar_half
            # self.sigma_y_half = sigma_y_half
            self.A_sigma_0_half = A_sigma_0_half
            self.D = D
            self.A_mu_0 = A_mu_0
            self.B = B
            self.d = d
            self.sigma_N_inv = sigma_N_inv
            self.neg_x_0_T_Q_B = neg_x_0_T_Q_B
            self.d_T_Q_B = d_T_Q_B
            self.mu_N = mu_N
            self.beta = beta
            if mean_only:
                self.covariance_chance_constraints = covariance_chance_constraints
            self.mean_only = mean_only
            self.k_form = k_form
            self.obstacles = obstacles[0]

        finally:
            pass
            # M.dispose()

    def populate_params(self, A, B, d, D, mu_0, sigma_0, sigma_N_inv, Q_bar, R_bar, u_0, x_target, mu_N, track_width, K=None, obs=None):
        n = self.n
        m = self.m
        l = self.l
        N = self.N
        # A = np.tile(np.eye(n), (N, 1))
        # B = np.kron(np.eye(N), np.random.randn(n, m))
        # D = np.kron(np.eye(N), np.random.randn(n, l))
        # mu_0 = np.zeros((n, 1))
        # sigma_0 = np.random.randn(n, n)
        # sigma_0 = np.dot(sigma_0, sigma_0.T)
        # sigma_N_inv = np.linalg.inv(1000 * sigma_0)
        sigma_y = np.dot(A, np.dot(sigma_0, A.T)) + np.dot(D, D.T)
        # sigma_y = np.linalg.cholesky(sigma_y)
        # Q_bar = np.eye(n*N)
        # R_bar = np.eye(m*N)
        x_0 = x_target.copy()

        self.mu_0_T_A_T_Q_bar_B.setValue(2*np.dot(np.dot(np.dot(mu_0.T, A.T), Q_bar), B))
        temp = 2*np.dot(sigma_y, np.dot(Q_bar, B)).reshape((-1, 1)).T
        self.vec_T_sigma_y_Q_bar_B.setValue(temp)
        # try:
        #     self.Q_bar_half_B.setValue(np.dot(scipy.linalg.cholesky(Q_bar), B))
        # except np.linalg.LinAlgError:
        self.Q_bar_half_B.setValue(np.dot(np.sqrt(Q_bar), B))
        self.Q_bar_half.setValue(np.sqrt(Q_bar))
        # try:
        #     self.R_bar_half.setValue(scipy.linalg.cholesky(R_bar))
        # except np.linalg.LinAlgError:
        self.R_bar_half.setValue(np.sqrt(R_bar))
        # try:
        #     self.Q_bar_half.setValue(scipy.linalg.cholesky(Q_bar))
        # except np.linalg.LinAlgError:
        #     self.Q_bar_half.setValue(np.sqrt(Q_bar))
        # try:
        #     self.sigma_y_half.setValue(sigma_y)
        # except np.linalg.LinAlgError:
        #     print("cholesky failed")
        #     self.sigma_y_half.setValue(np.sqrt(sigma_y))
        # try:
        #     self.A_sigma_0_half.setValue(np.dot(A, np.linalg.cholesky(sigma_0)))
        # except np.linalg.LinAlgError:
        self.A_sigma_0_half.setValue(np.dot(A, np.sqrt(sigma_0)))
        self.D.setValue(D)
        self.A_mu_0.setValue((np.dot(A, mu_0)))
        self.B.setValue(B)
        self.d.setValue(d)
        self.sigma_N_inv.setValue(sigma_N_inv)
        self.neg_x_0_T_Q_B.setValue(2*np.dot(np.dot(-x_0.T, Q_bar), B))
        self.d_T_Q_B.setValue(2*np.dot(np.dot(d.T, Q_bar), B))
        self.u_throttle.setValue(u_0[1])
        self.u_steering.setValue(u_0[0])
        self.mu_N.setValue(mu_N)
        self.beta.setValue(track_width/2)
        # self.M.writeTask('dump.opf')
        if self.mean_only:
            self.K.setValue(K)
            I_BK = np.eye(n * N) + np.dot(B, K)
            D_part = np.dot(I_BK, D)
            sigma0_part = np.dot(I_BK, np.dot(A, np.sqrt(sigma_0)))
            D_part_Ek = D_part.reshape((N, n, N*l))
            sigma0_part_Ek = sigma0_part.reshape((N, n, n))
            D_part_Ek_alpha = D_part_Ek[:, 6:, :]
            sigma0_part_Ek_alpha = sigma0_part_Ek[:, 6:, :]
            D_part_norm = np.linalg.norm(D_part_Ek_alpha, axis=2)
            sigma0_part_norm = np.linalg.norm(sigma0_part_Ek_alpha, axis=2)
            self.covariance_chance_constraints.setValue(D_part_norm + sigma0_part_norm)
        if self.obstacles:
            self.obs.setValue(obs.flatten())

    def solve(self):
        # print(self.u_0.getValue())
        # self.M.solve()
        t0 = time.time()
        self.M.solve()
        print('convex opt time: ', (time.time() - t0))
        try:
            if self.mean_only:
                K_level = np.zeros((self.m*self.N, self.n*self.N))
            else:
                if self.k_form == 2:
                    K_level = np.kron(np.eye(self.N), self.k.level().reshape((self.m, self.n)))
                else:
                    K_level = self.K.level()
            levels = (self.V.level(), K_level)
            # print(levels)
            return levels
        except SolutionError:
            raise RuntimeError

    def time(self):
        t0 = time.time()
        for ii in range(20):
            self.populate_params()
            self.solve()
        print((time.time() - t0) / 20)


if __name__ == '__main__':
    import cs_model
    import matplotlib.pyplot as plt
    n = 8
    m = 2
    l = 8
    N = 50
    v_range = np.array([[-1, 1], [-1, 1]])
    slew_rate = np.array([[-0.1, 0.1], [-0.1, 0.1]])
    obs = np.array([[[1.5, 16], [2, 17]], [[0.5, 29], [2, 31]]])
    solver = CSSolver(n, m, l, N, v_range, slew_rate, (True, obs), mean_only=True, k_form=1)
    try:
        state = np.array([5., 0., 0., 50., 50., 0., 0., 0.]).reshape((-1, 1))
        control = np.array([0., 0.3]).reshape((-1, 1))
        model = cs_model.Model(N)
        A, B, d = model.linearize_dynamics(np.tile(state, (1, N)), np.tile(control, (1, N)))
        D = 0.001 * np.ones((n, l, N))
        AA, BB, dd, DD = model.form_long_matrices_LTV(A.reshape((n, n, N), order='F'), B.reshape((n, m, N), order='F'), d.reshape((n, 1, N), order='F'), D)
        Q = np.kron(np.eye(N), np.diag([3., 0.1, 0.1, 0., 0., 10., 10., 0.]))
        R = np.kron(np.eye(N), 0.1*np.eye(m))
        x_target = np.tile(np.array([6., 0., 0., 60., 60., 0., 2.0, 0.]).reshape((-1, 1)), (N, 1))
        mu_N = np.array([5., 1., 1., 50., 50., 1., 1., 100.]).reshape((-1, 1))

        solver.populate_params(AA, BB, dd, DD, state, np.zeros((n, n)), np.ones((n, n)), Q, R, control, x_target, mu_N, 4, K=np.zeros((m*N, n*N)))
        # solver.M.setSolverParam("numThreads", 8)
        # solver.M.setSolverParam("intpntCoTolPfeas", 1e-3)
        # solver.M.setSolverParam("intpntCoTolDfeas", 1e-3)
        # solver.M.setSolverParam("intpntCoTolRelGap", 1e-3)
        # solver.M.setSolverParam("intpntCoTolInfeas", 1e-3)
        solver.M.setSolverParam("mioTolAbsGap", 10000)
        V, K = solver.solve()
        print(V.reshape((m, N)), K.reshape((m*N, n*N)))
        X_bar = np.dot(AA, state) + np.dot(BB, V.reshape((-1, 1))) + dd
        X_bar = X_bar.reshape((n, N), order='F')
        print(X_bar)
        plt.plot(X_bar[6, :], X_bar[7, :])
        plt.xlim(-3, 3)
        plt.show()
        # solver.time()
    finally:
        solver.M.dispose()
