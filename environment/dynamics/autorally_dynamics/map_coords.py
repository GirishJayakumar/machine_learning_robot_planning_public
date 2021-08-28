import numpy as np


class MapCA:

    def __init__(self, track_path):
        # file_name = 'maps/CCRF/CCRF_2021-01-10.npz'
        # file_name = 'maps/CCRF/ccrf_track_optimal.npz'
        file_name = track_path
        track_dict = np.load(file_name)
        try:
            p_x = track_dict['X_cen_smooth']
            p_y = track_dict['Y_cen_smooth']
        except KeyError:
            p_x = track_dict['pts'][:, 0]
            p_y = track_dict['pts'][:, 1]
            self.rho = track_dict['curvature']
        self.p = np.array([p_x, p_y])
        dif_vecs = self.p[:, 1:] - self.p[:, :-1]
        self.dif_vecs = dif_vecs
        self.slopes = dif_vecs[1, :] / dif_vecs[0, :]
        self.midpoints = self.p[:, :-1] + dif_vecs/2
        self.s = np.cumsum(np.linalg.norm(dif_vecs, axis=0))

    def localize(self, M, psi):
        dists = np.linalg.norm(np.subtract(M.reshape((-1,1)), self.midpoints), axis=0)
        mini = np.argmin(dists)
        p0 = self.p[:, mini]
        p1 = self.p[:, mini+1]
        # plt.plot(M[0], M[1], 'x')
        # plt.plot(p0[0], p0[1], 'o')
        # plt.plot(p1[0], p1[1], 'o')
        ortho = -1/self.slopes[mini]
        a = M[1] - ortho * M[0]
        a_0 = p0[1] - ortho*p0[0]
        a_1 = p1[1] - ortho*p1[0]
        printi=0
        if a_0 < a < a_1 or a_1 < a < a_0:
            norm_dist = np.sign(np.cross(p1 - p0, M - p0)) * np.linalg.norm(np.cross(p1 - p0, M - p0)) / np.linalg.norm(p1 - p0)
            s_dist = np.linalg.norm(np.dot(M-p0, p1-p0))
        else:
            printi=1
            norm_dist = np.sign(np.cross(p1 - p0, M - p0)) * np.linalg.norm(M - p0)
            s_dist = 0
        s_dist += self.s[mini]
        head_dist = psi - np.arctan2(self.dif_vecs[1, mini], self.dif_vecs[0, mini])
        if head_dist > np.pi:
            # print(psi, np.arctan2(self.dif_vecs[1, mini], self.dif_vecs[0, mini]))
            head_dist -= 2*np.pi
            # print(norm_dist, s_dist, head_dist * 180 / np.pi)
        elif head_dist < -np.pi:
            head_dist += 2*np.pi
            # print(norm_dist, s_dist, head_dist * 180 / np.pi)
        # if printi:
        #     print(norm_dist, s_dist, head_dist*180/np.pi)
        #     printi=0
        # plt.show()
        return head_dist, norm_dist, s_dist

    def get_cur_reg_from_s(self, s):
        nearest = np.argmin(np.abs(s.reshape((-1, 1)) - self.s.reshape((1, -1))), axis=1)
        x0 = self.p[0, nearest]
        y0 = self.p[1, nearest]
        theta0 = np.arctan2(self.dif_vecs[1, nearest], self.dif_vecs[0, nearest])
        s = self.s[nearest]
        curvature = self.rho[nearest]
        return x0, y0, theta0, s, curvature
