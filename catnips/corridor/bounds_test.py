#%% 
import numpy as np
from scipy.spatial import KDTree
import cvxpy as cvx
import time

class Corridor():
    def __init__(self, grid_pts_occupied, r=0.1) -> None:
        self.r = r
        self.grid_pts_kd = KDTree(grid_pts_occupied.reshape(-1, 3))

    def create_corridor(self, start, end):
        mid = 0.5* (start + end)

        neigh = self.grid_pts_kd.query_ball_point(mid, self.r)
            
        # Data processing into matrices for CVX
        data_out = self.grid_pts_kd.data[neigh]
        data_in = np.stack([start, end], axis=0)

        # IMPORTANT: SHIFT ALL POINTS TO ORIGIN
        data_out = data_out - mid[None, :]
        data_in = data_in - mid[None, :]

        data1 = data_out[:, :, None] * data_out[..., None, :]
        data1 = data1.transpose(0, 2, 1)
        data1 = data1.reshape(data_out.shape[0], 9)

        data2 = data_in[..., None] * data_in[..., None, :]
        data2 = data2.transpose(0, 2, 1)
        data2 = -data2.reshape(2, 9)

        data = np.concatenate([data1, data2], axis=0)
        h = np.ones((data.shape[0]))
        h[-2:] = -1
        
        #tnow = time.time()
        # X = self.generate_ellipse(data, h)

        X = self.generate_ellipse_simple(data_out, start, end)

        # assert(np.linalg.norm(X @ (start - mid)) == 1.)
        # assert(np.linalg.norm(X @ (end - mid)) == 1.)

        #print('Elapsed ellipse', time.time() - tnow)
        #tnow = time.time()
        A, B = self.generate_polytope(data_in, data_out, mid, X)

        #print('Elapsed polytope', time.time() - tnow)

        return A, B

    def generate_ellipse_simple(self, vertices, start, end):
        mid_pt = (start + end) / 2
        rad = np.linalg.norm(start - end) / 2

        # Find the closest point
        dd, ii = self.grid_pts_kd.query(mid_pt)
        closest_pt = self.grid_pts_kd.data[ii] - mid_pt
        closest_dist = dd

        if dd >= rad:
            X = (1 / rad) * np.eye(3)
            return X

        x_axis = (end - mid_pt) / np.linalg.norm(end - mid_pt)
        r_x = 1 / (rad)
        r_z = r_x

        dist_min = 0.
        while dist_min < 1. - 1e-6:

            # Define x-y-z coordinate frame
            inter_axis = (closest_pt) / np.linalg.norm(closest_pt)
            z_axis = np.cross(x_axis, inter_axis)
            y_axis = np.cross(z_axis, x_axis)

            R_body2world = np.stack([x_axis, y_axis, z_axis], axis=-1)
            R_world2body = R_body2world.T

            closest_pt_body = R_world2body @ closest_pt
            # Find minor axis
            r_y = np.sqrt((1 / closest_pt_body[1]**2) * (1 - (r_x * closest_pt_body[0])**2 ))
            r_z = r_y

            X = np.diag(np.array([r_x, r_y, r_z])) @ R_world2body
            dists_body = np.linalg.norm((X @ vertices.T).T, axis=-1)

            # Find pt that has smallest distance in body frame
            ind_min = np.argmin(dists_body)
            dist_min = dists_body[ind_min]
            closest_pt = vertices[ind_min]

        # By this point, the minimal pt should yield distance in body frame of 1 and also aligned in x-y. Therefore, we can
        # extend in the z-direction.
        second_pt_body = (R_world2body @ vertices.T).T

        r_z = np.sqrt((1. - (r_x * second_pt_body[:, 0])**2 - (r_y * second_pt_body[:, 1])**2) / second_pt_body[:, 2]**2)
        r_z = r_z[~np.isnan(r_z)]
        # r_z = np.sqrt((1. - (r_x * closest_pt[0])**2 - (r_y * closest_pt[1])**2) / closest_pt[2]**2)

        r_z = np.max(r_z)
        X = np.diag(np.array([r_x, r_y, r_z])) @ R_world2body

        keep_out = np.linalg.norm((X @ vertices.T), axis=0)
        assert (keep_out >= 1. - 1e-6).all()

        # Maximal bounding ellipsoid
        return X

    def generate_ellipse(self, data, h):
    
        # Construct the problem.
        x = cvx.Variable((3, 3), PSD=True)
        x_vec = cvx.reshape(x, 9)

        objective = cvx.Minimize(cvx.trace(x))
        # objective = cp.Minimize( cp.sum((dat @ x_vec)[:data1.shape[0]]) )

        # cvx.sum( cvx.multiply(data,  (data @ x.T) ),axis=1) >= h
        constraints = [
                        data @ x_vec >= h,
                        cvx.lambda_min(x) >= (1/self.r)**2
                        ]

        prob = cvx.Problem(objective, constraints)

        result = prob.solve()
        # print('Elapsed', time.time() - tnow)

        if x.value is None:
            raise AssertionError('Could not generate a valid ellipse during safety corridor creation.')

        X = np.array(x.value)

        eigs, V = np.linalg.eig(X)
    
        E = V @ np.sqrt(np.diag(eigs)) @ V.T

        return E

    def generate_polytope(self, data_in, data_out, mid, X):
        # Find Bounding polytope
        collision_pts = data_out

        # Add in box constraints to prevent polytope from stretching to infinity
        A_box = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
        B_box = (self.r / np.sqrt(3)) * np.ones(6)

        if len(collision_pts) == 0:
            A = A_box
            B = B_box
        else:
            keep_out = np.linalg.norm((X @ collision_pts.T), axis=0)

            # assert (keep_out > 1 - 1e-3).all()

            keep_out = np.linalg.norm((X @ collision_pts.T), axis=0)
            keep_in = np.linalg.norm((X @ data_in.T ), axis=0)

            # assert (keep_in < 1 + 1e-3).all()

            As = []
            Bs = []

            closest_pts = []

            while True:
                dists = np.linalg.norm((X @ collision_pts.T), axis=0)
                ind_min = np.argmin(dists)

                closest_pt = collision_pts[ind_min]

                X = X / dists[ind_min]

                a = 2* X.T @ X @ closest_pt
                b = a.T @ closest_pt

                As.append(a)
                Bs.append(b)
                closest_pts.append(closest_pt)

                collision_pts = np.delete(collision_pts, ind_min, axis=0)

                if len(collision_pts) > 0:
                    to_keep = (collision_pts @ a < b)
                    collision_pts = collision_pts[to_keep]

                    if len(collision_pts) == 0:
                        break
                else:
                    break

            #  Construct Polytope
            A = np.stack(As, axis=0)
            B = np.array(Bs)

            A = np.concatenate([A, A_box], axis=0)
            B = np.concatenate([B, B_box], axis=0)

        # Shift coordinate system back
        B = B + A @ mid

        return A, B

#%% 
t = np.linspace(0., 2*np.pi, 1000)
pts = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)])
corridor = Corridor(pts, r=5)

start = np.array([0.5, 0.5, 0.2])
end = -start

A, B = corridor.create_corridor(start, end)

#%%