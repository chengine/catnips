import numpy as np
from scipy.spatial import KDTree
import cvxpy as cvx

class Corridor():
    def __init__(self) -> None:
        pass
    
    def create_corridor(self, path, grid_pts_occupied, r=0.1):

        mid_points_path = 0.5* (path[1:] + path[:-1])

        grid_pts_kd = KDTree(grid_pts_occupied.reshape(-1, 3))

        neighbors = grid_pts_kd.query_ball_point(mid_points_path, self.r)

        A_list = []
        B_list = []

        for it, (neigh, start, end, mid) in enumerate(zip(neighbors, path[:-1], path[1:], mid_points_path)):
            
            # Data processing into matrices for CVX
            data_out = grid_pts_kd.data[neigh]
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
            
            X, E = self.generate_ellipse(data, h)
            A, B = self.generate_polytope(data_in, data_out, mid, X, E)

            A_list.append(A)
            B_list.append(B)

        return A_list, B_list

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

        return X, E

    def generate_polytope(self, data_in, data_out, mid, X, E):
        # Find Bounding polytope
        collision_pts = data_out

        # Add in box constraints to prevent polytope from stretching to infinity
        A_box = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
        B_box = (self.r / np.sqrt(3)) * np.ones(6)

        if len(collision_pts) == 0:
            A = A_box
            B = B_box
        else:
            keep_out = np.linalg.norm((E @ collision_pts.T), axis=0)

            # assert (keep_out > 1 - 1e-3).all()

            # E = E / keep_out.min()
            keep_out = np.linalg.norm((E @ collision_pts.T), axis=0)
            keep_in = np.linalg.norm((E @ data_in.T ), axis=0)

            # assert (keep_in < 1 + 1e-3).all()

            As = []
            Bs = []

            closest_pts = []

            while True:
                dists = np.linalg.norm((E @ collision_pts.T), axis=0)
                ind_min = np.argmin(dists)

                closest_pt = collision_pts[ind_min]

                X = X / dists[ind_min]**2
                E = E / dists[ind_min]

                a = 2* X @ closest_pt
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