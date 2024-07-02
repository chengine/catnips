import os
import numpy as np
from scipy.spatial import KDTree
# import cvxpy as cvx
import time
from .bounds_utils import *

class BoxCorridor():
    def __init__(self, grid_binary_occupied, grid_pts_occupied, r=0.1) -> None:
        self.r = r
        self.occupied = ~grid_binary_occupied
        self.lx, self.ly, self.lz = grid_pts_occupied[1, 1, 1] - grid_pts_occupied[0, 0, 0]

        # Solve a convex program to find a less conservative bounding polygon 
        # Here, it's a box for simplicity.

        # Find only the surface voxels. We do this by taking the binary image 
        # of the grid, eroding by 1, and finding the voxels that changed.
        # This is an AND between the occupied voxels and the negation of the 
        # resulting erosion.
        eroded_occupy = scipy.ndimage.binary_erosion(self.occupied)
        surface_voxels = np.logical_and(self.occupied, ~eroded_occupy)
        self.centers = grid_pts_occupied[surface_voxels]

        self.grid_pts_kd = scipy.spatial.KDTree(self.centers)

    def create_corridor(self, traj, refine=True):

        bounds = []
        delta = torch.tensor([self.lx ,self.ly, self.lz, self.lx, self.ly, self.lz]).cuda()

        mid_points_path = 0.5* (traj[1:] + traj[:-1])

        for start, end, mid in zip(traj[:-1], traj[1:], mid_points_path):
            dist = np.linalg.norm(start - end)
            if dist > self.r:
                r = dist
            else:
                r = self.r

            offset = np.array([r/(2*.866), r/(2*.866), r/(2*.866)])

            neighbors = self.grid_pts_kd.query_ball_point(mid, r)
            near_pts = torch.from_numpy(self.grid_pts_kd.data[neighbors]).cuda()
        
            minimal = torch.tensor(min_bounding_box(start, end, self.lx, self.ly, self.lz)).cuda()
            maximal = torch.tensor(np.array([mid + offset, mid - offset])).T.cuda()
            bound = grow_box(near_pts, minimal, maximal, delta)
            bounds.append(bound)

        if refine:
            bounds = refine_bounds(bounds)

        As = []
        Bs = []
        for bound in bounds:
            bound_max = bound[:, 0]
            bound_min = bound[:, 1]

            A = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
            B = np.concatenate([bound_max, -bound_min]).squeeze()

            As.append(A)
            Bs.append(B)

        return As, Bs, bounds
    
    def bounds2mesh(self, bounds, filename, transform=None, scale=None, i=0):
        if not os.path.exists(filename):
            # Create a new directory because it does not exist
            os.makedirs(filename)
    
        # bounds is list of B vectors in Ax <= B
        vox_mesh=o3d.geometry.TriangleMesh()
        for bound in bounds:
            top_bound = bound[:3]
            low_bound = -bound[3:]

            diff = top_bound - low_bound
            center = (top_bound + low_bound) / 2
 
            lx, ly, lz = np.abs(diff[0]), \
                        np.abs(diff[1]), \
                        np.abs(diff[2])

            cube=o3d.geometry.TriangleMesh.create_box(width=lx, height=ly,
            depth=lz)
            cube.translate(center, relative=False)
            vox_mesh+=cube

        if scale is not None:
            vox_mesh.scale(1/scale, center=np.array([0, 0, 0]))

        if transform is not None:
            vox_mesh.transform(np.linalg.inv(transform))

        vox_mesh.merge_close_vertices(1e-6)
        o3d.io.write_triangle_mesh(filename + f'{i}.ply', vox_mesh)
    
# class PolytopeCorridor():
#     def __init__(self, grid_pts_occupied, r=0.1) -> None:
#         self.r = r
#         self.grid_pts_kd = KDTree(grid_pts_occupied.reshape(-1, 3))

#     def create_corridor(self, path):
#         mid_points_path = 0.5* (path[1:] + path[:-1])

#         neighbors = self.grid_pts_kd.query_ball_point(mid_points_path, self.r)

#         A_list = []
#         B_list = []

#         for it, (neigh, start, end, mid) in enumerate(zip(neighbors, path[:-1], path[1:], mid_points_path)):
            
#             # Data processing into matrices for CVX
#             data_out = self.grid_pts_kd.data[neigh]
#             data_in = np.stack([start, end], axis=0)

#             # IMPORTANT: SHIFT ALL POINTS TO ORIGIN
#             data_out = data_out - mid[None, :]
#             data_in = data_in - mid[None, :]

#             data1 = data_out[:, :, None] * data_out[..., None, :]
#             data1 = data1.transpose(0, 2, 1)
#             data1 = data1.reshape(data_out.shape[0], 9)

#             data2 = data_in[..., None] * data_in[..., None, :]
#             data2 = data2.transpose(0, 2, 1)
#             data2 = -data2.reshape(2, 9)

#             data = np.concatenate([data1, data2], axis=0)
#             h = np.ones((data.shape[0]))
#             h[-2:] = -1
            
#             #tnow = time.time()
#             X = self.generate_ellipse(data, h)

#             # X = self.generate_ellipse_simple(data_out, start, end)

#             # assert(np.linalg.norm(X @ (start - mid)) == 1.)
#             # assert(np.linalg.norm(X @ (end - mid)) == 1.)

#             #print('Elapsed ellipse', time.time() - tnow)
#             #tnow = time.time()
#             A, B = self.generate_polytope(data_in, data_out, mid, X)

#             print(A @ start - B )
#             print(A @ end - B )
#             print(A)
#             print(B)
#             assert(np.all(A @ start - B <= 0.))
#             assert(np.all(A @ end - B <= 0.))

#             #print('Elapsed polytope', time.time() - tnow)

#             A_list.append(A)
#             B_list.append(B)

#         return A_list, B_list

#     def generate_ellipse_simple(self, vertices, start, end):
#         mid_pt = (start + end) / 2
#         rad = np.linalg.norm(start - end) / 2

#         # Find the closest point
#         dd, ii = self.grid_pts_kd.query(mid_pt)
#         closest_pt = self.grid_pts_kd.data[ii] - mid_pt
#         closest_dist = dd

#         if dd >= rad:
#             X = (1 / rad) * np.eye(3)
#             return X

#         x_axis = (end - mid_pt) / np.linalg.norm(end - mid_pt)
#         r_x = 1 / (rad)
#         r_z = r_x

#         dist_min = 0.
#         while dist_min < 1. - 1e-6:

#             # Define x-y-z coordinate frame
#             inter_axis = (closest_pt) / np.linalg.norm(closest_pt)
#             z_axis = np.cross(x_axis, inter_axis)
#             y_axis = np.cross(z_axis, x_axis)

#             R_body2world = np.stack([x_axis, y_axis, z_axis], axis=-1)
#             R_world2body = R_body2world.T

#             closest_pt_body = R_world2body @ closest_pt
#             # Find minor axis
#             r_y = np.sqrt((1 / closest_pt_body[1]**2) * (1 - (r_x * closest_pt_body[0])**2 ))
#             r_z = r_y

#             X = np.diag(np.array([r_x, r_y, r_z])) @ R_world2body
#             dists_body = np.linalg.norm((X @ vertices.T).T, axis=-1)

#             # Find pt that has smallest distance in body frame
#             ind_min = np.argmin(dists_body)
#             dist_min = dists_body[ind_min]
#             closest_pt = vertices[ind_min]

#         # By this point, the minimal pt should yield distance in body frame of 1 and also aligned in x-y. Therefore, we can
#         # extend in the z-direction.
#         second_pt_body = (R_world2body @ vertices.T).T

#         r_z = np.sqrt((1. - (r_x * second_pt_body[:, 0])**2 - (r_y * second_pt_body[:, 1])**2) / second_pt_body[:, 2]**2)
#         r_z = r_z[~np.isnan(r_z)]
#         # r_z = np.sqrt((1. - (r_x * closest_pt[0])**2 - (r_y * closest_pt[1])**2) / closest_pt[2]**2)

#         r_z = np.max(r_z)
#         X = np.diag(np.array([r_x, r_y, r_z])) @ R_world2body

#         keep_out = np.linalg.norm((X @ vertices.T), axis=0)
#         assert (keep_out >= 1. - 1e-6).all()

#         # Maximal bounding ellipsoid
#         return X

#     def generate_ellipse(self, data, h):
    
#         # Construct the problem.
#         x = cvx.Variable((3, 3), PSD=True)
#         x_vec = cvx.reshape(x, 9)

#         objective = cvx.Minimize(cvx.trace(x))
#         # objective = cp.Minimize( cp.sum((dat @ x_vec)[:data1.shape[0]]) )

#         # cvx.sum( cvx.multiply(data,  (data @ x.T) ),axis=1) >= h
#         constraints = [
#                         data @ x_vec >= h,
#                         cvx.lambda_min(x) >= (1/self.r)**2
#                         ]

#         prob = cvx.Problem(objective, constraints)

#         result = prob.solve()
#         # print('Elapsed', time.time() - tnow)

#         if x.value is None:
#             raise AssertionError('Could not generate a valid ellipse during safety corridor creation.')

#         X = np.array(x.value)

#         eigs, V = np.linalg.eig(X)
    
#         E = V @ np.sqrt(np.diag(eigs)) @ V.T

#         return E

#     def generate_polytope(self, data_in, data_out, mid, X):
#         # Find Bounding polytope
#         collision_pts = data_out

#         # Add in box constraints to prevent polytope from stretching to infinity
#         A_box = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
#         B_box = (self.r / np.sqrt(3)) * np.ones(6)

#         if len(collision_pts) == 0:
#             A = A_box
#             B = B_box
#         else:
#             keep_out = np.linalg.norm((X @ collision_pts.T), axis=0)

#             # assert (keep_out > 1 - 1e-3).all()

#             keep_out = np.linalg.norm((X @ collision_pts.T), axis=0)
#             keep_in = np.linalg.norm((X @ data_in.T ), axis=0)

#             # assert (keep_in < 1 + 1e-3).all()

#             As = []
#             Bs = []

#             closest_pts = []

#             while True:
#                 dists = np.linalg.norm((X @ collision_pts.T), axis=0)
#                 ind_min = np.argmin(dists)

#                 closest_pt = collision_pts[ind_min]

#                 X = X / dists[ind_min]

#                 a = 2* X.T @ X @ closest_pt
#                 b = a.T @ closest_pt

#                 As.append(a)
#                 Bs.append(b)
#                 closest_pts.append(closest_pt)

#                 collision_pts = np.delete(collision_pts, ind_min, axis=0)

#                 if len(collision_pts) > 0:
#                     to_keep = (collision_pts @ a < b)
#                     collision_pts = collision_pts[to_keep]

#                     if len(collision_pts) == 0:
#                         break
#                 else:
#                     break

#             #  Construct Polytope
#             A = np.stack(As, axis=0)
#             B = np.array(Bs)

#             A = np.concatenate([A, A_box], axis=0)
#             B = np.concatenate([B, B_box], axis=0)

#         # Shift coordinate system back
#         B = B + A @ mid

#         return A, B