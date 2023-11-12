#%% 
import numpy as np
from .splines_utils import create_time_pts, get_qp_matrices 
import cvxpy as cvx

class SplinePlanner():
    def __init__(self, spline_deg=8, N_sec=10) -> None:
        self.spline_deg = spline_deg
    
        ### Create the time points matrix/coefficients for the Bezier curve
        self.time_pts = create_time_pts(deg=spline_deg, N_sec=N_sec)

    def optimize_b_spline(self, As, Bs, x0, xf, derivatives=None):
        self.calculate_b_spline_coeff(As, Bs, x0, xf, derivatives=None)
        return self.eval_b_spline()

    def calculate_b_spline_coeff(self, As, Bs, x0, xf, derivatives=None):
        N_sections = len(As)         #Number of segments

        T = self.time_pts['time_pts']
        dT = self.time_pts['d_time_pts']
        ddT = self.time_pts['dd_time_pts']
        dddT = self.time_pts['ddd_time_pts']
        ddddT = self.time_pts['dddd_time_pts']

        # Copy time points N times
        T_list = [T]*N_sections
        dT_list = [dT]*N_sections
        ddT_list = [ddT]*N_sections
        dddT_list = [dddT]*N_sections
        ddddT_list = [ddddT]*N_sections

        if derivatives is None:
            v0, a0, j0 = None, None, None
        else:
            v0, a0, j0 = derivatives['vel'] , derivatives['accel'], derivatives['jerk']

        #Set up CVX problem
        A_prob, b_prob, C_prob, d_prob, P_prob = get_qp_matrices(T_list, dT_list, ddT_list, dddT_list, ddddT_list, As, Bs, x0, xf, 
                                vel0=v0, accel0=a0, jerk0=j0)
        
        n_var = C_prob.shape[-1]

        x = cvx.Variable(n_var)
        x_ = cvx.reshape(x, (N_sections, 3*T.shape[0]), order='C')
        cost = 0
        for pts_flat in x_:
            control_pts = cvx.reshape(pts_flat, (3, T.shape[0]), order='C').T

            cost += cvx.pnorm(control_pts[:-1] - control_pts[1:], 2)
        #cost = cvx.pnorm(Q @ x, 2)
        # cost = cvx.pnorm(x_reshaped, 2)
        obj = cvx.Minimize(cost)

        constraints = [A_prob @ x <= b_prob, C_prob @ x == d_prob]

        prob = cvx.Problem(obj, constraints)

        prob.solve()

        coeffs = []
        cof_splits = np.split(x.value, N_sections)
        for cof_split in cof_splits:
            xyz = np.split(cof_split, 3)
            cof = np.stack(xyz, axis=0)
            coeffs.append(cof)

        self.coeffs = np.array(coeffs)
        return self.coeffs, prob.value

    def eval_b_spline(self):
        T = self.time_pts['time_pts']
        dT = self.time_pts['d_time_pts']
        ddT = self.time_pts['dd_time_pts']
        dddT = self.time_pts['ddd_time_pts']
        ddddT = self.time_pts['dddd_time_pts']

        full_traj = []
        for coeff in self.coeffs:
            pos = (coeff @ T).T
            vel = (coeff @ dT).T
            acc = (coeff @ ddT).T
            jerk = (coeff @ dddT).T
            sub_traj = np.concatenate([pos, vel, acc, jerk], axis=-1)
            full_traj.append(sub_traj)

        return np.concatenate(full_traj, axis=0)

#%% 
# import json
# import time

# fp = 'bounds.json'
# with open(fp, "r") as infile:
#     meta = json.load(infile)

# As = meta['As']
# Bs = meta['Bs']
# x0 = meta['x0']
# xf = meta['xf']

# planner = SplinePlanner(spline_deg=5, N_sec=10)
# tnow = time.time()
# traj = planner.optimize_b_spline(As, Bs, x0, xf, derivatives=None)
# print(time.time() - tnow)