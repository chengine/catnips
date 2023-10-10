import torch
import numpy as np
import json
import heapq 
import matplotlib.pyplot as plt
from nav.math_utils import skew_matrix

class Simulator:

    def __init__(self, start_state):
        self.states = start_state[None, :]

        self.mass = 1
        self.I = torch.eye(3)
        self.invI = torch.eye(3)
        self.dt = 0.1
        self.g = 10

    def add_state(self, state):
        self.states = torch.cat( [self.states, state[None,:] ], dim=0 )

    def copy_states(self, states):
        self.states = states

    def advance(self, action, state_noise=None):
        if state_noise == None:
            state_noise = 0
        next_state = self.next_state(self.states[-1, :], action) + state_noise
        self.states = torch.cat( [self.states, next_state[None,:] ], dim=0 )

    def advance_smooth(self, action, detail = 5):
        cur = self.states[-1, :]

        for _ in range(detail):
            cur = self.next_state(cur, action, self.dt/detail)

        self.states = torch.cat( [self.states, cur[None,:] ], dim=0 )

    def get_current_state(self):
        return self.states[-1,:]

    def body_to_world(self, points):
        pos = self.states[:, 0:3]
        v   = self.states[:, 3:6]
        R_flat = self.states[:, 6:15]
        R = R_flat.reshape((-1, 3, 3))
        omega = self.states[:, 15:]

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  R @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def next_state(self, state, action, dt = None):
        #State is 18 dimensional [pos(3), vel(3), R (9), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
        if dt == None:
            dt = self.dt

        next_state = torch.zeros(18)

        #Actions are [total thrust, torque x, torque y, torque z]
        fz = action[0]
        tau = action[1:]

        #Define state vector
        pos = state[0:3]
        v   = state[3:6]
        R_flat = state[6:15]
        R = R_flat.reshape((3, 3))
        omega = state[15:]

        # The acceleration
        sum_action = torch.zeros(3)
        sum_action[2] = fz

        dv = (torch.tensor([0,0,-self.mass*self.g]) + R @ sum_action)/self.mass

        # The angular accelerations
        domega = self.invI @ (tau - torch.cross(omega, self.I @ omega))

        # Propagate rotation matrix using exponential map of the angle displacements
        next_R = next_rotation(R, omega, dt)

        next_state[0:3] = pos + v * dt
        next_state[3:6] = v + dv * dt

        next_state[6:15] = next_R.reshape(-1)

        next_state[15:] = omega + domega * dt

        return next_state

    def save_poses(self, filename):
        positions = self.state[:,0:3]
        v   = self.state[:,3:6]
        rot_matrix = self.state[:,6:15].reshape((-1, 3, 3))
        omega = self.state[:,15:]

        with open(filename,"w+") as f:
            for pos, rot in zip(positions, rot_matrix):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.detach().numpy()
                pose[:3, 3]  = pos.detach().numpy()
                pose[3,3] = 1

                json.dump(pose.tolist(), f)
                f.write('\n')


class QuadPlot:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 8))

        self.ax_map = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_graph = self.fig.add_subplot(1, 2, 2)
        self.ax_graph_right = self.ax_graph.twinx()

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.02, 0.02,  5)), dim=-1)
        self.robot_body = body.reshape(-1, 3)

        self.fig.tight_layout()

    def trajectory(self, traj, color = "g", show_cloud=True):
        ax = self.ax_map
        ax.auto_scale_xyz([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax.set_ylim3d(-1, 1)
        ax.set_xlim3d(-1, 1)
        ax.set_zlim3d( 0, 1)

        # PLOT PATH
        # S, 1, 3
        pos = traj.body_to_world( torch.zeros((1,3))).detach().numpy()
        # print(pos.shape)
        ax.plot( pos[:,0,0], pos[:,0,1],   pos[:,0,2],  )

        if show_cloud:
            # PLOTS BODY POINTS
            # S, P, 2
            body_points = traj.body_to_world( self.robot_body ).detach().numpy()
            for i, state_body in enumerate(body_points):
                if isinstance(color, list):
                    c = color[i] + "."
                else:
                    c = color + "."
                ax.plot( *state_body.T, c, ms=72./ax.figure.dpi, alpha = 0.5)

        # PLOTS AXIS
        # create point for origin, plus a right-handed coordinate indicator.
        size = 0.05
        points = torch.tensor( [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])
        colors = ["r", "g", "b"]

        # S, 4, 2
        points_world_frame = traj.body_to_world(points).detach().numpy()
        for state_axis in points_world_frame:
            for i in range(1, 4):
                ax.plot(state_axis[[0,i], 0],
                        state_axis[[0,i], 1],
                        state_axis[[0,i], 2],
                    c=colors[i - 1],)

    def plot_data(self, *args, **kwargs):
        self.ax_graph.plot(*arg, **kawrgs)

    def show(self):
        # sadly this messed with using ctrl-c after running
        plt.show()
        # plt.ion()
        # show = True

        # def handle_close(event):
        #     nonlocal show
        #     show = False
        #     print("Stop on close")

        # self.fig.canvas.mpl_connect("close_event", handle_close)
        # plt.show(block=False)

        # while show:
        #     plt.pause(1)
        # plt.close(self.fig)


def next_rotation(R, omega, dt):
    # Propagate rotation matrix using exponential map of the angle displacements
    angle = omega*dt
    theta = torch.norm(angle, p=2)
    if theta == 0:
        exp_i = torch.eye(3)
    else:
        exp_i = torch.eye(3)
        angle_norm = angle / theta
        K = skew_matrix(angle_norm)
        exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

    next_R = R @ exp_i
    return next_R

def astar(occupied, start, goal):
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)

    def inbounds(point):
        for x, size in zip(point, occupied.shape):
            if x < 0 or x >= size: return False
        return True

    neighbors = [( 1,0,0),(-1, 0, 0),
                 ( 0,1,0),( 0,-1, 0),
                 ( 0,0,1),( 0, 0,-1)]

    close_set = set()

    came_from = {}
    gscore = {start: 0}

    assert not occupied[start]
    assert not occupied[goal]

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            assert current == start
            data.append(current)
            return list(reversed(data))

        close_set.add(current)

        for i, j, k in neighbors:
            neighbor = (current[0] + i, current[1] + j, current[2] + k)
            if not inbounds( neighbor ):
                continue

            if occupied[neighbor]:
                continue

            tentative_g_score = gscore[current] + 1

            if tentative_g_score < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score

                fscore = tentative_g_score + heuristic(neighbor, goal)
                node = (fscore, neighbor)
                if node not in open_heap:
                    heapq.heappush(open_heap, node) 

    raise ValueError("Failed to find path!")

def settings():
    pass
    #PARAM
    # cfg = {"T_final": 2,
    #         "steps": 20,
    #         "lr": 0.001,#0.001,
    #         "epochs_init": 500, #2000,
    #         "fade_out_epoch": 0,#1000,
    #         "fade_out_sharpness": 10,
    #         "epochs_update": 500,
    #         }

    #PARAM nerf config

    #PARAM start and end positions for the planner. [x,y,z,yaw]

    # renderer = get_nerf('configs/playground.txt')
    # playgroud - under
    # start_state = torch.tensor([0, -0.8, 0.01, 0])
    # end_state   = torch.tensor([0,  0.9, 0.6 , 0])
    
    # playgroud - upper
    # start_state = torch.tensor([-0.11, -0.7, 0.7, 0])
    # end_state   = torch.tensor([-0.11, 0.45, 0.7, 0])

    # playground - diag
    # start_state = torch.tensor([ 0.25, -0.47, 0.01, 0])
    # end_state   = torch.tensor([-0.25,  0.6,  0.6 , 0])

    # playground - middle
    # start_state = torch.tensor([ 0.5, 0.2, 0.3, 0])
    # end_state   = torch.tensor([-0.3,   0, 0.5 , 0])

    renderer = get_nerf('configs/violin.txt')
    # violin - simple
    # start_state = torch.tensor([-0.3 ,-0.5, 0.1, 0])
    # end_state   = torch.tensor([-0.35, 0.7, 0.15 , 0])

    # violin - dodge
    # start_state = torch.tensor([-0.35,-0.5, 0.05, 0])
    # end_state   = torch.tensor([ 0.1,  0.6, 0.3 , 0])

    # violin - middle
    # start_state = torch.tensor([0,-0.5, 0.1, 0])
    # end_state   = torch.tensor([0, 0.7, 0.15 , 0])


    # renderer = get_nerf('configs/stonehenge.txt')
    # stonehenge - simple
    start_state = torch.tensor([-0.05,-0.9, 0.2, 0])
    end_state   = torch.tensor([-0.2 , 0.7, 0.15 , 0])

    # stonehenge - tricky
    # start_state = torch.tensor([ 0.4 ,-0.9, 0.2, 0])
    # end_state   = torch.tensor([-0.2 , 0.7, 0.15 , 0])

    # stonehenge - very simple
    # start_state = torch.tensor([-0.43, -0.75, 0.2, 0])
    # end_state = torch.tensor([-0.26, 0.48, 0.15, 0])

    #PARAM initial and final velocities
    start_vel = torch.tensor([0, 0, 0, 0])
    end_vel   = torch.tensor([0, 0, 0, 0])

# def rot_matrix_to_vec(R):
#     batch_dims = R.shape[:-2]

#     trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

#     def acos_safe(x, eps=1e-7):
#         """https://github.com/pytorch/pytorch/issues/8069"""
#         slope = np.arccos(1-eps) / eps
#         # TODO: stop doing this allocation once sparse gradients with NaNs (like in
#         # th.where) are handled differently.
#         buf = torch.empty_like(x)
#         good = abs(x) <= 1-eps
#         bad = ~good
#         sign = torch.sign(x[bad])
#         buf[good] = torch.acos(x[good])
#         buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
#         return buf

#     # angle = torch.acos((trace - 1) / 2)[..., None]
#     angle = acos_safe((trace - 1) / 2)[..., None]
#     # print(trace, angle)

#     vec = (
#         1
#         / (2 * torch.sin(angle + 1e-10))
#         * torch.stack(
#             [
#                 R[..., 2, 1] - R[..., 1, 2],
#                 R[..., 0, 2] - R[..., 2, 0],
#                 R[..., 1, 0] - R[..., 0, 1],
#             ],
#             dim=-1,
#         )
#     )

#     # needed to overwrite nanes from dividing by zero
#     vec[angle[..., 0] == 0] = torch.zeros(3, device=R.device)

#     # eg TensorType["batch_size", "views", "max_objects", 3, 1]
#     rot_vec = (angle * vec)[...]

#     return rot_vec

# def vec_to_rot_matrix(rot_vec):
#     assert not torch.any(torch.isnan(rot_vec))

#     angle = torch.norm(rot_vec, dim=-1, keepdim=True)

#     axis = rot_vec / (1e-10 + angle)
#     S = skew_matrix(axis)
#     # print(S.shape)
#     # print(angle.shape)
#     angle = angle[...,None]
#     rot_matrix = (
#             torch.eye(3)
#             + torch.sin(angle) * S
#             + (1 - torch.cos(angle)) * S @ S
#             )
#     return rot_matrix

# def skew_matrix(vec):
#     batch_dims = vec.shape[:-1]
#     S = torch.zeros(*batch_dims, 3, 3)
#     S[..., 0, 1] = -vec[..., 2]
#     S[..., 0, 2] =  vec[..., 1]
#     S[..., 1, 0] =  vec[..., 2]
#     S[..., 1, 2] = -vec[..., 0]
#     S[..., 2, 0] = -vec[..., 1]
#     S[..., 2, 1] =  vec[..., 0]
#     return S
