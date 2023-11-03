#%% 
import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R

#Import utilies
from nerf.nerf import NeRFWrapper
from catnips.purr_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

# Flightroom
# "./outputs/flightroom/nerfacto/2023-06-22_215809"

# Statues
# "./outputs/statues/nerfacto/2023-07-09_182722"

nerfwrapper = NeRFWrapper("./outputs/statues/nerfacto/2023-07-09_182722")

# In the world frame
x0_world = np.array([.5, 0.5, 0.1])
xf_world = np.array([-.5, 0.5, 0.1])

# Convert configuration to NeRF coordinates (if necessary)
x0_ns = nerfwrapper.data_frame_to_ns_frame(x0_world)
xf_ns = nerfwrapper.data_frame_to_ns_frame(xf_world)

# grid = np.array([
#     [2.75, 8.25],
#     [-2.5, 2.5],
#     [0., 3.]
#     ])   

# if world_transform is not None:
#     x0 = world_transform[:3, :3] @ x0 + world_transform[:3, -1]
#     xf = world_transform[:3, :3] @ xf + world_transform[:3, -1]
    
    # grid = world_transform[:3, :3] @ grid + world_transform[:3, -1][:, None]

    # # Sort grid after transformation
    # grid = np.sort(grid)

grid = np.array([
    [-1., 1.],
    [-1., 1.],
    [-0.5, 0.5]
    ])   

#Create robot body
agent_lims = .05*np.array([[-1, 1], [-1, 1], [-0.3, 0.3]])

# #Configs
sigma = 0.01
spline_deg = 8
discretization = 100

catnips_configs = {
    'spline_deg': spline_deg,
    'sigma': sigma,
    'discretization': discretization,
    'dilation': None
}

position_configs = {
    'start': x0,
    'end': xf
}

save_dir = f'path_data/sigma{sigma}'
save_purr_fp = f'purr_data/purr_sigma_{sigma}'

catnips_planner = CATNIPS(data_path=None, grid=grid, agent_body=agent_lims, 
                        configs=catnips_configs, position_configs=position_configs, 
                        get_density=query_density)

catnips_planner.save_purr(save_purr_fp, transform=np.linalg.inv(world_transform))

# Execute

derivs = {
    'vel0': np.zeros(3),
    'accel0': np.zeros(3),
    'jerk0': np.zeros(3)
}

t = np.linspace(0, 2*np.pi, 100, endpoint=False)
start = np.stack([0.5*np.cos(t), 0.5*np.sin(t), -0.1*np.ones_like(t)], axis=-1)

if world_transform is not None:
    start = (world_transform[:3, :3] @ start.T).T + world_transform[:3, -1][None, :]

end = np.roll(start, 50, axis=0)

full_traj = []
for (x0, xf) in zip(start, end):
    traj, success = catnips_planner.get_traj(x0, xf=xf, N=20, derivs=derivs)
    # full_traj.append(traj.tolist())

    pt_in_world = (np.linalg.inv(world_transform[:3, :3]) @ (traj[:, :3] - world_transform[:3, -1][None, :]).T).T
    full_traj.append(pt_in_world.tolist())

data = {
    'traj': full_traj
    }

fp = 'purr_data/path.json'
with open(fp, "w") as outfile:
    json.dump(data, outfile, indent=4)

# full_traj = []

# while True:
#     # traj = catnips_planner.get_traj(x0, N=20, save=True, save_dir=save_dir)
#     traj = catnips_planner.get_traj(x0, N=20, separate=True, derivs=derivs)

#     # traj is T x N x 3 if separate is True, TN x 3 if False 
#     next_pt = traj[0][10]
#     if np.linalg.norm(x0[:3] - xf[:3]) < 0.1:
#         break

#     x0 = next_pt[:3]

#     # derivs = {
#     #     'vel0': next_pt[3:6],
#     #     'accel0': None, # next_pt[6:9],
#     #     'jerk0': None, #next_pt[9:]
#     # } 

#     pt_in_world = np.linalg.inv(world_transform[:3, :3]) @ (x0 - world_transform[:3, -1])

#     full_traj.append(pt_in_world)
#     print(pt_in_world)

# full_traj = np.stack(full_traj, axis=0)
# %%
