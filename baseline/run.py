#%% 
import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R
import time

#Import utilies
from nerf.nerf import NeRFWrapper
from baseline_grid.baseline_grid import BaselineGrid
from corridor.init_path import PathInit
from corridor.bounds import BoxCorridor, PolytopeCorridor
from planner.spline_planner import SplinePlanner
from planner.mpc import MPC 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

# Stonehenge
nerfwrapper = NeRFWrapper("./outputs/stonehenge/nerfacto/2023-10-26_111046")
exp_name = 'stonehenge'

# Statues
# nerfwrapper = NeRFWrapper("./outputs/statues/nerfacto/2023-07-09_182722")
# exp_name = 'statues'

# Flightroom
# nerfwrapper = NeRFWrapper("./outputs/flightroom/nerfacto/2023-10-15_232532")
# exp_name = 'flightroom'

world_frame = False

#%%
### Catnips configs

# Stonehenge
grid = np.array([
    [-1.4, 1.1],
    [-1.4, 1.1],
    [-0.1, 0.5]
    ])   

# Statues
# grid = np.array([
#     [-1., 1.],
#     [-1, 1.],
#     [-.5, .5]
#     ])   

# Flightroom
# grid = np.array([
#     [-1., 1.],
#     [-1., 1.],
#     [-0.5, 0.5]
#     ])   

#Create robot body
agent_body = .02*np.array([[-1, 1], [-1, 1], [-0.3, 0.3]])

# #Configs
cutoff = 1e3
discretization = 150

basegrid_configs = {
    'grid': grid,               # Bounding box of scene
    'agent_body': agent_body,   # Bounding box of agent in body frame
    'cutoff': cutoff,             # chance of being below interpenetration vol.
    'discretization': discretization,   # number of partitions per side of voxel grid
    'density_factor': 1,        # scaling factor to density
    'get_density': nerfwrapper.get_density  # queries NeRF to get density
}

basegrid = BaselineGrid(basegrid_configs)      # Instantiate class
basegrid.load_basegrid()                     # MUST load details about the PURR
basegrid.create_basegrid()                 # Generates PURR voxel grid
basegrid.save_basegrid(f'./basegrid_data/{exp_name}/basegrid/', transform=nerfwrapper.transform.cpu().numpy(), scale=nerfwrapper.scale, save_property=True)

# basegrid.create_basegrid_simple(1e3)
# basegrid.save_basegrid_simple(f'./basegrid_data/{exp_name}/basegrid/', transform=nerfwrapper.transform.cpu().numpy(), scale=nerfwrapper.scale)
#%%

# TODO: need to change occupied points to be vertices instead of centers for corridor generation (A* ok to use centers)
occupied_vertices = basegrid.basegrid_vertices
astar_path = PathInit(basegrid.basegrid, basegrid.conv_centers)

N_test = 100
t = np.linspace(0, np.pi, N_test)
num_sec = 20

# Stonehenge
r = 1.12
dz = 0.05
center = np.array([-0.21, -0.132, 0.16])

# Statues
# r = 0.475
# dz = 0.05
# center = np.array([-0.064, -0.0064, -0.025])

# Flightroom
# r = 2.6
# dz = 0.2
# center = np.array([0.10, 0.057, 0.585])

x0 = np.stack([r*np.cos(t), r*np.sin(t), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)
xf = np.stack([r*np.cos(t + np.pi), r*np.sin(t + np.pi), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)

x0 = x0 + center
xf = xf + center

list_plan = []
list_astar = []

corridor = BoxCorridor(basegrid.basegrid, basegrid.conv_centers, r=0.1)
# planner = SplinePlanner(spline_deg=3, N_sec=10)
planner = SplinePlanner() # MPC(N=40)

for it, (start, end) in enumerate(zip(x0, xf)):
    if world_frame:
        x0_ns = nerfwrapper.data_frame_to_ns_frame(torch.from_numpy(start).to(device, dtype=torch.float32)).squeeze().cpu().numpy()
        xf_ns = nerfwrapper.data_frame_to_ns_frame(torch.from_numpy(end).to(device, dtype=torch.float32)).squeeze().cpu().numpy()

    else:
        x0_ns = start
        xf_ns = end

    try:
        # This is in the ns frame
        path, straight_path = astar_path.create_path(x0_ns, xf_ns, num_sec=num_sec)

        # cat_path = nerfwrapper.ns_frame_to_data_frame(torch.from_numpy(cat_path).to(device, dtype=torch.float32)).squeeze()

        # Create bounds
        # Convert this to the world frame

        # path = nerfwrapper.ns_frame_to_data_frame(torch.from_numpy(path).to(device, dtype=torch.float32)).squeeze().cpu().numpy()
        # occupied_pts = nerfwrapper.ns_frame_to_data_frame(torch.from_numpy(occupied_pts).to(device, dtype=torch.float32)).squeeze().cpu().numpy()
        
        tnow = time.time()
        As, Bs, bounds = corridor.create_corridor(straight_path)    
        print('Elapsed', time.time() - tnow)

        corridor.bounds2mesh(Bs, f'./basegrid_data/{exp_name}/bounds/', transform=nerfwrapper.transform.cpu().numpy(), scale=nerfwrapper.scale, i=it)

        # Create dynamically feasible/smooth path
        # try:
        traj = planner.optimize_b_spline(As, Bs, straight_path[0], straight_path[-1], derivatives=None)

        # traj, efforts = planner.solve(As, Bs, x0_ns, xf_ns)
        if world_frame:
            traj = nerfwrapper.ns_frame_to_data_frame(torch.from_numpy(traj[..., :3]).to(device, dtype=torch.float32)).squeeze().cpu().numpy()
        # traj = nerfwrapper.ns_frame_to_data_frame(torch.from_numpy(path[..., :3]).to(device, dtype=torch.float32)).squeeze().cpu().numpy()

    # except:
    #     print('Unfeasible solution. Using initialization.')
    #     traj = path

    # derivs = {
    #     'vel0': np.zeros(3),
    #     'accel0': np.zeros(3),
    #     'jerk0': np.zeros(3)
    # }

        list_plan.append(traj)
        list_astar.append(straight_path)
    except:
        print('Error in start/end locations or unable to optimize spline.')

data = {
    'traj': [plan.tolist() for plan in list_plan],
    'astar': [plan.tolist() for plan in list_astar],
    }

fp = f'basegrid_data/{exp_name}/path.json'
with open(fp, "w") as outfile:
    json.dump(data, outfile, indent=4)

# %%
