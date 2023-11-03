#%% 
import numpy as np
import torch
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R

#Import utilies
from catnips.purr_utils import *

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
# config_path = Path("./outputs/flightroom/nerfacto/2023-06-22_215809/config.yml") # Path to config file 

# with open("./outputs/flightroom/nerfacto/2023-06-22_215809/dataparser_transforms.json", 'r') as f:
#     transform = json.load(f)

config_path = Path("./outputs/statues/nerfacto/2023-07-09_182722/config.yml") # Path to config file 

with open("./outputs/statues/nerfacto/2023-07-09_182722/dataparser_transforms.json", 'r') as f:
    transform = json.load(f)

scale = transform["scale"]
transform = transform["transform"]

# Uncomment this if you want to plan in the unit cube
transform = np.eye(4)
scale = 1.


transform = torch.tensor(transform, device=device, dtype=torch.float32)

# This is from world coordinates to scene coordinates
# rot = R.from_euler('xyz', [0, 0., 30], degrees=True)
# world_transform = np.eye(4)
# world_transform[:3, :3] = rot.as_matrix()
# world_transform[:3, -1] = np.array([5., 0., 0.75])
# world_transform = np.linalg.inv(world_transform)
world_transform = np.eye(4, dtype=np.float32)

# Prepare model
_, pipeline, _, _ = eval_setup(
    config_path, 
    test_mode="inference",
)

#%% 
def get_density(xyz, transform=transform, scale=scale):
    pts = xyz @ transform[:3, :3].T + transform[:3, -1][None,:]
    pts *= scale

    ray_samples = RaySamples(
        frustums=Frustums(
            origins=pts,
            directions=torch.zeros_like(pts, device=device),
            starts=0,
            ends=0,
            pixel_area=None,
        )
    )
    density, _ = pipeline.model.field.get_density(ray_samples)
    return density

##### END OF NGP SPECIFIC ########--------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------#
### CATNIPS ###
#Create get_density function
query_density = lambda x: get_density(x, transform=transform, scale=scale)

# In the world frame
x0 = np.array([.5, 0.5, 0.1])
xf = np.array([-.5, 0.5, 0.1])

# grid = np.array([
#     [2.75, 8.25],
#     [-2.5, 2.5],
#     [0., 3.]
#     ])   

if world_transform is not None:
    x0 = world_transform[:3, :3] @ x0 + world_transform[:3, -1]
    xf = world_transform[:3, :3] @ xf + world_transform[:3, -1]
    
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