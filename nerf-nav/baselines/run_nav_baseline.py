#%% 
import numpy as np
import torch
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R

#Import utilies
from catnips.purr_utils import *
from nav import (Planner, vec_to_rot_matrix)

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################### MAIN LOOP ##########################################
def simulate(planner_cfg, density_fn, iteration):
    '''
    Encapsulates planning.
    '''

    start_state = planner_cfg['start_state']
    end_state = planner_cfg['end_state']
    penalty = planner_cfg['penalty']
    
    # Creates a workspace to hold all the trajectory data
    basefolder = f"baseline_paths/{planner_cfg['exp_name']}/{penalty}_iter{iteration}"
    try:
        os.makedirs(basefolder)
        os.mkdir(basefolder + "/init_poses")
        os.mkdir(basefolder + "/init_costs")
        print("created", basefolder)
    except:
        pass
  
    # Initialize Planner
    traj = Planner(start_state, end_state, planner_cfg, density_fn)

    traj.basefolder = basefolder

    # Create a coarse trajectory to initialize the planner by using A*. 
    traj.a_star_init()

    # From the A* initialization, perform gradient descent on the flat states of agent to get a trajectory
    # that minimizes collision and control effort.
    traj.learn_init()

####################### END OF MAIN LOOP ##########################################

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
#Create get_density function
query_density = lambda x: get_density(x, transform=transform, scale=scale)

grid = np.array([
    [-1., 1.],
    [-1., 1.],
    [-0.5, 0.5]
    ])   
discretization = 100

#Create robot body
agent_lims = .05*np.array([[-1, 1], [-1, 1], [-0.3, 0.3]])

# Discretizations of sample points in x,y,z direction
body_nbins = [10, 10, 5]

### ----- NERF-NAV PARAMETERS ----- #

mass = 1.           # mass of drone
g = 9.81             # gravitational constant
# Approximate as a disk (1/4 MR^2, 1/4 MR^2, 1/2 MR^2)
I = [[.01, 0, 0], [0, .01, 0], [0, 0, 0.02]]   # inertia tensor

### PLANNER CONFIGS
# X, Y, Z

# Rotation vector
start_R = [0., 0., 0.0]     # Starting orientation (Euler angles)
end_R = [0., 0., 0.0]       # Goal orientation

# Angular and linear velocities
init_rates = torch.zeros(3) # All rates

T_final = 10.                # Final time of simulation

planner_lr = 1e-4          # Learning rate when learning a plan
epochs_init = 500          # Num. Gradient descent steps to perform during initial plan
fade_out_epoch = 0
fade_out_sharpness = 10
epochs_update = 250         # Num. grad descent steps to perform when replanning

# Change rotation vector to rotation matrix 3x3
start_R = vec_to_rot_matrix( torch.tensor(start_R))
end_R = vec_to_rot_matrix(torch.tensor(end_R))

#In NeRF training, the camera is pointed along positive z axis, whereas Blender assumes -z, hence we need to rotate the pose
rot = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device=device, dtype=torch.float32)

# Main loop
N_trials = 100
trials = np.arange(0, N_trials)

# For statistical tests, generate positions in a ring. Fly from one end to another
r = 0.7
t = np.linspace(0, 2*np.pi, num=N_trials, endpoint=False)
x = r*np.cos(t)
y = r*np.sin(t)
z = -0.1*np.ones_like(x)
pos_init = np.stack([x, y, z, np.zeros(z.shape)], axis=-1)

pos_fin = np.roll(pos_init, 50, axis=0)
pos_fin[:, -1] =  np.pi*np.ones_like(x)

pos_init = torch.tensor(pos_init, dtype=torch.float32)
pos_fin = torch.tensor(pos_fin, dtype=torch.float32)

#%% Make Astar
kernel = make_kernel(body_lims, grid, discretization)

field, feasible, cell_sizes, extras = generate_avoid_zone(grid, kernel, N=discretization, 
scale=1e2, sigma=0.01, get_density=query_density, 
dilation=None, baseline=1e6)

source = euc_to_index(grid, xf[:3].cpu().numpy(), N=discretization, kernel=kernel)
parental_field = create_parental_field(extras['baseline_grid'], source)

def astar_path(start):
    target = euc_to_index(grid, start, N=discretization, kernel=kernel)
    traj, path = path_from_parent(parental_field, target, feasible)

    return traj

#%% Run

#Store configs in dictionary
planner_cfg = {
"T_final": T_final,
"lr": planner_lr,
"epochs_init": epochs_init,
"fade_out_epoch": fade_out_epoch,
"fade_out_sharpness": fade_out_sharpness,
"epochs_update": epochs_update,
'start_state': None,
'end_state': None,
'exp_name': 'statues',                  # Experiment name
'I': torch.tensor(I).float().to(device),
'g': g,
'mass': mass,
'body': agent_lims,
'nbins': body_nbins,
'penalty': None,
'astar_func': astar_path
}

penalties = [1e3, 1e6, 1e9]
for penalty in penalties:
    planner_cfg['penalty'] = penalty
    for it, (start, end) in enumerate(zip(pos_init, pos_fin)):
        start_state = torch.cat( [start[:3], init_rates, start_R.reshape(-1), init_rates], dim=0 )
        end_state   = torch.cat( [end[:3],   init_rates, end_R.reshape(-1), init_rates], dim=0 )
        planner_cfg['start_state'] = start_state.to(device)
        planner_cfg['end_state'] = end_state.to(device)
        
        simulate(planner_cfg, query_density, it)

end_text = 'End of simulation'
print(f'{end_text:.^20}')
