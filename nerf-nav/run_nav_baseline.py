#%% 
import os
import numpy as np
import torch
import json

#Import utilies
from nav import (Planner, vec_to_rot_matrix)
from nerf.nerf import NeRFWrapper

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

#%%

# Stonehenge
# nerfwrapper = NeRFWrapper("./outputs/stonehenge/nerfacto/2023-10-26_111046")
# exp_name = 'stonehenge'
# world_frame = False

# Statues
# nerfwrapper = NeRFWrapper("./outputs/statues/nerfacto/2023-07-09_182722")
# exp_name = 'statues'
# world_frame = False

# Flightroom
nerfwrapper = NeRFWrapper("./outputs/flightroom/nerfacto/2023-10-15_232532")
exp_name = 'flightroom'
world_frame = True

### Baseline configs

# Stonehenge
# grid = np.array([
#     [-1.4, 1.1],
#     [-1.4, 1.1],
#     [-0.1, 0.5]
#     ])   
# body_nbins = [5, 5, 17]

# Statues
# grid = np.array([
#     [-1., 1.],
#     [-1, 1.],
#     [-.5, .5]
#     ])   
# body_nbins = [9, 9, 15]

# Flightroom
grid = np.array([
    [-1., 1.],
    [-1., 1.],
    [-0.5, 0.5]
    ])   
body_nbins = [9, 9, 15]

#Create robot body
agent_body = .03*np.array([[-1, 1], [-1, 1], [-0.3, 0.3]])

### ----- NERF-NAV PARAMETERS ----- #

mass = 1.           # mass of drone
g = 9.81             # gravitational constant
# Approximate as a disk (1/4 MR^2, 1/4 MR^2, 1/2 MR^2)
I = [[.01, 0, 0], [0, .01, 0], [0, 0, 0.02]]   # inertia tensor assuming drone = disk

### PLANNER CONFIGS, device=device
# X, Y, Z

# Rotation vector
start_R = [0., 0., 0.0]     # Starting orientation (Euler angles)
end_R = [0., 0., 0.0]       # Goal orientation

# Angular and linear velocities
init_rates = torch.zeros(3) # All rates

# Change rotation vector to rotation matrix 3x3
start_R = vec_to_rot_matrix( torch.tensor(start_R))
end_R = vec_to_rot_matrix(torch.tensor(end_R))

# Configuration
N_test = 100
t = np.linspace(0, np.pi, N_test)

# Stonehenge
# r = 1.12
# dz = 0.05
# center = np.array([-0.21, -0.132, 0.16])

# Statues
# r = 0.475
# dz = 0.05
# center = np.array([-0.064, -0.0064, -0.025])

# Flightroom
r = 2.6
dz = 0.2
center = np.array([0.10, 0.057, 0.585])

x0 = np.stack([r*np.cos(t), r*np.sin(t), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)
xf = np.stack([r*np.cos(t + np.pi), r*np.sin(t + np.pi), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)

x0 = x0 + center
xf = xf + center

x0 = torch.from_numpy(x0).to(dtype=torch.float32)
xf = torch.from_numpy(xf).to(dtype=torch.float32)

#%% Run

T_final = 10.                # Final time of simulation

planner_lr = 1e-3          # Learning rate when learning a plan
epochs_init = 1000          # Num. Gradient descent steps to perform during initial plan
fade_out_epoch = 0
fade_out_sharpness = 10
epochs_update = 250         # Num. grad descent steps to perform when replanning

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
'exp_name': exp_name,                  # Experiment name
'I': torch.tensor(I).float().to(device),
'g': g,
'mass': mass,
'body': agent_body,
'nbins': body_nbins,
'penalty': None,
'astar': None
}

# Load in Astar initialization
astar_fp = f'baseline_paths/{exp_name}/path.json'
with open(astar_fp, 'r') as f:
    meta = json.load(f)

astar_paths = [nerfwrapper.data_frame_to_ns_frame(torch.tensor(astar).to(device, dtype=torch.float32)).squeeze().cpu() for astar in meta['astar']]
astar_counter = 0

penalties = [1e4]
for penalty in penalties:
    planner_cfg['penalty'] = penalty
    for it, (start, end) in enumerate(zip(x0, xf)):

        if world_frame:
            start = nerfwrapper.data_frame_to_ns_frame(start.to(device)).squeeze().cpu()
            end = nerfwrapper.data_frame_to_ns_frame(end.to(device)).squeeze().cpu()
            
        # match the astar initialization with the current config
        if torch.linalg.norm(astar_paths[astar_counter][0] - start) <= 1e-1:
            print('Found Astar match')

            planner_cfg['astar'] = astar_paths[astar_counter].cpu().numpy()
            astar_counter += 1
        else:
            print(f'Point {it} did not match.')
            continue

        start_state = torch.cat( [start[:3], init_rates, start_R.reshape(-1), init_rates], dim=0)
        end_state   = torch.cat( [end[:3],   init_rates, end_R.reshape(-1), init_rates], dim=0 )
        planner_cfg['start_state'] = start_state.to(device)
        planner_cfg['end_state'] = end_state.to(device)
        
        simulate(planner_cfg, nerfwrapper.get_density, it)
    astar_counter = 0

end_text = 'End of simulation'
print(f'{end_text:.^20}')

# %%
