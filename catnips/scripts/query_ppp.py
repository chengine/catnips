#%%
import torch
import numpy as np
from pathlib import Path
from typing import List
import json
import time
import trimesh
#from torchquad import MonteCarlo, set_up_backend
# set_up_backend("torch", data_type="float")

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
config_path = Path("./outputs/flight_room/nerfacto/2023-06-18_220905/config.yml") # Path to config file 
filename = f'point_cloud_flightroom.ply'

with open("./outputs/flight_room/nerfacto/2023-06-18_220905/dataparser_transforms.json", 'r') as f:
    transform = json.load(f)

scale = transform["scale"]
transform = transform["transform"]
transform = np.array(transform)

# Prepare model
_, pipeline, _, _ = eval_setup(
    config_path, 
    test_mode="inference",
)

#%% 
def get_density(xyz, transform=transform, scale=scale):
    pts = transform[:3, :3] @ xyz + transform[:3, -1]
    pts *= scale

    ray_samples = RaySamples(
        frustums=Frustums(
            origins=pts,
            directions=torch.zeros_like(pts),
            starts=0,
            ends=0,
            pixel_area=None,
        ),
    )

    field_outputs = pipeline.model.field.forward(ray_samples, compute_normals=False)
    density = field_outputs[FieldHeadNames.DENSITY]
    return density

#Create get_density function
query_density = lambda x: get_density(x, transform=transform, scale=scale)

#%% PPP
with torch.no_grad():
    sim_window = torch.tensor([
        [-1, 1],
        [-1, 1],
        [-1, 1]], device=device)

    now = time.time()
    N_pts = 5000000
    mc = MonteCarlo()

    #Camera angle
    camera_angle = 0.6911112070083618 
    W = 400
    focal = 0.01
    #r = focal*np.tan(camera_angle/2)/(W)
    r = 0.05
    correction = 1/(np.pi * r**2)
    N = int(1e6)

    #Find volume integral over density to find intensity measure
    A = correction*mc.integrate(
            get_density,
            dim=3,
            N=N_pts,
            integration_domain=sim_window,
            backend='torch'
        )

    print(f'Volume integral: {A}')
    print(f'Volume integral time: {time.time() - now}')

    s = 200
    x, y, z = torch.meshgrid(torch.linspace(sim_window[0, 0], sim_window[0, 1], s), 
        torch.linspace(sim_window[1, 0], sim_window[1, 1], s), 
        torch.linspace(sim_window[2, 0], sim_window[2, 1], s))
    pts = torch.stack([x, y, z], axis=-1)
    query_pts = pts.view((-1, 3))
    dense = correction*get_density(query_pts)
    lambdaMax = torch.amax(dense).cpu().numpy()
    print(f'Maximum density: {lambdaMax}')
    print(f'Finding max density time: {time.time() - now}')
    ###END -- find maximum lambda -- END ###
    
    #define thinning probability function
    def fun_p(x):
        return correction*get_density(x)/A.cpu().numpy()
    
    #Simulate a Poisson point process
    
    #Simulate number of points present in window
    numbPoints = np.random.poisson(A.cpu().numpy())
    print(f'Number of points present: {numbPoints}')

    total_pts = 0
    pts_list = []
    while total_pts < numbPoints:
        pts = torch.rand((N, 3))
        pts[:, 0] = sim_window[0, 0] + (sim_window[0, 1] - sim_window[0, 0])*pts[:, 0]
        pts[:, 1] = sim_window[1, 0] + (sim_window[1, 1] - sim_window[1, 0])*pts[:, 1]
        pts[:, 2] = sim_window[2, 0] + (sim_window[2, 1] - sim_window[2, 0])*pts[:, 2]

        #calculate spatially-dependent thinning probabilities
        #pts_split = np.array_split(pts, numbPoints//chunk, axis=0)
        #p = []
        #for pt_chunk in pts_split:
        p = fun_p(pts)
        #    p.append(densities)
        #p = np.concatenate(p, axis=0)
        
        #Generate Bernoulli variables (ie coin flips) for thinning
        booleRetained=torch.rand((N, ))<p  #points to be thinned
        
        #x/y locations of retained points
        ptsRetained = pts[booleRetained, :]
        total_pts += ptsRetained.shape[0]
        pts_list.append(ptsRetained)
        print(total_pts)

pts_thinned = torch.cat(pts_list, dim=0)
print('Elapsed', time.time() - now)

#Save point cloud
pcd = trimesh.points.PointCloud(pts_thinned)
pcd.export(filename)
print(f'Saved point cloud to {filename}')