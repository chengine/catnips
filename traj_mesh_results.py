#%% 
import os
import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt
import time

name = 'stonehenge'
mesh_fp = f'{name}.ply'

mesh_ = o3d.io.read_triangle_mesh(mesh_fp)
mesh_ = mesh_.filter_smooth_taubin(number_of_iterations=100)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_)

# Create robot body
r = 0.0289
X, Y, Z = np.meshgrid(
    np.linspace(-1., 1., 200, endpoint=True).astype(np.float32),
    np.linspace(-1., 1., 200, endpoint=True).astype(np.float32),
    np.linspace(-1., 1., 200, endpoint=True).astype(np.float32)
)
grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
body = grid_points[np.linalg.norm(grid_points, axis=-1) <=1.]
body = r*body

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

# signed distance is a [32,32,32] array
sdf_query = lambda x: scene.compute_signed_distance(x).numpy()

def volume_intersection_query(xyz):
    body_trans = body + xyz
    sdfs = sdf_query(body_trans)
    percent_intersect = np.sum(sdfs <= 0.) / body.shape[0]
    print(np.sum(sdfs <= 0.), body.shape[0])
    return 4/3 * np.pi * (r**3) * percent_intersect

method = 'catnips' # or baseline or nerfnav
exp_name = 'stonehenge'
fp = f'{method}/catnips_data/{exp_name}/path.json'
with open(fp, 'r') as f:
    meta = json.load(f)

traj = meta['traj']
traj = np.array(traj, dtype=np.float32)

sdfs = []
vols = []
for i, sub_traj in enumerate(traj):
    # Each one is one trajectory with N points
    sub_traj_sdf = []
    sub_traj_vol = []
    for pt in sub_traj:
        sub_traj_sdf.append(sdf_query(pt.reshape(-1, 3)).squeeze())
        sub_traj_vol.append(volume_intersection_query(pt))

    sub_traj_sdf = np.stack(sub_traj_sdf)
    sub_traj_vol = np.stack(sub_traj_vol)
    sdfs.append(sub_traj_sdf)
    vols.append(sub_traj_vol)

    print('Trajectory ', i)
    raise
sdfs = np.stack(sdfs, axis=0)
vols = np.stack(vols, axis=0)    

# %%
save_fp = f'results_processed/{method}/{exp_name}'
if not os.path.exists(save_fp):
    # Create a new directory because it does not exist
    os.makedirs(save_fp)

data = {
    'sdfs': sdfs.tolist(),
    'vols': vols.tolist()
}

with open(save_fp + '/data.json', 'w') as f:
    json.dump(data, f, indent=4)