#%% 
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json

save_path = './data/nerfstudio/flight_room/mocap_transform.json'
reference_path = './data/nerfstudio/flight_room/transforms.json'

data = pd.read_csv('./data/nerfstudio/flight_room/data.csv')

with open(reference_path, 'r') as f:
    meta = json.load(f)
#%% 
x = np.array(data['field.pose.position.x'])
y = np.array(data['field.pose.position.y'])
z = np.array(data['field.pose.position.z'])

wx = np.array(data['field.pose.orientation.x'])
wy = np.array(data['field.pose.orientation.y'])
wz = np.array(data['field.pose.orientation.z'])
ww = np.array(data['field.pose.orientation.w'])

r = R.from_quat([ww[0], wx[0], wy[0], wz[0]])
rot = r.as_matrix()
trans = np.array([x[0], y[0], z[0]])

# T = np.eye(4)
# T[:3, :3] = rot
# T[:3, -1] = trans

T = np.array([
    [0., 0., -1., 0.],
    [-1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.]
])

frames = meta["frames"]
frame = frames[0]
ref_transform = np.array(frame["transform_matrix"])

to_mocap_T = T @ np.linalg.inv(ref_transform)

from_mocap_T = np.linalg.inv(to_mocap_T)

meta = {
    'transform_matrix': from_mocap_T.tolist()
}
with open(save_path, 'w') as f:
    json.dump(meta, f, indent=4)
