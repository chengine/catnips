#%% 
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json
from PIL import Image
import os

save_path = './data/nerfstudio/flight_room/transforms.json'
reference_path = './data/nerfstudio/flight_room/transforms_ref.json'
filepath = './data/nerfstudio/flight_room/images_png/'
mask_path = './data/nerfstudio/flight_room/masks/'
rgb_path = './data/nerfstudio/flight_room/images/'

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None:
            images.append(np.array(img))
            filenames.append(filename.split('.')[0])
    return images, filenames

#%% Load mocap data
data = pd.read_csv('./data/nerfstudio/flight_room/mocap.csv')

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

T = np.eye(4)
T[:3, :3] = rot
T[:3, -1] = trans

#%% Load image data

imgs, filenames = load_images_from_folder(filepath)

# Generate mask
H, W = imgs[0].shape[:-1]

for img, fn in zip(imgs, filenames):
    rgb = img[..., :3]
    alpha = img[..., -1]

    rgb = Image.fromarray(rgb)
    rgb.save(rgb_path + fn + '.jpg')

    alpha = Image.fromarray(alpha)
    alpha.save(mask_path + fn + '.jpg')

# %% Change transform file
with open(reference_path, 'r') as f:
    meta = json.load(f)

frames = meta["frames"]

new_frames = []
for it, frame in enumerate(frames):
    ref_transform = np.array(frame["transform_matrix"])

    # if it == 0:
    #     T = T @ np.linalg.inv(ref_transform)

    # ref_transform = T @ ref_transform

    new_frame = {
    'transform_matrix': ref_transform.tolist(),
    'file_path': "images/" + fn + ".jpg",
    'mask_path': "masks/" + fn + ".jpg"
    }

    new_frames.append(new_frame)

meta["frames"] = new_frames

with open(save_path, 'w') as f:
    json.dump(meta, f, indent=4)
