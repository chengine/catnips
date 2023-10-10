#%% Imports
import numpy as np
import torch
import cv2

from pathlib import Path
from typing import List

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def query_model_and_return(config_path: Path, poses: List[float]):
    # Given a pretrained model and a set of poses, query the model with the given poses and return the outputs
    # config_path: path to the pretrained model config file
    # poses: set of poses to query 

    # Prepare cameras
    poses = np.array(poses).astype(np.float32)

    image_height, image_width = 800, 800 #TODO: Should not be hardcode.
    camera_angle_x = 71.83499829361337*np.pi/180
    focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

    cx = image_width / 2.0
    cy = image_height / 2.0
    camera_to_world = torch.from_numpy(poses[:, :3])

    cameras = Cameras(
        camera_to_worlds=camera_to_world,
        fx=focal_length,
        fy=focal_length,
        cx=cx,
        cy=cy,
        camera_type=CameraType.PERSPECTIVE,
    )

    # Prepare model
    _, pipeline, _, _ = eval_setup(
        config_path, 
        test_mode="inference",
    )

    cameras = cameras.to(pipeline.device)

    colormap_options = colormaps.ColormapOptions(colormap="turbo")
    rendered_output_names = ["rgb", "depth"]

    rendered_image_lst = []
    for camera_idx in range(cameras.size):
        aabb_box = None
        camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        render_image = []
        for rendered_output_name in rendered_output_names:
            output_image = outputs[rendered_output_name]

            # convert depth from (bs, 1) to (bs, 3) which has (r, g, b) three channels
            if rendered_output_name == "depth":
                pass
                # output_image = (
                #     colormaps.apply_depth_colormap(
                #         depth=output_image,
                #         colormap_options=colormap_options,
                #     )
                #     .cpu()
                #     .numpy()
                # )
            else:
                output_image = (
                    colormaps.apply_colormap(
                        image=output_image,
                        colormap_options=colormap_options,
                    )
                    .cpu()
                    .numpy()
                )
            render_image.append(output_image)

        # render_image = np.concatenate(render_image, axis=1)
        rendered_image_lst.append(render_image)
    return rendered_image_lst

#%% 
config_path = Path("./outputs/statues/nerfacto/2023-07-09_182722/config.yml") # Path to config file 

pose = np.array(
    [-0.6860039234161377,
        0.14042872190475464,
        -0.22347505390644073,
        -0.2781355381011963,
        -0.263902485370636,
        -0.3746193051338196,
        0.5746986865997314,
        0.668873131275177,
        -0.004100356251001358,
        0.6166058778762817,
        0.40005379915237427,
        0.17839041352272034,
        0.0,
        0.0,
        0.0,
        1.0])
pose = pose.reshape(4, 4)

rendered_image_lst = query_model_and_return(config_path, [pose])

# %% Load in NeRF and Blender images
blender_depth = np.load('renders/depth.npz')['dmap']
blender_rgb = cv2.imread('renders/rgb.png')/255.
nerf_depth = rendered_image_lst[0][1].cpu().numpy()
nerf_rgb = rendered_image_lst[0][0]

#%%
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(15, 15))
ax1.imshow(nerf_rgb)
ax2.imshow(nerf_depth)
ax3.imshow(blender_rgb)
ax4.imshow(blender_depth)

#%% Composite Blender and NeRF

composite = np.zeros_like(blender_rgb)
path_mask = (blender_depth - nerf_depth.squeeze())<0
nerf_mask = ~path_mask

# Take on path values
composite[path_mask] = blender_rgb[path_mask]
composite[nerf_mask] = nerf_rgb[nerf_mask]

#%% Visualize
fig = plt.figure(1)
plt.imshow(composite)
cv2.imwrite('renders/composite.jpg', cv2.cvtColor((255*composite).astype(np.uint8), cv2.COLOR_RGB2BGR))
# %%
