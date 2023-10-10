import bpy

import sys, os

from mathutils import Matrix, Vector

import math

import numpy as np

import json

import time

import glob

from numpy import linalg as la
    
################### FOR OUR METHOD ################################
# transforms_path = 'data/nerfstudio/flight_room/transforms.json'

transforms_path = 'data/colmap/statues/transforms.json'

base_path = bpy.path.abspath('//') + transforms_path
data = []

pose_collection = bpy.data.collections.new('Input Image Poses')
bpy.context.scene.collection.children.link(pose_collection)

with open(base_path, 'r') as f:
    meta = json.load(f)
    
#identity_from_frame = np.array([
#        [
#          -0.5808951136484417,
#          0.09293514559271433,
#          -0.8086562656389417,
#          -3.970670786872867
#        ],
#        [
#          -0.8135680962619065,
#          -0.09785434253291189,
#          0.5731775601281526,
#          2.8171808242763197
#        ],
#        [
#          -0.0258622158082496,
#          0.9908519670045847,
#          0.13245199364011948,
#          0.7077808077919308
#        ],
#        [
#          0.0,
#          0.0,
#          0.0,
#          1.0
#        ]
#        ])

#identity_to_frame = np.array([
#        [0., 0., -1., 1],
#        [-1., 0., 0., 0.],
#        [0., 1., 0., 1.67],
#        [0., 0., 0., 1.]
#        ])
        
#identity_from_frame[:3, -1] *= 0.8
        
apply_transform = np.eye(4) # identity_to_frame @ np.linalg.inv(identity_from_frame)
    
frames = meta["frames"]
fl_x = meta["fl_x"]
fl_y = meta["fl_y"]
cx = meta["cx"]
cy = meta["cy"]

K = np.eye(3)
K[0, 0] = fl_x
K[1, 1] = fl_y
K[0, -1] = cx
K[1, -1] = cy

sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

s_u = resolution_x_in_px / sensor_width_in_mm
# TODO include aspect ratio
f_in_mm = K[0,0] / s_u

for iter, frame in enumerate(frames):
    transform = Matrix(frame["transform_matrix"])
    
    # create the first camera
    cam = bpy.data.cameras.new(f"Camera {iter}")

    # create the first camera object
    cam_obj = bpy.data.objects.new(f"Camera {iter}", cam)
    
    cam.lens = f_in_mm 
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width  = sensor_width_in_mm

    cam_obj.matrix_world = Matrix(apply_transform @ np.array(transform))
    
    pose_collection.objects.link(cam_obj)

    bpy.context.view_layer.update()



print("--------------------    DONE WITH BLENDER SCRIPT    --------------------")
