import bpy

import sys, os

from mathutils import Matrix, Vector

import math

import numpy as np

import json

from numpy import linalg as la
    
################### FOR OUR METHOD ################################

exp_name = 'stonehenge'
method = 'baseline'
if method == 'baseline':
    data_folder = 'basegrid_data'
    base = bpy.path.abspath('//') + f'{method}/{data_folder}/{exp_name}/path.json'
elif method == 'catnips':
    data_folder = 'catnips_data'
    base = bpy.path.abspath('//') + f'{method}/{data_folder}/{exp_name}/path.json'
elif method =='nerf-nav':
    data_folder = 'baseline_paths'
    base = bpy.path.abspath('//') + f'{method}/{data_folder}/{exp_name}/100.0_iter0/init_costs/19.json'


my_coll = bpy.data.collections.new(f'{exp_name}')
bpy.context.scene.collection.children.link(my_coll)
# material = bpy.data.materials["Material2"]

with open(base, 'r') as f:
    meta = json.load(f)
    
if method == 'nerf-nav':
    locations = [meta["pos"]]
else:
    locations = meta["traj"]

for i, traj in enumerate(locations):
    traj = np.array(traj)[..., :3]

    #FOR AXES
    #start_pt = bpy.data.objects.new(f'start_point{iter}', start_obj.data)
    #start_pt.scale = [0.015, 0.015, 0.015]
    #my_coll.objects.link(start_pt)

    #end_pt = bpy.data.objects.new(f'end_point{iter}', end_obj.data)
    #end_pt.scale = [0.015, 0.015, 0.015]
    #my_coll.objects.link(end_pt)

    start = traj[0, :3]
    end = traj[-1, :3]

    #start_pt.location = [start[0], start[1], start[2]]
    #end_pt.location = [end[0], end[1], end[2]]
    bpy.context.view_layer.update()

    # make a new curve
    crv = bpy.data.curves.new('crv', 'CURVE')
    crv.dimensions = '3D'

    # make a new spline in that curve
    spline = crv.splines.new(type='POLY')

    # a spline point for each point
    spline.points.add(len(traj)-1) # theres already one point by default

    # assign the point coordinates to the spline points
    for p, new_co in zip(spline.points, traj[:, :3]):
        p.co = (new_co.tolist() + [1.0]) # (add nurbs weight)

    # make a new object with the curve
    obj = bpy.data.objects.new(f'traj_{i}', crv)
    obj.data.bevel_depth = 0.005
    # obj.data.materials.append(material)
    my_coll.objects.link(obj)


print("--------------------    DONE WITH BLENDER SCRIPT    --------------------")