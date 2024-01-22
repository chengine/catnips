import os
import numpy as np
import json
import time
from .purr_utils import *
import open3d as o3d

### CATNIPS class ###

class Catnips():
    def __init__(self, configs=None, load_path=None) -> None:
        self.configs = configs
        self.load_path = load_path

    def load_purr(self):
        if self.load_path is not None:
            # Load purr data
            with open(self.load_path + 'purr.npy', 'rb') as f:
                self.field = np.load(f)

            with open(self.load_path + 'conv_centers.npy', 'rb') as f:
                self.feasible = np.load(f)

            with open(self.load_path + 'kernel.npy', 'rb') as f:
                self.kernel = np.load(f)

            with open(self.load_path + 'meta.json', 'r') as f:
                meta = json.load(f)

            self.grid = np.array(meta['grid'])
            self.agent_body = np.array(meta['agent body'])
            self.discretization = meta['discretization']
            self.sigma = meta['sigma']
            self.Aaux = meta['Aaux']
            self.dt = meta['dt']
            self.Vmax = meta['Vmax']
            self.gamma = meta['gamma']

            self.cell_sizes = np.array(meta['cell sizes'])

        elif self.configs is not None:
            self.grid = self.configs["grid"]
            self.agent_body = self.configs["agent_body"]

            self.discretization = self.configs["discretization"]
            self.sigma = self.configs["sigma"]
            self.Aaux = self.configs['Aaux']
            self.dt = self.configs['dt']
            self.Vmax = self.configs['Vmax']
            self.gamma = self.configs['gamma']

            self.density_factor = self.configs["density_factor"]
            self.get_density = self.configs["get_density"]

        
        else:
            raise ValueError('Did not provide path to PURR or configs.')

        self.cell_sizes = [(self.grid[0, 1]-self.grid[0, 0])/self.discretization, 
                (self.grid[1, 1]-self.grid[1, 0])/self.discretization, 
                (self.grid[2, 1]-self.grid[2, 0])/self.discretization]

        return 
    
    def create_purr(self):
        ### Initialization: Create the robot kernel and the PURR
        # Create kernel that is upper bound of robot geometry
        self.kernel = generate_kernel(self.agent_body, self.grid, self.discretization)

        self.purr, self.conv_centers, self.cdf = generate_purr(self.grid, self.kernel, self.get_density,
                                                    discretization=self.discretization, 
                                                    density_factor=self.density_factor,
                                                    sigma=self.sigma, Aaux=self.Aaux, dt=self.dt, 
                                                    Vmax=self.Vmax, gamma=self.gamma)   
        
        self.purr_vertices = centers_to_vertices(self.conv_centers[self.purr == False], self.cell_sizes[0], self.cell_sizes[1], self.cell_sizes[2])

    def save_purr(self, filename, transform=None, scale=None, save_property=False):
        if not os.path.exists(filename):
            # Create a new directory because it does not exist
            os.makedirs(filename)

        # NOTE!: transform and scale are transformations and scaling from data frame to nerf frame, so 
        # must undo these operations here. 

        # Generate voxel mesh
        lx, ly, lz = self.cell_sizes
        vox_mesh = o3d.geometry.TriangleMesh()
        
        collision = self.conv_centers[self.purr == False]
        for coor in collision:
            cube=o3d.geometry.TriangleMesh.create_box(width=lx, height=ly,
            depth=lz)

            cube.translate(coor, relative=False)

            vox_mesh+=cube

        if scale is not None:
            vox_mesh.scale(1/scale, center=np.array([0, 0, 0]))

        if transform is not None:
            vox_mesh.transform(np.linalg.inv(transform))

        vox_mesh.merge_close_vertices(1e-6)
        o3d.io.write_triangle_mesh(filename + 'purr.ply', vox_mesh, print_progress=True)

        if save_property:
            # Save purr data
            np.save(filename + 'purr.npy', self.purr)
            np.save(filename + 'conv_centers.npy', self.conv_centers)
            np.save(filename + 'kernel.npy', self.kernel.cpu().numpy())

            purr_meta = {
                'grid': self.grid.tolist(),
                'agent body': self.agent_body.tolist(),
                'discretization': self.discretization,
                'sigma': self.sigma,
                'cell sizes': [lx, ly, lz]
            }

            with open(filename + 'meta.json', 'w') as f:
                json.dump(purr_meta, f, indent=4)

    # def create_pug(self, V_percent, sigma):
    #     self.pug, self.centers = generate_pug(
    #         self.grid, self.get_density, discretization=self.discretization, density_factor=self.density_factor,
    #         sigma=sigma, Aaux=self.Aaux, dt=self.dt, 
    #         V_percent=V_percent, gamma=self.gamma) 
        
    # def save_pug(self, filename, transform=None, scale=None):
    #     if not os.path.exists(filename):
    #         # Create a new directory because it does not exist
    #         os.makedirs(filename)

    #     # Generate voxel mesh
    #     lx, ly, lz = self.cell_sizes
    #     vox_mesh = o3d.geometry.TriangleMesh()
        
    #     collision = self.centers[self.pug == False]
    #     for coor in collision:
    #         cube=o3d.geometry.TriangleMesh.create_box(width=lx, height=ly,
    #         depth=lz)

    #         cube.translate(coor, relative=False)

    #         vox_mesh+=cube

    #     if scale is not None:
    #         vox_mesh.scale(1/scale, center=np.array([0, 0, 0]))

    #     if transform is not None:
    #         vox_mesh.transform(np.linalg.inv(transform))

    #     vox_mesh.merge_close_vertices(1e-6)
    #     o3d.io.write_triangle_mesh(filename + 'pug.ply', vox_mesh, print_progress=True)

    def viz_point_cloud(self, num_pts=100000):

        coll_pts = self.conv_centers[self.purr == False]
        coll_pts = coll_pts.reshape(-1, 3)

        cdf_pts = self.cdf[self.purr == False]  # Probability that occupied space is less than some threshold
        cdf_pts = 1. - cdf_pts      # More unsafe points have a lower cdf
        cdf_pts = cdf_pts.reshape(-1)

        rand_offsets = (np.random.rand(num_pts, 3) - 0.5) * np.array(self.cell_sizes)[None,...]

        # Choice cell indices based on their cdf values
        center_indices = np.random.choice(np.arange(len(coll_pts)), size=num_pts, replace=True, p=cdf_pts)

        pc_centers = coll_pts[center_indices] + rand_offsets

        return pc_centers
    
    def viz_slices(self, height, axis, ax):
        min_bound = self.conv_centers[0, 0, 0] - self.cell_sizes/2

        if axis=='x':
            i = 0
        elif axis=='y':
            i = 1

        elif axis=='z':
            i = 2
        else:
            raise ValueError('Not a valid axis')

        transformed_pt = height - min_bound[i]

        indices = transformed_pt / self.cell_sizes[i]

        return_indices = indices.copy()
        # If querying points outside of the bounds, project to the nearest side
        if return_indices < 0:
            return_indices = 0

            print('Point is outside of minimum bounds. Projecting to nearest side. This may cause unintended behavior.')

        elif return_indices > self.grid_occupied.shape[i]:
            return_indices = self.grid_occupied.shape[i]

            print('Point is outside of maximum bounds. Projecting to nearest side. This may cause unintended behavior.')

        return_indices = return_indices.astype(np.uint32)

        cdf_pts = self.cdf[self.purr == False]  # Probability that occupied space is less than some threshold
        cdf_pts = 1. - cdf_pts      # More unsafe points have a lower cdf

        if axis=='x':
            cdf_slice = cdf_pts[return_indices, :, :]

        elif axis=='y':
            cdf_slice = cdf_pts[:, return_indices, :]

        elif axis=='z':
            cdf_slice = cdf_pts[:, :, return_indices]

        ax.imshow(cdf_slice)

        return ax