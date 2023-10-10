import numpy as np
import json
import time
import purr_utils
import open3d as o3d
### CATNIPS class ###

class CATNIPS():
    def __init__(self, configs, get_density) -> None:
        
        self.get_density = get_density

        # data_path=None, grid=None, agent_body=None

        # if data_path is not None: 
        #     # Load purr data
        #     with open(data_path + '_field.npy', 'rb') as f:
        #         self.field = np.load(f)

        #     with open(data_path + '_points.npy', 'rb') as f:
        #         self.feasible = np.load(f)

        #     with open(data_path + '_kernel.npy', 'rb') as f:
        #         self.kernel = np.load(f)

        #     with open(data_path + '.json', 'r') as f:
        #         meta = json.load(f)


        #     self.grid = np.array(meta['grid'])
        #     self.agent_body = np.array(meta['agent body'])
        #     self.discretization = meta['discretization']
        #     self.sigma = meta['sigma']
        #     self.cell_sizes = np.array(meta['cell sizes'])

        # else:

        self.grid = configs["grid"]
        self.agent_body = configs["agent_body"]

        self.discretization = configs["discretization"]
        self.sigma = configs["sigma"]
        self.density_factor = configs["density_factor"]

        ### Initialization: Create the robot kernel and the PURR
        # Create kernel that is upper bound of robot geometry
        tnow = time.time()
        self.kernel = make_kernel(self.agent_body, self.grid, self.discretization)

        self.field, self.feasible = generate_avoid_zone(self.grid, self.kernel, self.get_density,
                                                    discretization=self.discretization, 
                                                    density_factor=self.density_factor,
                                                    sigma=self.sigma)   

        self.cell_sizes = [(self.grid[0, 1]-self.grid[0, 0])/self.discretization, 
                        (self.grid[1, 1]-self.grid[1, 0])/self.discretization, 
                        (self.grid[2, 1]-self.grid[2, 0])/self.discretization]

        print(f'Time to generate PURR: {time.time() - tnow}')
        
    def save_purr(self, filename, transform=None, scale=None):
        # Generate voxel mesh
        lx, ly, lz = self.cell_sizes
        vox_mesh=o3d.geometry.TriangleMesh()
        
        collision = self.feasible[self.field == False]
        for coor in collision:
            cube=o3d.geometry.TriangleMesh.create_box(width=lx, height=ly,
            depth=lz)

            vox_mesh+=cube

        if scale is not None:
            vox_mesh.scale(1/scale, center=np.array([0, 0, 0]))

        if transform is not None:
            vox_mesh.transform(transform)

        vox_mesh.merge_close_vertices(1e-6)
        o3d.io.write_triangle_mesh(filename + '.ply', vox_mesh)

        # # Save purr data
        # np.save(filename + '_field.npy', self.field)
        # np.save(filename + '_points.npy', self.feasible)
        # np.save(filename + '_kernel.npy', self.kernel.cpu().numpy())

        # purr_meta = {
        #     'grid': self.grid.tolist(),
        #     'agent body': self.agent_body.tolist(),
        #     'discretization': self.discretization,
        #     'sigma': self.sigma,
        #     'cell sizes': [lx, ly, lz]
        # }

        # with open(filename + '.json', 'w') as f:
        #     json.dump(purr_meta, f, indent=4)