import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import poisson
# import unfoldNd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def maxpool3d(input, kernel):
#     # Kernel is binary mask
#     # Pad tensor

#     pad = (kernel.shape[0]//2, kernel.shape[1]//2, kernel.shape[2]//2)
#     lib_module = unfoldNd.UnfoldNd(kernel.shape, padding=pad)

#     inp_unf = lib_module(input)
#     #inp_unf = torch.nn.functional.unfold(input, kernel.shape, padding=pad)
#     inp_unf = inp_unf.transpose(1, 2)
#     inp_unf = inp_unf.view((1, -1, kernel.shape[0], kernel.shape[1], kernel.shape[2]))
#     inp_unf = inp_unf.permute(2, 3, 4, 0, 1)[kernel].squeeze()
#     inp_unf = inp_unf.transpose(0, 1).view((input.shape[2], input.shape[3], input.shape[4], -1))
#     inp_unf = torch.max(inp_unf, dim=-1)[0]

#     return inp_unf

# --------------------------------------------------------------------------------#
### Utiliy function to make components of the PURR and solve for the trajectory ###
def generate_kernel(robot_lims, grid, N):
    # Functions find the voxelized overapproximation of the Minkowski sum 

    lx, ly, lz = (grid[0, 1]-grid[0, 0])/N, (grid[1, 1]-grid[1, 0])/N, (grid[2, 1]-grid[2, 0])/N
    #How many cells does the robot body take up
    x_extent = robot_lims[0, 1] + robot_lims[0, 0]
    y_extent = robot_lims[1, 1] + robot_lims[1, 0]
    z_extent = robot_lims[2, 1] + robot_lims[2, 0]

    # Assuming robot body is a bounding box
    center = np.array([x_extent, y_extent, z_extent])/2

    #Radius of sphere bounding the bounding box
    r = np.linalg.norm(np.array([robot_lims[0, 1], robot_lims[1, 1], robot_lims[2, 1]]) - center)

    # Perform Minkowski Sum of sphere over a voxel
    # Determine maximum size of kernel in voxels
    x_max = 2*np.ceil(r/lx) + 1
    y_max = 2*np.ceil(r/ly) + 1
    z_max = 2*np.ceil(r/lz) + 1

    kernel = np.ones((x_max.astype(np.uint32), y_max.astype(np.uint32), z_max.astype(np.uint32)))

    # Need to figure out how much to shave off
    # Method: Go through the brick encompassing one quadrant of the sphere centered at one of the
    # corners of the voxel. We use the method that if any corner of a voxel is in the sphere, then reject.
    test_grid = np.array([[-lx*np.ceil(r/lx), lx],
    [-ly*np.ceil(r/ly), ly],
    [-lz*np.ceil(r/lz), lz]])

    num_x, num_y, num_z = (np.ceil(r/lx) + 2).astype(np.uint32), (np.ceil(r/ly) + 2).astype(np.uint32), (np.ceil(r/lz) + 2).astype(np.uint32)

    x, y, z = np.meshgrid(np.linspace(test_grid[0, 0], test_grid[0, 1], num_x, endpoint=True), 
        np.linspace(test_grid[1, 0], test_grid[1, 1], num_y, endpoint=True), 
        np.linspace(test_grid[2, 0], test_grid[2, 1], num_z, endpoint=True))
    pts = np.stack([x, y, z], axis=-1)

    mask = np.sum(pts**2, axis=-1) > r**2       #mask will have 1 if outside of sphere

    # Splitting based on vertices (x, y, z) in ([0 1], [0 1], [0 1])
    x0y0z0 = mask[:-1, :-1, :-1]   #[0, 0, 0]
    x1y0z0 = mask[1:, :-1, :-1]    #[1, 0, 0]
    x0y1z0 = mask[:-1, 1:, :-1]    #[0, 1, 0]
    x0y0z1 = mask[:-1, :-1, 1:]    #[0, 0, 1]
    x1y1z0 = mask[1:, 1:, :-1]     #[1, 1, 0]
    x0y1z1 = mask[:-1, 1:, 1:]     #[0, 1, 1]
    x1y0z1 = mask[1:, :-1, 1:]     #[1, 0, 1]
    x1y1z1 = mask[1:, 1:, 1:]      #[1, 1, 1]

    partial_mask = np.stack([x0y0z0, x1y0z0, x0y1z0, x0y0z1, x1y1z0, x0y1z1, x1y0z1, x1y1z1], axis=-1)
    partial_mask = np.prod(partial_mask, axis=-1) < 1     # If 1, this means at least one vertex in sphere

    mask_x_flip = partial_mask[::-1, :, :]
    mask_y_flip = partial_mask[:, ::-1, :]
    mask_z_flip = partial_mask[:, :, ::-1]
    mask_xy_flip = mask_x_flip[:, ::-1, :]
    mask_yz_flip = mask_y_flip[:, :, ::-1]
    mask_xz_flip = mask_x_flip[:, :, ::-1]
    mask_xyz_flip = mask_xy_flip[:, :, ::-1]

    along_y_1 = np.concatenate([partial_mask, mask_y_flip[:, 1:, :]], axis=1)
    along_y_2 = np.concatenate([mask_x_flip, mask_xy_flip[:, 1:, :]], axis=1)
    along_y_3 = np.concatenate([mask_z_flip, mask_yz_flip[:, 1:, :]], axis=1)
    along_y_4 = np.concatenate([mask_xz_flip, mask_xyz_flip[:, 1:, :]], axis=1)

    along_x_1 = np.concatenate([along_y_1, along_y_2[1:, :, :]], axis=0)
    along_x_2 = np.concatenate([along_y_3, along_y_4[1:, :, :]], axis=0)

    mask = np.concatenate([along_x_1, along_x_2[:, :, 1:]], axis=-1)
    mask = np.transpose(mask, (1, 0, 2))

    kernel = mask*kernel

    # kernel = (2*np.ceil(x_extent/(2*lx) - 1).astype(np.uint32) + 1, 2*np.ceil(y_extent/(2*ly) - 1).astype(np.uint32) + 1, 2*np.ceil(z_extent/(2*lz) - 1).astype(np.uint32) + 1)

    return torch.tensor(kernel, device=device, dtype=torch.float32)

# Generates Probabilistically Unsafe Robot Regions
def generate_purr(grid, kernel, get_density, discretization=100, density_factor=1., sigma=0.95, Aaux=1e-8, dt=1e-2, Vmax=1e-5, gamma=1.):

    with torch.no_grad():
        # Querying density for grid vertices
        x, y, z = torch.meshgrid(torch.linspace(grid[0, 0], grid[0, 1], discretization, device=device), 
        torch.linspace(grid[1, 0], grid[1, 1], discretization, device=device), 
        torch.linspace(grid[2, 0], grid[2, 1], discretization, device=device))
        pts = torch.stack([x, y, z], axis=-1)
        query_pts = pts.view((-1, 3))

        density = density_factor*get_density(query_pts)
        density = density.view(pts.shape[:-1])

    #Find the coefficients for the interpolation
    lx, ly, lz = (grid[0, 1]-grid[0, 0])/discretization, (grid[1, 1]-grid[1, 0])/discretization, (grid[2, 1]-grid[2, 0])/discretization

    vertices = torch.tensor([
        [0., 0., 0.],
        [lx, 0., 0.],
        [0., ly, 0.],
        [0., 0., lz],
        [lx, ly, 0.],
        [0., ly, lz],
        [lx, 0., lz],
        [lx, ly, lz]
    ], device=device, dtype=torch.float32)

    # Evaluating A @ b = rho (b = coefficients of interpolation, A = monomials, rho = density at vertex)
    features = torch.stack([torch.ones(8, device=device), vertices[:, 0], vertices[:, 1], vertices[:, 2], 
    vertices[:, 0]*vertices[:, 1], vertices[:, 1]*vertices[:, 2], vertices[:, 0]*vertices[:, 2],
    vertices[:, 0]*vertices[:, 1]*vertices[:, 2]], dim=-1)

    inv_features = torch.inverse(features)

    # Splitting based on vertices (x, y, z) in ([0 1], [0 1], [0 1])
    x0y0z0 = density[:-1, :-1, :-1]   #[0, 0, 0]
    x1y0z0 = density[1:, :-1, :-1]    #[1, 0, 0]
    x0y1z0 = density[:-1, 1:, :-1]    #[0, 1, 0]
    x0y0z1 = density[:-1, :-1, 1:]    #[0, 0, 1]
    x1y1z0 = density[1:, 1:, :-1]     #[1, 1, 0]
    x0y1z1 = density[:-1, 1:, 1:]     #[0, 1, 1]
    x1y0z1 = density[1:, :-1, 1:]     #[1, 0, 1]
    x1y1z1 = density[1:, 1:, 1:]      #[1, 1, 1]

    densities = torch.stack([x0y0z0, x1y0z0, x0y1z0, x0y0z1, x1y1z0, x0y1z1, x1y0z1, x1y1z1], dim=0)
    coeffs = inv_features @ densities.reshape((8, -1))  #Coefficients (8) for each cell

    feats = torch.tensor([lx*ly*lz, lx**2 * ly*lz, ly**2 *lx*lz, 
                        lz**2 * lx*ly, lx**2 * ly**2 * lz, ly**2 * lz**2 * lx, 
                        lx**2 * lz**2 * ly, (lx*ly*lz)**2], 
                        device=device, dtype=torch.float32)

    intensity = feats.reshape((1, 8))@coeffs
    intensity = intensity.reshape(densities.shape[1:])
    
    #The size of each dimension depends on the relative size between the resolution of the density field and the agent
    robo_intensity = F.conv3d(intensity[None, None, ...], kernel[None, None, ...]).squeeze().cpu().numpy()

    Naux = Vmax/(Aaux*dt)       # Number of auxiliary particles
    lambda_scale = gamma/(Aaux)     # multiplication factor to rho to get lambda

    robo_intensity_flat = lambda_scale * robo_intensity.reshape(-1)
    max_num_particles = Naux*np.ones_like(robo_intensity_flat)

    cdf = poisson.cdf(max_num_particles, robo_intensity_flat).reshape(robo_intensity.shape)
    safe_zone = (cdf >= sigma)

    # Create center points of the safe zone
    center_pts = pts[:-1, :-1, :-1] + (torch.tensor([lx, ly, lz], device=device) / 2)[None, None, None, :]    # Assuming grid is ordered from smallest to largest
    x_off, y_off, z_off = kernel.shape[0]//2, kernel.shape[1]//2, kernel.shape[2]//2
    trunc_center_pts = center_pts[x_off:-x_off, y_off:-y_off, z_off:-z_off]

    return safe_zone, trunc_center_pts.cpu().numpy(), cdf

# Generates Probabilistic Unoccupied Grid
def generate_pug(grid, get_density, discretization=100, density_factor=1., sigma=0.95, Aaux=1e-8, dt=1e-2, V_percent=0.05, gamma=1.):

    with torch.no_grad():
        # Querying density for grid vertices
        x, y, z = torch.meshgrid(torch.linspace(grid[0, 0], grid[0, 1], discretization, device=device), 
        torch.linspace(grid[1, 0], grid[1, 1], discretization, device=device), 
        torch.linspace(grid[2, 0], grid[2, 1], discretization, device=device))
        pts = torch.stack([x, y, z], axis=-1)
        query_pts = pts.view((-1, 3))

        density = density_factor*get_density(query_pts)
        density = density.view(pts.shape[:-1])

    #Find the coefficients for the interpolation
    lx, ly, lz = (grid[0, 1]-grid[0, 0])/discretization, (grid[1, 1]-grid[1, 0])/discretization, (grid[2, 1]-grid[2, 0])/discretization

    vertices = torch.tensor([
        [0., 0., 0.],
        [lx, 0., 0.],
        [0., ly, 0.],
        [0., 0., lz],
        [lx, ly, 0.],
        [0., ly, lz],
        [lx, 0., lz],
        [lx, ly, lz]
    ], device=device, dtype=torch.float32)

    # Evaluating A @ b = rho (b = coefficients of interpolation, A = monomials, rho = density at vertex)
    features = torch.stack([torch.ones(8, device=device), vertices[:, 0], vertices[:, 1], vertices[:, 2], 
    vertices[:, 0]*vertices[:, 1], vertices[:, 1]*vertices[:, 2], vertices[:, 0]*vertices[:, 2],
    vertices[:, 0]*vertices[:, 1]*vertices[:, 2]], dim=-1)

    inv_features = torch.inverse(features)

    # Splitting based on vertices (x, y, z) in ([0 1], [0 1], [0 1])
    x0y0z0 = density[:-1, :-1, :-1]   #[0, 0, 0]
    x1y0z0 = density[1:, :-1, :-1]    #[1, 0, 0]
    x0y1z0 = density[:-1, 1:, :-1]    #[0, 1, 0]
    x0y0z1 = density[:-1, :-1, 1:]    #[0, 0, 1]
    x1y1z0 = density[1:, 1:, :-1]     #[1, 1, 0]
    x0y1z1 = density[:-1, 1:, 1:]     #[0, 1, 1]
    x1y0z1 = density[1:, :-1, 1:]     #[1, 0, 1]
    x1y1z1 = density[1:, 1:, 1:]      #[1, 1, 1]

    densities = torch.stack([x0y0z0, x1y0z0, x0y1z0, x0y0z1, x1y1z0, x0y1z1, x1y0z1, x1y1z1], dim=0)
    coeffs = inv_features @ densities.reshape((8, -1))  #Coefficients (8) for each cell

    feats = torch.tensor([lx*ly*lz, lx**2 * ly*lz, ly**2 *lx*lz, 
                        lz**2 * lx*ly, lx**2 * ly**2 * lz, ly**2 * lz**2 * lx, 
                        lx**2 * lz**2 * ly, (lx*ly*lz)**2], 
                        device=device, dtype=torch.float32)

    intensity = feats.reshape((1, 8))@coeffs
    intensity = intensity.reshape(densities.shape[1:]).squeeze().cpu().numpy()

    Vmax = lx*ly*lz*V_percent
    Naux = Vmax/(Aaux*dt)       # Number of auxiliary particles
    lambda_scale = gamma/(Aaux)     # multiplication factor to rho to get lambda

    intensity_flat = lambda_scale * intensity.reshape(-1)
    max_num_particles = Naux*np.ones_like(intensity_flat)

    cdf = poisson.cdf(max_num_particles, intensity_flat).reshape(intensity.shape)
    pug = (cdf >= sigma)

    # Create center points of the safe zone
    center_pts = pts[:-1, :-1, :-1] + (torch.tensor([lx, ly, lz], device=device) / 2)[None, None, None, :]    # Assuming grid is ordered from smallest to largest

    return pug, center_pts.cpu().numpy()

def centers_to_vertices(points, lx, ly, lz):
    # points: N x 3

    vertices = np.array([
        [-lx/2, -ly/2, -lz/2],
        [lx/2, -ly/2, -lz/2],
        [-lx/2, ly/2, -lz/2],
        [-lx/2, -ly/2, lz/2],
        [lx/2, ly/2, -lz/2],
        [lx/2, -ly/2, lz/2],
        [-lx/2, ly/2, lz/2],
        [lx/2, ly/2, lz/2]
    ])

    points_ = points[..., None]     # N x 3 x 1
    vertices_ = (vertices.T)[None,...]

    possible_vertices = points_ + vertices_

    return np.unique(possible_vertices, axis=0)