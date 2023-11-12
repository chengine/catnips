import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy
import numpy as np
import torch

### WARNING: This code has not been fixed yet. ###

####### Rectilinear bounds utils #########

def grow_box(occ_pts, minimal, maximal, delta):
    occupied_pts = torch.cat([occ_pts, -occ_pts], dim=-1)

    bounds = minimal.T.reshape(-1)
    bounds = bounds*torch.tensor([1., 1., 1., -1., -1., -1.]).cuda()

    maximal_bounds = maximal.T.reshape(-1)
    maximal_bounds = maximal_bounds * torch.tensor([1., 1., 1., -1., -1., -1.]).cuda()

    side_ind = np.arange(6)

    # Stop loop if no more possible degrees of freedom
    while len(side_ind) > 0:

        next_side_ind = []
        for side in side_ind:
            # Check to see if adding to a particular side will cause collision
            delta_side = delta[side]
            bounds_copy = torch.clone(bounds)
            bounds_copy[side] += delta_side

            is_collide = check_collision(bounds_copy, occupied_pts)
  
            # If occupied pts are in the bounds when we grow it, do
            # not use that dilation and remove it from future growths
            # in that direction.
            if not is_collide and (bounds_copy[side] <= maximal_bounds[side]):
                next_side_ind.append(side)
                bounds[side] += delta_side
        side_ind = next_side_ind

    bounds = bounds*torch.tensor([1., 1., 1., -1., -1., -1.]).cuda()
    return bounds.reshape(-1, 3).T.cpu().numpy()

def check_collision(bounds, pts):
    n = pts.shape[0]
    ul_expanded = bounds[None, :].expand(n, 6)
    coll_per_pt = torch.all(ul_expanded >= pts, dim=-1)

    #If the pt is in the bounds, then the element corresponding to 
    #that pt is 6.
    return torch.any(coll_per_pt)

def get_box_matrices(traj_pts, occ_pts):
    # This is only returning A and b and Q matrices for a single segment of the trajectory

    start = torch.tensor(traj_pts[0]).reshape(1, 3)
    end = torch.tensor(traj_pts[-1]).reshape(1, 3)

    n = occ_pts.shape[0]        # Number of occupied pts

    # Assuming the decision variables are (6): [u, l] (upper and lower limits)
    a = torch.block_diag(-torch.eye(3), torch.eye(3))
    A = [a, a]
    a_ = torch.tensor([[1., 1., 1., 0., 0., 0.], [0., 0., 0., -1., -1., -1.]])
    A = A + n*[a_]
    A = torch.cat(A, dim=0)
    up_low = torch.cat([-torch.eye(3), torch.eye(3)], dim=-1)
    A = torch.cat([A, up_low], dim=0)

    b = torch.cat([-start, -end], dim=0)
    b = torch.cat([b, -b], dim=-1)
    b = b.reshape(-1)

    sum_occ_pts = torch.sum(occ_pts, dim=-1)
    b_ = torch.cat([sum_occ_pts, -sum_occ_pts], dim=-1)
    b_ = b_.reshape(-1)
    
    b = torch.cat([b, b_, torch.zeros(3)])

    return A.cpu().numpy(), b.cpu().numpy()

def get_local_box(cell_sizes, traj, dilation=0.):
    bounds = []
    lx, ly, lz = cell_sizes
    for sub_path in traj:
        bound = min_bounding_box(sub_path, lx, ly, lz, dilation=dilation)
        bounds.append(bound)

    return bounds

def min_bounding_box(start, end, lx, ly, lz, dilation=0.):
    init_cell = start
    end_cell = end

    # Find which direction the segment runs
    dir = end_cell - init_cell
    dir = np.abs(dir/np.linalg.norm(dir)).astype(np.int32)

    cardinal = np.where(dir == 1)   #0 == x, 1 == y, 2 == z
    cardinal = cardinal[0]

    assert len(cardinal) == 1

    if cardinal == 0:
        # segment runs in x direction
        zbound = [init_cell[-1] + lz/2 + dilation*lz, init_cell[-1] - lz/2 - dilation*lz,]
        ybound = [init_cell[1] + ly/2 + dilation*ly, init_cell[1] - ly/2 - dilation*ly,]
        
        # Find whether init or end is bigger
        xlims = np.sort(np.array([init_cell[0], end_cell[0]]))
        xbound = [xlims[-1] + lx/2 + dilation*lx, xlims[0] - lx/2 - dilation*lx,]

    elif cardinal == 1:
        # segment runs in y direction
        zbound = [init_cell[-1] + lz/2 + dilation*lz, init_cell[-1] - lz/2 - dilation*lz,]
        xbound = [init_cell[0] + lx/2 + dilation*lx, init_cell[0] - lx/2 - dilation*lx,]
        
        # Find whether init or end is bigger
        ylims = np.sort(np.array([init_cell[1], end_cell[1]]))
        ybound = [ylims[-1] + ly/2 + dilation*ly, ylims[0] - ly/2 - dilation*ly,]

    elif cardinal == 2:
        # segment runs in z direction
        ybound = [init_cell[1] + ly/2 + dilation*ly, init_cell[1] - ly/2 - dilation*ly,]
        xbound = [init_cell[0] + lx/2 + dilation*lx, init_cell[0] - lx/2 - dilation*lx,]
        
        # Find whether init or end is bigger
        zlims = np.sort(np.array([init_cell[-1], end_cell[-1]]))
        zbound = [zlims[-1] + lz/2 + dilation*lz, zlims[0] - lz/2 - dilation*lz,]

    else:
        raise('Something terribly wrong')

    bound = np.array([xbound, ybound, zbound])

    return bound

def refine_bounds(bounds):
    # Idea is to reduce the overlap between consecutive bounding boxes to 
    # remove kinks in the trajectory. We do this by only taking the furthest box
    # that still overlaps with the current one and ignore the bounding boxes in
    # between.

    current_ind = 0
    current_bounds = bounds[current_ind]
    max_ind = len(bounds)

    new_bounds = [bounds[0]]
    while current_ind < max_ind-1:
        current_ind += 1

        # Get the next bounding box
        bound = bounds[current_ind]

        is_collide = box_collision(bound, current_bounds)
        if not is_collide:
            # If the vertices are not in the bounding box, 
            # we add the testing bounding box to new_bounds
            current_ind -= 1

            current_bounds = bounds[current_ind]
            new_bounds.append(bounds[current_ind])

    new_bounds.append(bounds[-1])

    return new_bounds

def isOverlapping1D(box1, box2):
    max1, min1 = box1
    max2, min2 = box2
    return (max1 >= min2) and (max2 >= min1)

def box_collision(bounds1, bounds2):
    box1x = bounds1[0,:]
    box1y = bounds1[1,:]
    box1z = bounds1[2,:]

    box2x = bounds2[0,:]
    box2y = bounds2[1,:]
    box2z = bounds2[2,:]

    return isOverlapping1D(box1x, box2x) and isOverlapping1D(box1y, box2y) and isOverlapping1D(box1z, box2z)

### PLOTTING BOUNDS UTILS ###

def plot_bounds(bound, ax):
    upper = bound[:, 0]
    lower = bound[:, -1]

    points = np.array([
        [lower[0], lower[1], lower[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]],
        [lower[0], upper[1], upper[2]]
    ])

    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    verts = [[points[0], points[1], points[2], points[3]],
    [points[4], points[5], points[6], points[7]],
    [points[0], points[1], points[5], points[4]],
    [points[2], points[3], points[7], points[6]],
    [points[1], points[2], points[6], points[5]],
    [points[4], points[7], points[3], points[0]]]

    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', edgecolors='red', linewidths=1, alpha=.1))

def bounds2vertices(bound):
    upper = bound[:, 0]
    lower = bound[:, -1]

    points = torch.tensor([
        [lower[0], lower[1], lower[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]],
        [lower[0], upper[1], upper[2]]
    ])

    return points

def bound2mesh(bounds, filename, transform=None, scale=None):
    vox_mesh=o3d.geometry.TriangleMesh()
    for bound in bounds:
        lx, ly, lz = np.abs(bound[0, 0]-bound[0, 1]), \
                    np.abs(bound[1, 0]-bound[1, 1]), \
                    np.abs(bound[2, 0]-bound[2, 1])

        center = (bound[:, 0] + bound[:, 1])/2
        cube=o3d.geometry.TriangleMesh.create_box(width=lx, height=ly,
        depth=lz)
        #cube.paint_uniform_color(v.color)
        cube.translate(center, relative=False)
        vox_mesh+=cube

    if scale is not None:
        vox_mesh.scale(1/scale, center=np.array([0, 0, 0]))

    if transform is not None:
        vox_mesh.transform(transform)

    #vox_mesh.translate([lx,ly,lz], relative=True)
    vox_mesh.merge_close_vertices(1e-6)
    o3d.io.write_triangle_mesh(filename, vox_mesh)