import numpy as np
import dijkstra3d

# TODO: This code has not been fixed yet.

def create_parental_field(field, source):
    field = field.astype(np.uint32)

    #Find shortest path trajectory
    field[ field == 0 ] = field.size + 1        #Ensures that background (unsafe) set to infeasible

    parents = dijkstra3d.parental_field(field, source=source, connectivity=6) # default is 26 connected
    
    return parents

def path_from_parent(parents, target, feasible):
    p_path = dijkstra3d.path_from_parents(parents, target=target)
    p_path = p_path.astype(np.int32)
    p_path = np.flip(p_path, axis=0)

    traj = feasible[p_path[:, 0], p_path[:, 1], p_path[:, 2]]

    return traj, p_path

def euc_to_index(grid, pt, N=100, kernel=None):
    #Grid is N x N x N
    #pt is R3

    lx, ly, lz = (grid[0, 1]-grid[0, 0])/N, (grid[1, 1]-grid[1, 0])/N, (grid[2, 1]-grid[2, 0])/N
    x, y, z = pt

    if kernel is None:
        offset = np.zeros(3).astype(np.uint32)
    else:
        offset = np.floor(np.array(kernel.shape)/2)

    x_index = np.floor((x - grid[0, 0]) / lx - 0.5 - offset[0]).astype(np.uint32)
    y_index = np.floor((y - grid[1, 0]) / ly - 0.5 - offset[1]).astype(np.uint32)
    z_index = np.floor((z - grid[2, 0]) / lz - 0.5 - offset[2]).astype(np.uint32)

    return (x_index, y_index, z_index)

def straight_splits(path, traj):
    assert path.shape[0] == traj.shape[0]

    sorted_path = []
    sorted_traj = []

    path_lengths = []
    running_path = [path[0]]
    running_traj = [traj[0]]
    run_direction = path[1] - running_path[0]
    running_path.append(path[1])
    running_traj.append(traj[1])

    for index, pt in zip(path[2:], traj[2:]):
        # Check if the current point is in the same direction
        current_dir = index - running_path[-1]
        metric = np.sum((current_dir - run_direction)**2)
        if metric > 0:
            # Direction has changed
            sorted_path.append(np.array(running_path))
            sorted_traj.append(np.array(running_traj))
            path_lengths.append(len(running_path))

            running_path = [running_path[-1], index]
            running_traj = [running_traj[-1], pt]
            run_direction = current_dir
        else:
            # Direction has not changed
            running_path.append(index)
            running_traj.append(pt)

    sorted_path.append(np.array(running_path))
    sorted_traj.append(np.array(running_traj))
    path_lengths.append(len(running_path))

    max_path_len = max(path_lengths)

    return sorted_path, sorted_traj, max_path_len