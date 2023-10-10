import numpy as np
import dijkstra3d

# Perform coarse path planning
def astar3D(field, source, target, feasible):

    #Performs A* 
    inds = dijkstra3d.binary_dijkstra(field, source, target, connectivity=6, background_color=1)
    inds = inds.astype(np.int32)

    print(f'Returning path of length {len(inds)}')

    traj = feasible[inds[:, 0], inds[:, 1], inds[:, 2]]

    return traj, inds

def chunk_path(path, num_sec):
    # Path: N x 3
    # num_sec: number of sections to split path into

    # Returns list of subpaths [sp1, sp2,...]

    N_path = len(path)-1

    path_no_end = path[:-1]
    path_no_start = path[1:]

    new_path = []

    # 3 cases
    if N_path == num_sec:
        # No need to do anything. 
        for (start, end) in zip(path_no_end, path_no_start):
            sub_path = np.concatenate([start, end], axis=0)

            new_path.append(sub_path.reshape(-1, 3))
        
    elif N_path < num_sec:
        # Need to interpolate points to match num_sections
        indices = np.arange(num_sec)
        split_indices = np.array_split(indices, N_path)

        for (start, end, sample_ind) in zip(path_no_end, path_no_start, split_indices):
            t = np.linspace(0., 1., len(sample_ind), endpoint=True)
            sub_path = np.reshape(start, (1, 3)) + t[:, None] * np.reshape(end - start, (1, 3))

            new_path.append(sub_path.reshape(-1, 3))
        
    elif N_path > num_sec:
        # Undesirable condition, as the input path is already the shortest it can be. Raise an error asking
        # to bump up the number of sections. Return the path.

        for (start, end) in zip(path_no_end, path_no_start):
            sub_path = np.concatenate([start, end], axis=0)

            new_path.append(sub_path.reshape(-1, 3))

        print('Requested number of sections is less than the fewest amount of sections. Returning the path with fewest sections.')
    
    else:
        raise ValueError('Chunk Path function has error.')

    return new_path


class PathInit():
    # Uses A* as path initialization
    def __init__(self, grid_occupied, grid_points) -> None:
        self.grid_occupied = grid_occupied      # Binary Field (Nx x Ny x Nz)
        self.grid_points = grid_points          # Points corresponding to the field (Nx x Ny x Nz)

        self.cell_sizes = self.grid_points[1, 1, 1] - self.grid_points[0, 0, 0]

    def create_path(self, x0, xf, num_sec=10):
        source = self.get_indices(x0)   # Find nearest grid point and find its index
        target = self.get_indices(xf)

        path3d, indices = astar3D(self.grid_occupied, source, target, self.grid_points)

        try:
            assert len(path3d) > 0
        except:
            raise AssertionError('Could not find a feasible initialize path. Please change the initial/final positions to not be in collision.')

        path = self.fewest_straight_lines(path3d)

        path = chunk_path(path, num_sec)

        return path

    def get_indices(self, point):
        min_bound = self.grid_points[0, 0, 0] - self.cell_sizes/2

        lx, ly, lz = self.cell_sizes
        transformed_pt = point - min_bound

        indices = np.array([transformed_pt[0] / lx, 
                            transformed_pt[1]/ ly, 
                            transformed_pt[2]/lz])

        # If querying points outside of the bounds, project to the nearest side
        for i, ind in enumerate(indices):
            if ind < 0.:
                indices[i] = 0

                print('Point is outside of minimum bounds. Projecting to nearest side. This may cause unintended behavior.')

            elif ind > self.grid_occupied.shape[i]:
                indices[i] = self.grid_occupied.shape[i]

                print('Point is outside of maximum bounds. Projecting to nearest side. This may cause unintended behavior.')

        indices = indices.astype(np.uint32)

        return indices


    # Process path so you have fewest number of straight line paths as possible
    def fewest_straight_lines(self, path, num_test_pts=100):
        root_index = 0
        candidate_index = 1

        root = path[root_index]
        candidate = path[candidate_index]
        direction = (candidate - root) / np.linalg.norm(candidate-root, keepdims=True)

        new_path = [root]
        while True:
            candidate_index += 1
            candidate = path[candidate_index]

            new_direction = (candidate - root) / np.linalg.norm(candidate-root, keepdims=True)

            if np.linalg.norm(new_direction - direction) < 1e-2:
                # Going in a straight line, proceed to next iteration
                continue

            direction = new_direction

            # Sample from the root to new_candidate 
            t = np.linspace(0., 1., num_test_pts, endpoint=False)
            test_points = np.reshape(root, (1, 3)) + t[..., None] * np.reshape(candidate - root, (1, 3))

            test_pts_is_collide = []
            for test_pt in test_points:
                index = self.get_indices(test_pt)
                test_pts_is_collide.append(self.grid_occupied[index[..., 0], index[..., 1], index[..., 2]])

            if np.any(np.array(test_pts_is_collide)):
                # Some point is in collision, then we make the previous candidate the root
                root = path[candidate_index - 1]
                candidate_index -= 1

                new_path.append(root)

                # Edge case: If second to last pt is clear but last pt is not.
                if candidate_index == len(path) - 2:
                    new_path.append(path[-1])
                    break

            # Termination condition if last segment is clear
            if candidate_index == len(path) - 1:
                new_path.append(path[-1])
                break

        return np.stack(new_path, axis=0)