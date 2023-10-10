#%% 
import numpy as np
import cvxpy as cvx
import json

# Coordinates of vertices of pillars
box_left = np.array([
    [4.14399, 1.48582, 1.],
    [4.14514, 0.814384, 1.],
    [4.80363, 0.814625, 1.],
    [4.80123, 1.48924, 1.]
])

box_right = np.array([
    [4.14399, -0.436273, 1.],
    [4.14514, -1.10771, 1.],
    [4.80363, -1.10747, 1.],
    [4.80123, -0.432849, 1.]
])

box_back = np.array([
    [5.88814, 0.272528, 1.],
    [6.37485, -0.171005, 1.],
    [6.82766, 0.329075, 1.],
    [6.33973, 0.769414, 1.]
])

z_axis = np.array([0., 0., 1.])

# Generate A and B matrices such Ax <= b
A_left = []
B_left = []

A_right = []
B_right = []

A_back = []
B_back = []

for it, pt in enumerate(box_left):
    if it == 3:
        next_pt = box_left[0]
    else:
        next_pt = box_left[it + 1]

    to_next_pt = next_pt - pt
    out_vec = np.cross(to_next_pt, z_axis)

    a = out_vec
    b = np.dot(out_vec, pt)
    
    A_left.append(a)
    B_left.append(b)
A_left = np.stack(A_left, axis=0)
B_left = np.array(B_left)

for it, pt in enumerate(box_right):
    if it == 3:
        next_pt = box_right[0]
    else:
        next_pt = box_right[it + 1]

    to_next_pt = next_pt - pt
    out_vec = np.cross(to_next_pt, z_axis)

    a = out_vec
    b = np.dot(out_vec, pt)
    
    A_right.append(a)
    B_right.append(b)
A_right = np.stack(A_right, axis=0)
B_right = np.array(B_right)

for it, pt in enumerate(box_back):
    if it == 3:
        next_pt = box_back[0]
    else:
        next_pt = box_back[it + 1]

    to_next_pt = next_pt - pt
    out_vec = np.cross(to_next_pt, z_axis)

    a = out_vec
    b = np.dot(out_vec, pt)
    
    A_back.append(a)
    B_back.append(b)
A_back = np.stack(A_back, axis=0)
B_back = np.array(B_back)

#%%
centroid_left = np.mean(box_left, axis=0)
centroid_right = np.mean(box_right, axis=0)
centroid_back = np.mean(box_back, axis=0)

print(((A_left @ centroid_left) <= B_left).all())
print(((A_right @ centroid_right) <= B_right).all())
print(((A_back @ centroid_back) <= B_back).all())

#%% Load trajectories
path_file = './purr_data/path.json'

with open(path_file, 'r') as f:
    meta = json.load(f)

trajectories = meta["traj"]

#%% Formulate cvx program

class ClosestDistance():
    def __init__(self, A1, B1, A2, B2, A3, B3) -> None:
        
        self.A1 = A1
        self.B1 = B1

        self.A2 = A2
        self.B2 = B2

        self.A3 = A3
        self.B3 = B3

        self.x = cvx.Variable(3)
        self.x_bot = cvx.Parameter(3)
        self.A_box = cvx.Parameter((4, 3))
        self.B_box = cvx.Parameter(4)
        objective = cvx.Minimize(cvx.norm(self.x - self.x_bot))
        constraints = [self.A_box @ self.x <= self.B_box]
        self.prob = cvx.Problem(objective, constraints)

    def solve_closest_distance(self, A, B, x):
        self.x_bot.value = x
        self.A_box.value = A
        self.B_box.value = B

        self.prob.solve()
        return self.prob.value

    def solve_closest_to_all(self, x):
        cd1 = self.solve_closest_distance(self.A1, self.B1, x)
        cd2 = self.solve_closest_distance(self.A2, self.B2, x)
        cd3 = self.solve_closest_distance(self.A3, self.B3, x)

        return np.min([cd1, cd2, cd3])

cd = ClosestDistance(A_left, B_left, A_right, B_right, A_back, B_back)

test_trajectory = trajectories[0]
min_dists = []
for point in test_trajectory:
    min_dist = cd.solve_closest_to_all(point)
    min_dists.append(min_dist)

min_dists = np.array(min_dists)
# %%
