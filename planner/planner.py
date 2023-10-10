import time
        


def class Planner():
    def __init__(self) -> None:
        pass        
        
        self.spline_deg = configs['spline_deg']

        self.x0 = position_configs['start'][:3]
        self.xf = position_configs['end'][:3]


        ### Path Initialization: Create A* parental field to the final position xf.
        # NOTE: You have the option to have moving final positions, in which case
        # the parental field is useless, but we create one at initialization anyways.
        # Use method .get_traj and pass in an xf

        #TODO: Check if the trajectory is flipped
        tnow = time.time()
        source = euc_to_index(self.grid, self.xf[:3], N=self.discretization, kernel=self.kernel)
        self.parental_field = create_parental_field(self.field, source)
        print(f'Parental Field Time: {time.time() - tnow}')
    
        ### Create the time points matrix/coefficients for the Bezier curve
        self.derivT = create_time_pts(deg=self.spline_deg)

    def get_traj(self, x0, xf=None, N=20, save=False, save_dir=None, separate=False, derivs=None):
        # Generate the Bezier trajectory with N samples per spline piece. 
        # NOTE: This assumes that xf at initialization is static.

        # Initial position
        self.x0 = x0[:3]

        ### GENERATING PATH INITIALIZATION
        target = euc_to_index(self.grid, self.x0[:3], N=self.discretization, kernel=self.kernel)
    
        tnow = time.time()

        if xf is None:
            # Query from the parental field
            traj, path = path_from_parent(self.parental_field, target, self.feasible)
        else:
            # If given a final position, simply perform raw A*
            self.xf = xf[:3]
            source = euc_to_index(self.grid, self.xf[:3], N=self.discretization, kernel=self.kernel)
            traj, path = astar3D(self.field, source, target, self.feasible)

        #TODO: Check if the flip is necessary
        #traj = np.flip(traj, axis=0)
        #path = np.flip(path, axis=0)

        print(f'Created path: {time.time() - tnow}')
        print(f'Path length: {len(path)}')

        tnow = time.time()
        _, sorted_traj, _ = straight_splits(path, traj)
        print(f'Created straight segments: {time.time() - tnow}')

        tnow = time.time()
        # Less conservative bounds
        occupied = (self.field == False)

        bounds = get_bounds(self.cell_sizes, sorted_traj, occupied=occupied, feasible=self.feasible)
        bounds = refine_bounds(bounds)
        print(f'Got bounds: {time.time() - tnow}')

        tnow = time.time()

        if derivs is not None:
            vel0 = derivs['vel0']
            accel0 = derivs['accel0']
            jerk0 = derivs['jerk0']

        try:
            cntrl_pts, cost = get_b_spline(bounds, sorted_traj, self.derivT, self.x0, self.xf,
                                v0=vel0, a0=accel0, j0=jerk0)
            print(f'Calcualted B-spline: {time.time() - tnow}')
            # print('Output cost', cost)

            # tnow = time.time()
            eval_pts = np.linspace(0., 1., N)
            b_traj = eval_b_spline(eval_pts, cntrl_pts)
            traj = np.concatenate(b_traj, axis=0)
            # print(f'Evaluating B-spline: {time.time() - tnow}')

            self.data = {
            'traj': traj.tolist(),
            'traj_raw': traj.tolist(),
            'coeffs': cntrl_pts.tolist(),
            'cost': cost,
            }

            if save:
                fp = save_dir
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fp = fp + '/'
                bound_fp = fp + 'bound.ply'
                fp = fp + 'path.json'

                bound2mesh(bounds, bound_fp)

                with open(fp, "w") as outfile:
                    json.dump(self.data, outfile, indent=4)

            if separate:
                return b_traj, True
            else:
                return traj, True
        except:
            print('Could not find feasible solution')
            return traj, False