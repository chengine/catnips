#%%
import numpy as np
from scipy.integrate import odeint
import do_mpc
from casadi import *
import time

#Drone Params
dt = 0.03
mass = 0.18 # kg
g = 9.81 # m/s^2
I = np.array([(0.00025, 0, 2.55e-6),
              (0, 0.000232, 0),
              (2.55e-6, 0, 0.0003738)])
invI = np.linalg.inv(I)

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def f(pos, vel, rates, effort):

    d_pos = vel
    d_vel = np.array([
        -rates[0]* sin(rates[2]),
        rates[0] * cos(rates[2]) * sin(rates[1]),
        g - rates[0] * cos(rates[2]) * cos(rates[1])
    ])
    d_rates = effort

    return d_pos, d_vel, d_rates

def f_step(state, actions):

    pos, vel, rates = state[:3], state[3:6], state[6:]

    deriv = np.zeros(9)

    deriv[:3] = vel
    deriv[3:6] = np.array([
        -rates[0]* sin(rates[2]),
        rates[0] * cos(rates[2]) * sin(rates[1]),
        g - rates[0] * cos(rates[2]) * cos(rates[1])
    ])
    deriv[6:9] = actions

    return deriv

def step(x, u):
    t = np.linspace(0., dt, 10)
    dyn = lambda x, t, u: f_step(x, u)
    sol = odeint(dyn, x, t, args=(u,))

    return sol[-1]

def state2pose(x):
    # TODO!!!: Rewrite this to handle the simplified drone dynamics

    pos = x[:3]

    pose = np.eye(4)
    pose[:3, -1] = pos

    return pose

class MPC():
    def __init__(self, N=40) -> None:
        self.N = N

        self.init_dynamics()

    def init_dynamics(self):
        model_type = 'continuous' # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        ### 9 D Drone
        # # Define states
        self.pos = self.model.set_variable(var_type='_x', var_name='pos', shape=(3,1))
        self.vel = self.model.set_variable(var_type='_x', var_name='vel', shape=(3,1))
        self.rates = self.model.set_variable(var_type='_x', var_name='rates', shape=(3,1))

        # Define inputs
        self.tau = self.model.set_variable(var_type='_u', var_name='tau', shape=(3, 1))

        d_pos, d_vel, d_rates = f(self.pos, self.vel, self.rates, self.tau)

        self.model.set_rhs('pos', d_pos)
        self.model.set_rhs('vel', SX(d_vel))
        self.model.set_rhs('rates', d_rates)

        self.model.setup()

    def solve(self, As, Bs, x0, xf):

        K = len(As)
        constraint_index = []
        for i, split in enumerate(np.array_split(np.arange(self.N), K)):
            constraint_index += [i]*(len(split) - 1)
            if i < K-1:
                constraint_index.append((i, i+1))

        setup_mpc = {
            'n_horizon': self.N,
            't_step': dt,
            'n_robust': 1,
            'store_full_solution': True,
        }

        mpc = do_mpc.controller.MPC(self.model)
        mpc.set_param(**setup_mpc)
        mpc.settings.supress_ipopt_output()

        ### For 9 D drone
        mterm = self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2
        lterm = (g-self.rates[0])**2 + self.rates[1]**2 + self.rates[2]**2 + 10*(self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)
        mpc.set_rterm(
            tau=1e-2
        )

        mpc.set_objective(mterm=mterm, lterm=lterm)

        mpc.prepare_nlp()

        for i, constraint_num in enumerate(constraint_index):

            current_pos = mpc.opt_x['_x',i, 0, 0, 'pos']

            if isinstance(constraint_num, int):
                current_ind = constraint_num
            elif isinstance(constraint_num, tuple):
                current_ind = constraint_num[0]
                next_ind = constraint_num[1]
            else:
                raise NotImplementedError

            # Point is also in the current polytope
            A_current = np.array(As[current_ind])
            B_current = np.array(Bs[current_ind])

            lhs = A_current @ current_pos

            mpc.nlp_cons.append(
                lhs
            )

            mpc.nlp_cons_ub.append(B_current)
            mpc.nlp_cons_lb.append(-1e10*np.ones_like(B_current))

            if isinstance(constraint_num, tuple):
                # Point is also in the next polytope
                A_next = np.array(As[next_ind])
                B_next = np.array(Bs[next_ind])
                lhs = A_next @ current_pos

                mpc.nlp_cons.append(
                    lhs
                )

                mpc.nlp_cons_ub.append(B_next)
                mpc.nlp_cons_lb.append(-1e10*np.ones_like(B_next))

            # Make sure above the floor
            # lhs = current_pos[-1]
            # mpc.nlp_cons.append(
            #     lhs
            # )

            # mpc.nlp_cons_ub.append(0.5)
            # mpc.nlp_cons_lb.append(z_floor)

        # Final position constraint
        final_pos = mpc.opt_x['_x',self.N, 0, 0, 'pos']

        mpc.nlp_cons.append(
            final_pos
        )

        mpc.nlp_cons_ub.append(xf[:3])
        mpc.nlp_cons_lb.append(xf[:3])

        mpc.create_nlp()
        mpc.set_initial_guess()

        tnow = time.time()
        if len(x0) == 3:
            start = np.zeros(9)
            start[:3] = x0
        else:
            start = x0

        _ = mpc.make_step(start)
        print('MPC Time: ', time.time() - tnow)

        states = mpc.data.prediction(('_x',)).squeeze().T
        controls = mpc.data.prediction(('_u',)).squeeze().T

        return states, controls