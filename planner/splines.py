import cvxpy as cvx
import scipy
import sympy as sym
import numpy as np
import torch

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
# --------------------------------------------------------------------------------#

def create_time_pts(deg=8):
    #Find coefficients for T splines, each connecting one waypoint to the next
    
    # THESE COEFFICIENTS YOU CAN STORE, SO YOU ONLY NEED TO COMPUTE THEM ONCE!
        
    tf = 1.
    time_pts = np.linspace(0., tf, 10)

    T = b_spline_terms(time_pts, deg)   #(deg + 1) x 2
    dT = b_spline_term_derivs(time_pts, deg, 1)
    ddT = b_spline_term_derivs(time_pts, deg, 2)
    dddT = b_spline_term_derivs(time_pts, deg, 3)
    ddddT = b_spline_term_derivs(time_pts, deg, 4)

    return [T, dT, ddT, dddT, ddddT]

def get_b_spline(bounds, traj, derivT, x0, xf, v0=None, a0=None, j0=None):
    # x0 = traj[0][0]
    # xf = traj[-1][-1]
    
    N = len(bounds)         #Number of segments

    T, dT, ddT, dddT, ddddT = derivT

    # Copy time points N times
    T = [T]*N
    dT = [dT]*N
    ddT = [ddT]*N
    dddT = [dddT]*N
    ddddT = [ddddT]*N

    #Set up CVX problem
    tnow = time.time()
    P, q, G, h, A, b, Q = get_qp_matrices(T, dT, ddT, dddT, ddddT, bounds, x0, xf, 
                            vel0=v0, accel0=a0, jerk0=j0)
    n_var = P.shape[0]

    x = cvx.Variable(n_var)
    x_reshaped = cvx.reshape(x, (n_var//(N*3), (N*3)))
    x_reshaped = x_reshaped.T

    #cost = cvx.pnorm(Q @ x, 2)
    cost = cvx.pnorm(x_reshaped[:, :-1] - x_reshaped[:, 1:], 2)
    obj = cvx.Minimize(cost)

    constraints = [A @ x <= b, G @ x == h]

    prob = cvx.Problem(obj, constraints)

    tnow = time.time()

    prob.solve()

    coeffs = []
    cof_splits = np.split(x.value, N)
    for cof_split in cof_splits:
        xyz = np.split(cof_split, 3)
        cof = np.stack(xyz, axis=0)
        coeffs.append(cof)
    return np.array(coeffs), prob.value

def get_qp_matrices(T, dT, ddT, dddT, ddddT, bounds, x0, xf, vel0=None, accel0=None, jerk0=None):
    
    N = len(bounds)
    deg = T[0].shape[0]
    w = deg*N*3
    k = deg*3
    k3 = deg

    P = []
    A = []
    b = []
    G = torch.zeros((3*4*N, w))

    # Create cost matrix P, consisting only of jerk
    for i in range(N):
        #f = ddddT[i][:, -1]#+ ddddT[i][:, 0] #+ dddT[i][:, -1] + dddT[i][:, 0] + ddT[i][:, -1] + ddT[i][:, 0]
        f = np.sum(ddddT[i], axis=-1) #+ np.sum(dddT[i], axis=-1) + np.sum(ddT[i], axis=-1)
        # f = np.sum(dT[i], axis=-1)
        p_cof = torch.tensor(1e8*f).reshape(1, -1)
        P.append(p_cof)
        P.append(p_cof)
        P.append(p_cof)

        # Ax <= b
        A_high_t = torch.eye(k)
        A_low_t = -torch.eye(k)
        A.append(torch.cat([A_high_t, A_low_t], axis=0))

        bound = bounds[i]
        upper = bound[:, 0]
        lower = bound[:, -1]
        b_high = torch.tensor(np.repeat(upper, k3, axis=0))
        b_low = torch.tensor(-np.repeat(lower, k3, axis=0))
        b.append(b_high)
        b.append(b_low)

        # Gx = h
        if i < N-1:
            pos1_cof = torch.tensor(T[i][:, -1]).reshape(1, -1)
            pos2_cof = torch.tensor(-T[i+1][:, 0]).reshape(1, -1)

            p1 = torch.block_diag(pos1_cof, pos1_cof, pos1_cof)
            p2 = torch.block_diag(pos2_cof, pos2_cof, pos2_cof)

            vel1_cof = torch.tensor(dT[i][:, -1]).reshape(1, -1)
            vel2_cof = torch.tensor(-dT[i+1][:, 0]).reshape(1, -1)

            v1 = torch.block_diag(vel1_cof, vel1_cof, vel1_cof)
            v2 = torch.block_diag(vel2_cof, vel2_cof, vel2_cof)

            acc1_cof = torch.tensor(ddT[i][:, -1]).reshape(1, -1)
            acc2_cof = torch.tensor(-ddT[i+1][:, 0]).reshape(1, -1)

            a1 = torch.block_diag(acc1_cof, acc1_cof, acc1_cof)
            a2 = torch.block_diag(acc2_cof, acc2_cof, acc2_cof)

            jer1_cof = torch.tensor(dddT[i][:, -1]).reshape(1, -1)
            jer2_cof = torch.tensor(-dddT[i+1][:, 0]).reshape(1, -1)

            j1 = torch.block_diag(jer1_cof, jer1_cof, jer1_cof)
            j2 = torch.block_diag(jer2_cof, jer2_cof, jer2_cof)

            G_t1 = torch.cat([p1, v1, a1, j1], axis=0)
            G_t2 = torch.cat([p2, v2, a2, j2], axis=0)
            G_t = torch.cat([G_t1, G_t2], axis=-1)

            n, m = G_t.shape
            n_e = m//2
            G[n*i: n*(i+1), n_e*i:n_e*(i+2)] = G_t
    
    # Create cost matrix
    Q = torch.block_diag(*P)
    # P = Q.T @ Q
    P = Q.T
    q = torch.zeros(w).reshape((-1,))

    # Create inequality matrices
    A = torch.block_diag(*A)
    b = torch.cat(b, axis=0)
    b = b.reshape((-1,))

    # Create equality matrices
    h = torch.zeros(G.shape[0])

    # Append initial and final position constraints
    p0_cof = torch.tensor(T[0][:, 0]).reshape(1, -1)
    pf_cof = torch.tensor(T[-1][:, -1]).reshape(1, -1)

    p0 = torch.block_diag(p0_cof, p0_cof, p0_cof)
    pf = torch.block_diag(pf_cof, pf_cof, pf_cof)

    G_ = torch.zeros((3*2, w))
    G_[:3, 0:n_e] = p0
    G_[3:, -n_e:] = pf

    h_ = torch.tensor(np.concatenate([x0, xf], axis=0))

    # Add initial vel/accel/jerk constraint if exists to G_ and h_
    if vel0 is not None:
        v0_cof = torch.tensor(dT[0][:, 0]).reshape(1, -1)
        v0 = torch.block_diag(v0_cof, v0_cof, v0_cof)
        Gv = torch.zeros((3, w))
        Gv[:3, 0:n_e] = v0
        G_ = torch.cat([G_, Gv], axis=0)

        h_ = torch.cat([h_, torch.tensor(vel0)], axis=0)

    if accel0 is not None:
        a0_cof = torch.tensor(ddT[0][:, 0]).reshape(1, -1)
        a0 = torch.block_diag(a0_cof, a0_cof, a0_cof)
        Ga = torch.zeros((3, w))
        Ga[:3, 0:n_e] = a0
        G_ = torch.cat([G_, Ga], axis=0)

        h_ = torch.cat([h_, torch.tensor(accel0)], axis=0)

    if jerk0 is not None:
        j0_cof = torch.tensor(dddT[0][:, 0]).reshape(1, -1)
        j0 = torch.block_diag(j0_cof, j0_cof, j0_cof)
        Gj = torch.zeros((3, w))
        Gj[:3, 0:n_e] = j0
        G_ = torch.cat([G_, Gj], axis=0)

        h_ = torch.cat([h_, torch.tensor(jerk0)], axis=0)

    # Concatenate G and h matrices
    G = torch.cat([G, G_], axis=0)

    h = torch.cat([h, h_], axis=0)
    h = h.reshape((-1,))

    return P.cpu().numpy(), q.cpu().numpy(), G.cpu().numpy(), h.cpu().numpy(), A.cpu().numpy(), b.cpu().numpy(), Q.cpu().numpy()

def eval_b_spline(t, coeffs):
    terms = b_spline_terms(t, coeffs.shape[-1] - 1)
    terms1 = b_spline_term_derivs(t, coeffs.shape[-1] - 1, 1)
    terms2 = b_spline_term_derivs(t, coeffs.shape[-1] - 1, 2)
    terms3 = b_spline_term_derivs(t, coeffs.shape[-1] - 1, 3)

    full_traj = []
    for coeff in coeffs:
        pos = (coeff @ terms).T
        vel = (coeff @ terms1).T
        acc = (coeff @ terms2).T
        jerk = (coeff @ terms3).T
        sub_traj = np.concatenate([pos, vel, acc, jerk], axis=-1)
        full_traj.append(sub_traj)

    return full_traj

def b_spline_terms(t, deg):
    # terms are (K choose k)(1-t)**(K-k) * t**k
    terms = []
    for i in range(deg + 1):
        scaling = scipy.special.comb(deg, i)
        term = scaling * (1-t)**(deg - i) *t**i
        terms.append(term)

    return np.array(terms).astype(np.float32)

def b_spline_term_derivs(pts, deg, d):
    # terms are (K choose k)(1-t)**(K-k) * t**k
    terms = []
    for i in range(deg + 1):
        scaling = scipy.special.comb(deg, i)
        t = sym.Symbol('t')
        term = []
        for pt in pts:
            term.append(scaling * sym.diff((1-t)**(deg - i) *t**i, t, d).subs(t, pt))
        terms.append(np.array(term))

    return np.array(terms).astype(np.float32)