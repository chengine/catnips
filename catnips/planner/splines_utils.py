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

def create_time_pts(deg=8, N_sec=10, tf=1.):
    #Find coefficients for T splines, each connecting one waypoint to the next
    
    # THESE COEFFICIENTS YOU CAN STORE, SO YOU ONLY NEED TO COMPUTE THEM ONCE!
    time_pts = np.linspace(0., tf, N_sec)

    T = b_spline_terms(time_pts, deg)   #(deg + 1) x 2
    dT = b_spline_term_derivs(time_pts, deg, 1)
    ddT = b_spline_term_derivs(time_pts, deg, 2)
    dddT = b_spline_term_derivs(time_pts, deg, 3)
    ddddT = b_spline_term_derivs(time_pts, deg, 4)

    data = {
    'time_pts': T,
    'd_time_pts': dT,
    'dd_time_pts': ddT,
    'ddd_time_pts': dddT,
    'dddd_time_pts': ddddT
    }

    return data

######################################################################################################

def get_qp_matrices(T, dT, ddT, dddT, ddddT, As, Bs, x0, xf, vel0=None, accel0=None, jerk0=None):
    
    N_sec = len(As)
    deg = T[0].shape[0]
    w = deg*N_sec*3
    k = deg*3
    k3 = deg

    P = []
    A = []
    b = []

    # Create equality matrices
    C = torch.zeros((3* 4* N_sec, w))
    d = torch.zeros(C.shape[0])

    # Create cost matrix P, consisting only of jerk
    for i in range(N_sec):
        #f = ddddT[i][:, -1]#+ ddddT[i][:, 0] #+ dddT[i][:, -1] + dddT[i][:, 0] + ddT[i][:, -1] + ddT[i][:, 0]
        f = np.sum(ddddT[i], axis=-1) #+ np.sum(dddT[i], axis=-1) + np.sum(ddT[i], axis=-1)
        # f = np.sum(dT[i], axis=-1)
        p_cof = torch.tensor(1e8*f).reshape(1, -1)
        P.append(p_cof)
        P.append(p_cof)
        P.append(p_cof)

        # Ax <= b
        A_ = torch.tensor(As[i])
        b_ = torch.tensor(Bs[i])
        A_x = deg*[A_[:, 0].reshape(-1, 1)]
        A_y = deg*[A_[:, 1].reshape(-1, 1)]
        A_z = deg*[A_[:, 2].reshape(-1, 1)]

        A_xs = torch.block_diag(*A_x)
        A_ys = torch.block_diag(*A_y)
        A_zs = torch.block_diag(*A_z)

        A_blck = torch.cat([A_xs, A_ys, A_zs], dim=-1)
        A.append(A_blck)
        b.extend(deg*[b_])

        # Cx = d
        if i < N_sec-1:
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

            C_t1 = torch.cat([p1, v1, a1, j1], axis=0)
            C_t2 = torch.cat([p2, v2, a2, j2], axis=0)
            C_t = torch.cat([C_t1, C_t2], axis=-1)

            n, m = C_t.shape
            n_e = m//2
            C[n*i: n*(i+1), n_e*i:n_e*(i+2)] = C_t
    
    # Create cost matrix
    Q = torch.block_diag(*P)
    # P = Q.T @ Q
    P = Q.T
    q = torch.zeros(w).reshape((-1,))

    # Create inequality matrices
    A = torch.block_diag(*A)
    b = torch.cat(b, dim=0)
    b = b.reshape((-1,))


    # Append initial and final position constraints
    p0_cof = torch.tensor(T[0][:, 0]).reshape(1, -1)
    pf_cof = torch.tensor(T[-1][:, -1]).reshape(1, -1)

    p0 = torch.block_diag(p0_cof, p0_cof, p0_cof)
    pf = torch.block_diag(pf_cof, pf_cof, pf_cof)

    C_ = torch.zeros((3*2, w))
    C_[:3, 0:n_e] = p0
    C_[3:, -n_e:] = pf

    d_ = torch.tensor(np.concatenate([x0, xf], axis=0))

    # # Add initial vel/accel/jerk constraint if exists to G_ and h_
    # if vel0 is not None:
    #     v0_cof = torch.tensor(dT[0][:, 0]).reshape(1, -1)
    #     v0 = torch.block_diag(v0_cof, v0_cof, v0_cof)
    #     Gv = torch.zeros((3, w))
    #     Gv[:3, 0:n_e] = v0
    #     G_ = torch.cat([G_, Gv], axis=0)

    #     h_ = torch.cat([h_, torch.tensor(vel0)], axis=0)

    # if accel0 is not None:
    #     a0_cof = torch.tensor(ddT[0][:, 0]).reshape(1, -1)
    #     a0 = torch.block_diag(a0_cof, a0_cof, a0_cof)
    #     Ga = torch.zeros((3, w))
    #     Ga[:3, 0:n_e] = a0
    #     G_ = torch.cat([G_, Ga], axis=0)

    #     h_ = torch.cat([h_, torch.tensor(accel0)], axis=0)

    # if jerk0 is not None:
    #     j0_cof = torch.tensor(dddT[0][:, 0]).reshape(1, -1)
    #     j0 = torch.block_diag(j0_cof, j0_cof, j0_cof)
    #     Gj = torch.zeros((3, w))
    #     Gj[:3, 0:n_e] = j0
    #     G_ = torch.cat([G_, Gj], axis=0)

    #     h_ = torch.cat([h_, torch.tensor(jerk0)], axis=0)

    # Concatenate G and h matrices
    C = torch.cat([C, C_], axis=0)

    d = torch.cat([d, d_], axis=0)
    d = d.reshape((-1,))

    return A.cpu().numpy(), b.cpu().numpy(), C.cpu().numpy(), d.cpu().numpy(), P.cpu().numpy()
