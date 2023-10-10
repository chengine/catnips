import torch
import numpy as np
import numpy.linalg as la

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rot_x = lambda phi: torch.tensor([
#         [1., 0., 0.],
#         [0., torch.cos(phi), -torch.sin(phi)],
#         [0., torch.sin(phi), torch.cos(phi)]], dtype=torch.float32)

rot_x_np = lambda phi: np.array([
        [1., 0., 0.],
        [0., np.cos(phi), -np.sin(phi)],
        [0., np.sin(phi), np.cos(phi)]], dtype=np.float32)

rot_x = lambda phi: torch.tensor([
        [1., 0., 0.],
        [0., torch.cos(phi), -torch.sin(phi)],
        [0., torch.sin(phi), torch.cos(phi)]], dtype=torch.float32, device=device)

def mahalanobis(u, v, cov):
    delta = u - v
    return delta @ torch.inverse(cov) @ delta

def nerf_matrix_to_ngp_torch(pose, trans):
    neg_yz = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=torch.float32)

    flip_yz = torch.tensor([
        [0, 1, 0], 
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=torch.float32)
    return flip_yz@ pose @ neg_yz, flip_yz @ trans

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
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
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def calcSO3Err(R_gt, R_est):
    ''' angle between two rotation matrices (in degrees) '''
    rotDiff = np.dot(R_gt, np.transpose(R_est))
    trace = np.trace(rotDiff) 
    if trace < -1 and (-1 - trace) < 0.0001:
        return np.rad2deg(np.arccos(-1))
    if trace > 3 and (trace - 3) < 0.0001:
        return np.rad2deg(np.arccos(1))
    return np.rad2deg(np.arccos((trace-1.0)/2.0))

def calcSE3Err(T_gt, T_est):
    ''' translation err & angle between two rotation matrices (in degrees) '''
    ang_err_deg = calcSO3Err(T_gt[0:3, 0:3], T_est[0:3, 0:3])
    t_err = np.linalg.norm(T_gt[0:3, 3] - T_est[0:3, 3])
    return t_err, ang_err_deg

def skew_matrix_torch(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

def rot_matrix_to_vec(R):
    batch_dims = R.shape[:-2]

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

    def acos_safe(x, eps=1e-7):
        """https://github.com/pytorch/pytorch/issues/8069"""
        slope = np.arccos(1-eps) / eps
        # TODO: stop doing this allocation once sparse gradients with NaNs (like in
        # th.where) are handled differently.
        buf = torch.empty_like(x)
        good = abs(x) <= 1-eps
        bad = ~good
        sign = torch.sign(x[bad])
        buf[good] = torch.acos(x[good])
        buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
        return buf

    # angle = torch.acos((trace - 1) / 2)[..., None]
    angle = acos_safe((trace - 1) / 2)[..., None]
    # print(trace, angle)

    vec = (
        1
        / (2 * torch.sin(angle + 1e-10))
        * torch.stack(
            [
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ],
            dim=-1,
        )
    )

    # needed to overwrite nanes from dividing by zero
    vec[angle[..., 0] == 0] = torch.zeros(3, device=R.device)

    # eg TensorType["batch_size", "views", "max_objects", 3, 1]
    rot_vec = (angle * vec)[...]

    return rot_vec

def vec_to_rot_matrix(rot_vec):
    assert not torch.any(torch.isnan(rot_vec))

    angle = torch.norm(rot_vec, dim=-1, keepdim=True)

    axis = rot_vec / (1e-10 + angle)
    S = skew_matrix(axis)
    # print(S.shape)
    # print(angle.shape)
    angle = angle[...,None]
    rot_matrix = (
            torch.eye(3)
            + torch.sin(angle) * S
            + (1 - torch.cos(angle)) * S @ S
            )
    return rot_matrix

def skew_matrix(vec):
    batch_dims = vec.shape[:-1]
    S = torch.zeros(*batch_dims, 3, 3)
    S[..., 0, 1] = -vec[..., 2]
    S[..., 0, 2] =  vec[..., 1]
    S[..., 1, 0] =  vec[..., 2]
    S[..., 1, 2] = -vec[..., 0]
    S[..., 2, 0] = -vec[..., 1]
    S[..., 2, 1] =  vec[..., 0]
    return S
