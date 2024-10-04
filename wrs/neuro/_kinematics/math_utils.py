import math
import torch

# constants
_EPS = torch.finfo(torch.float32).eps  # 1.192e-07
_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}


def unit_vector(vector, toggle_length=False):
    """
    :param vector: 1-by-3 nparray
    :return: the unit of a vector
    author: weiwei
    date: 20200701osaka
    """
    length = torch.norm(vector)
    if torch.isclose(length, torch.tensor(.0)):
        if toggle_length:
            return 0.0, torch.zeros_like(vector)
        else:
            return torch.zeros_like(vector)
    if toggle_length:
        return length, vector / torch.norm(vector)
    else:
        return vector / torch.norm(vector)


def rotmat_from_axangle(axis, angle):
    """
    Compute the rodrigues matrix using the given axis and angle

    :param axis: 1x3 array
    :param angle:  angle in radian
    :return: 3x3 rotmat
    author: weiwei
    date: 20161220
    """
    axis = unit_vector(axis)
    if torch.isclose(torch.norm(axis), torch.tensor(.0)):
        return torch.eye(3, dtype=angle.dtype, device=angle.device)
    a = torch.cos(angle / 2.0)
    b, c, d = -axis * torch.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    # rotmat = torch.tensor([[aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac)],
    #                        [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab)],
    #                        [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc]])
    # return rotmat.requires_grad_(angle.requires_grad)
    rotmat = torch.zeros((3, 3), dtype=angle.dtype, device=angle.device)
    rotmat[0, 0] = aa + bb - cc - dd
    rotmat[0, 1] = 2 * (bc + ad)
    rotmat[0, 2] = 2 * (bd - ac)
    rotmat[1, 0] = 2 * (bc - ad)
    rotmat[1, 1] = aa + cc - bb - dd
    rotmat[1, 2] = 2 * (cd + ab)
    rotmat[2, 0] = 2 * (bd + ac)
    rotmat[2, 1] = 2 * (cd - ab)
    rotmat[2, 2] = aa + dd - bb - cc
    return rotmat


def rotmat_from_euler(ai, aj, ak, axes='sxyz'):
    first_ax, parity, repetition, frame = _AXES2TUPLE[axes]
    i = first_ax
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    result_rotmat = torch.eye(3)
    if repetition:
        result_rotmat[i, i] = cj
        result_rotmat[i, j] = sj * si
        result_rotmat[i, k] = sj * ci
        result_rotmat[j, i] = sj * sk
        result_rotmat[j, j] = -cj * ss + cc
        result_rotmat[j, k] = -cj * cs - sc
        result_rotmat[k, i] = -sj * ck
        result_rotmat[k, j] = cj * sc + cs
        result_rotmat[k, k] = cj * cc - ss
    else:
        result_rotmat[i, i] = cj * ck
        result_rotmat[i, j] = sj * sc - cs
        result_rotmat[i, k] = sj * cc + ss
        result_rotmat[j, i] = cj * sk
        result_rotmat[j, j] = sj * ss + cc
        result_rotmat[j, k] = sj * cs - sc
        result_rotmat[k, i] = -sj
        result_rotmat[k, j] = cj * si
        result_rotmat[k, k] = cj * ci
    return result_rotmat


def homomat_from_posrot(pos=torch.zeros(3), rotmat=torch.eye(3)):
    """
    build a 4x4 homogeneous matrix
    :param pos: 1x3
    :param rotmat: 3x3
    :return:
    author: weiwei
    date: 20190313
    """
    homomat = torch.eye(4)
    homomat[:3, :3] = rotmat
    homomat[:3, 3] = pos
    return homomat


def delta_w_between_rotmat(src_rotmat, tgt_rotmat):
    """
    compute angle*ax from src_rotmat to tgt_rotmat
    the following relation holds for the returned delta_w
    rotmat_from_axangle(np.linalg.norm(deltaw), unit_vec(deltaw)).dot(src_rotmat) = tgt_rotmat
    :param src_rotmat: 3x3
    :param tgt_rotmat: 3x3
    :return:
    author: weiwei
    date: 20200326
    """
    delta_rotmat = tgt_rotmat @ src_rotmat.T
    tmp_vec = torch.stack([delta_rotmat[2, 1] - delta_rotmat[1, 2],
                           delta_rotmat[0, 2] - delta_rotmat[2, 0],
                           delta_rotmat[1, 0] - delta_rotmat[0, 1]])
    tmp_vec_norm = torch.norm(tmp_vec)
    if torch.isclose(tmp_vec_norm, torch.tensor(0.0)):
        return torch.zeros(3, dtype=src_rotmat.dtype, device=src_rotmat.device)
    else:
        trace = torch.trace(delta_rotmat)
        return torch.atan2(tmp_vec_norm, (trace - 1.0)) / tmp_vec_norm * tmp_vec

    # if tmp_vec_norm > _EPS:
    #     trace = torch.trace(delta_rotmat)
    #     delta_w = torch.atan2(tmp_vec_norm, (trace - 1.0)) / tmp_vec_norm * tmp_vec
    # else:
    #     trace = torch.trace(delta_rotmat)
    #     if trace > 2.9999:
    #         delta_w = torch.tensor([0, 0, 0], dtype=src_rotmat.dtype, device=src_rotmat.device)
    #     else:
    #         axis=torch.diag
    # elif delta_rotmat[0, 0] > 0 and delta_rotmat[1, 1] > 0 and delta_rotmat[2, 2] > 0:
    #     delta_w = torch.tensor([0, 0, 0], dtype=src_rotmat.dtype, device=src_rotmat.device)
    # else:
    #     delta_w = torch.pi / 2 * (torch.diag(delta_rotmat) + 1)
    # return delta_w


# def delta_w_between_rotmat(src_rotmat, tgt_rotmat):
#     """
#     compute angle*ax from src_rotmat to tgt_rotmat
#     the following relation holds for the returned delta_w
#     rotmat_from_axangle(np.linalg.norm(deltaw), unit_vec(deltaw)).dot(src_rotmat) = tgt_rotmat
#     :param src_rotmat: 3x3
#     :param tgt_rotmat: 3x3
#     :return:
#     author: weiwei
#     date: 20200326
#     """
#     delta_rotmat = tgt_rotmat @ src_rotmat.T
#     clipped_trace = torch.clip((torch.trace(delta_rotmat) - 1) / 2.0, -1.0, 1.0)
#     angle = torch.acos(clipped_trace)
#     if torch.isclose(angle, torch.tensor(0.0)):
#         return torch.zeros(3, dtype=src_rotmat.dtype, device=src_rotmat.device)
#     else:
#         axis = torch.stack([delta_rotmat[2, 1] - delta_rotmat[1, 2],
#                             delta_rotmat[0, 2] - delta_rotmat[2, 0],
#                             delta_rotmat[1, 0] - delta_rotmat[0, 1]]) / (2 * torch.sin(angle))
#         return angle * axis


def rel_pose(pos0, rotmat0, pos1, rotmat1):
    """
    relpos of rot1, pos1 with respect to rot0 pos0
    :param rot0: 3x3 nparray
    :param pos0: 1x3 nparray
    :param rot1:
    :param pos1:
    :return:
    author: weiwei
    date: 20180811, 20240223
    """
    rel_pos = rotmat0.T @ (pos1 - pos0)
    rel_rotmat = rotmat0.T @ rotmat1
    return (rel_pos, rel_rotmat)


def diff_between_posrot(src_pos, src_rotmat, tgt_pos, tgt_rotmat):
    """
    compute the error between the given tcp and tgt_tcp
    :param src_pos:
    :param src_rotmat
    :param tgt_pos: the position vector of the goal (could be a single value or a list of jntid)
    :param tgt_rotmat: the rotation matrix of the goal (could be a single value or a list of jntid)
    :return: a 1x6 nparray where the first three indicates the displacement in pos,
                the second three indictes the displacement in rotmat
    author: weiwei
    date: 20230929
    """
    delta = torch.zeros(6)
    delta[0:3] = (tgt_pos - src_pos)
    delta[3:6] = delta_w_between_rotmat(src_rotmat, tgt_rotmat)
    return delta.t() @ delta
