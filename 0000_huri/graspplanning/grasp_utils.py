import copy
import pickle

import numpy as np

import environment.bulletcdhelper as bch
import environment.collisionmodel as cm
import utiltools.robotmath as rm

bcdchecker = bch.MCMchecker(toggledebug=False)

def ishndobjcollided(hndfa, jawwidth, homomat, objcm):
    """

    :param homomat:
    :param obstaclecmlist:
    :return:

    author: ruishuang, revised by weiwei
    date: 20191122
    """

    hnd = hndfa.genHand()
    hnd.set_homomat(homomat)
    setjawwidth = 30 if jawwidth>=30 else jawwidth+10
    hnd.setjawwidth(setjawwidth)
    iscollided = bcdchecker.isMeshListMeshListCollided(hnd.cmlist, [objcm])
    return iscollided

def define_grasp(hndfa, finger_center, finger_normal, hand_normal, jawwidth, objcm, toggleflip = True):
    """

    :param hndfa:
    :param finger_center:
    :param finger_normal:
    :param hand_normal:
    :param jawwidth:
    :param objcm:
    :param toggleflip:
    :return:

    author: chenhao, revised by weiwei
    date: 20200104
    """

    effect_grasp = []
    hnd = hndfa.genHand()
    grasp = hnd.approachat(finger_center[0], finger_center[1], finger_center[2],
                           finger_normal[0], finger_normal[1], finger_normal[2],
                           hand_normal[0], hand_normal[1], hand_normal[2], jawwidth=jawwidth)
    # if not ishndobjcollided(hndfa, grasp[0], grasp[2], obj_cmodel):
    effect_grasp.append(grasp)
    if toggleflip:
        grasp_flipped = hnd.approachat(finger_center[0], finger_center[1], finger_center[2],
                                       -finger_normal[0], -finger_normal[1], -finger_normal[2],
                                       hand_normal[0], hand_normal[1], hand_normal[2], jawwidth=jawwidth)
        if not ishndobjcollided(hndfa, grasp_flipped[0], grasp_flipped[2], objcm):
            effect_grasp.append(grasp_flipped)
    return effect_grasp

def define_grasp_with_rotation(hndfa, grasp_center, finger_normal, hand_normal, jawwidth, objcm,
                               rotation_interval=15, rotation_range=(-90, 90), toggleflip = True):
    """

    :param hndfa:
    :param grasp_center:
    :param finger_normal:
    :param hand_normal:
    :param jawwidth:
    :param objcm:
    :param rotation_interval:
    :param rotation_range:
    :param toggleflip:
    :return:

    author: chenhao, revised by weiwei
    date: 20200104
    """

    effect_grasp = []
    for rotate_angle in range(rotation_range[0], rotation_range[1], rotation_interval):
        hnd = hndfa.genHand()
        hand_normal_rotated = np.dot(rm.rodrigues(finger_normal, rotate_angle), np.asarray(hand_normal))
        grasp = hnd.approachat(grasp_center[0], grasp_center[1], grasp_center[2],
                               finger_normal[0], finger_normal[1], finger_normal[2],
                               hand_normal_rotated[0], hand_normal_rotated[1], hand_normal_rotated[2], jawwidth=jawwidth)
        # if not ishndobjcollided(hndfa, grasp[0], grasp[2], obj_cmodel) == False:
        effect_grasp.append(grasp)
        if toggleflip:
            grasp_flipped = hnd.approachat(grasp_center[0], grasp_center[1], grasp_center[2],
                                           -finger_normal[0], -finger_normal[1], -finger_normal[2],
                                           hand_normal_rotated[0], hand_normal_rotated[1], hand_normal_rotated[2], jawwidth=jawwidth)
            if not ishndobjcollided(hndfa, grasp_flipped[0], grasp_flipped[2], objcm):
                effect_grasp.append(grasp_flipped)
    return effect_grasp

def define_suction(hndfa, finger_center, finger_normal, hand_normal, objcm):
    """

    :param hndfa:
    :param finger_center:
    :param finger_normal:
    :param hand_normal:
    :param objcm:
    :param toggleflip:
    :return:

    author: chenhao, revised by weiwei
    date: 20200104
    """

    effect_grasp = []
    hnd = hndfa.genHand(usesuction=True)
    grasp = hnd.approachat(finger_center[0], finger_center[1], finger_center[2],
                           finger_normal[0], finger_normal[1], finger_normal[2],
                           hand_normal[0], hand_normal[1], hand_normal[2], jawwidth=0)
    if not ishndobjcollided(hndfa, grasp[0], grasp[2], objcm):
        effect_grasp.append(grasp)
    return effect_grasp

def define_suction_with_rotation(hndfa, grasp_center, finger_normal, hand_normal, objcm,
                                 rotation_interval=15, rotation_range=(-90, 90)):
    """

    :param hndfa:
    :param grasp_center:
    :param finger_normal:
    :param hand_normal:
    :param objcm:
    :param rotation_interval:
    :param rotation_range:
    :param toggleflip:
    :return:

    author: chenhao, revised by weiwei
    date: 20200104
    """

    effect_grasp = []
    for rotate_angle in range(rotation_range[0], rotation_range[1], rotation_interval):
        hnd = hndfa.genHand(usesuction=True)
        hand_normal_rotated = np.dot(rm.rodrigues(finger_normal, rotate_angle), np.asarray(hand_normal))
        grasp = hnd.approachat(grasp_center[0], grasp_center[1], grasp_center[2],
                               finger_normal[0], finger_normal[1], finger_normal[2],
                               hand_normal_rotated[0], hand_normal_rotated[1], hand_normal_rotated[2], jawwidth=0)
        if not ishndobjcollided(hndfa, grasp[0], grasp[2], objcm) == False:
            effect_grasp.append(grasp)
    return effect_grasp

def write_pickle_file(model_name, effect_grasps, root=None):
    """

    :param model_name:
    :param effect_grasps:
    :return:

    author: chenhao, revised by weiwei
    date: 20200104
    """

    if root is None:
        directory = "./"
    else:
        directory = root+"/"

    try:
        data = pickle.load(open(directory+'predefinedgrasps.pickle', 'rb'))
    except:
        print("load failed, create new file.")
        data = {}

    data[model_name] = effect_grasps
    for k, v in data.items():
        print(k, len(v))
    pickle.dump(data, open(directory+'predefinedgrasps.pickle', 'wb'))

def load_pickle_file(model_name, root=None):
    """

    :param model_name:
    :param effect_grasps:
    :return:

    author: chenhao, revised by weiwei
    date: 20200105
    """

    if root is None:
        directory = "./"
    else:
        directory = root+"/"

    try:
        data = pickle.load(open(directory+'predefinedgrasps.pickle', 'rb'))
        for k, v in data.items():
            print(k, len(v))
        effect_grasps = data[model_name]
        return effect_grasps
    except:
        print("load failed, create new graqsp file or grasp first.")
        raise ValueError("File or data not found!")

