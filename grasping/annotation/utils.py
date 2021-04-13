import copy
import math
import pickle
import numpy as np
import basis.robot_math as rm


def define_grasp(hnd,
                 objcm,
                 gl_jaw_center,
                 gl_hndz,
                 gl_hndx,
                 jaw_width,
                 toggle_flip=True):
    """
    :param hnd:
    :param fgr_center:
    :param fgr_normal:
    :param hnd_normal:
    :param jaw_width:
    :param objcm:
    :param toggle_flip:
    :return: a list like [[jaw_width, gl_jaw_center, pos, rotmat], ...]
    author: chenhao, revised by weiwei
    date: 20200104
    """
    grasp_info_list = []
    grasp_info = hnd.grip_at(gl_jaw_center, gl_hndz, gl_hndx, jaw_width)
    if not hnd.is_mesh_collided([objcm]):
        grasp_info_list.append(grasp_info)
        if toggle_flip:
            grasp_info_flipped = [grasp_info[0], grasp_info[1], grasp_info[2],
                                  rm.rotmat_from_axangle(grasp_info[3][:, 2], math.pi).dot(grasp_info[3])]
            grasp_info_list.append(grasp_info_flipped)
    return grasp_info_list


def define_grasp_with_rotation(hnd,
                               objcm,
                               gl_jaw_center,
                               gl_hndz,
                               gl_hndx,
                               jaw_width,
                               rotation_ax,
                               rotation_interval=math.radians(60),
                               rotation_range=(math.radians(-180), math.radians(180)),
                               toggle_flip=True):
    """
    :param hnd: 
    :param objcm: 
    :param gl_jaw_center: 
    :param gl_hndz: 
    :param gl_hndx: 
    :param jaw_width: 
    :param rotation_interval: 
    :param rotation_range: 
    :param toggle_flip: 
    :return: a list [[jaw_width, gl_jaw_center, pos, rotmat], ...]
    author: chenhao, revised by weiwei
    date: 20200104
    """
    grasp_info_list = []
    for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
        tmp_rotmat = rm.rotmat_from_axangle(rotation_ax, rotate_angle)
        gl_hndz_rotated = np.dot(tmp_rotmat, gl_hndz)
        gl_hndx_rotated = np.dot(tmp_rotmat, gl_hndx)
        grasp_info = hnd.grip_at(gl_jaw_center, gl_hndz_rotated, gl_hndx_rotated, jaw_width)
        if not hnd.is_mesh_collided([objcm]):
            grasp_info_list.append(grasp_info)
            if toggle_flip:
                grasp_info_flipped = [grasp_info[0], grasp_info[1], grasp_info[2],
                                      rm.rotmat_from_axangle(grasp_info[3][:, 2], math.pi).dot(grasp_info[3])]
                grasp_info_list.append(grasp_info_flipped)
    return grasp_info_list


# def define_suction(hndfa, finger_center, finger_normal, hand_normal, objcm):
#     """
#
#     :param hndfa:
#     :param finger_center:
#     :param finger_normal:
#     :param hand_normal:
#     :param objcm:
#     :param toggleflip:
#     :return:
#
#     author: chenhao, revised by weiwei
#     date: 20200104
#     """
#
#     effect_grasp = []
#     hnd = hndfa.genHand(usesuction=True)
#     grasp = hnd.approachat(finger_center[0], finger_center[1], finger_center[2],
#                            finger_normal[0], finger_normal[1], finger_normal[2],
#                            hand_normal[0], hand_normal[1], hand_normal[2], jaw_width=0)
#     if not ishndobjcollided(hndfa, grasp[0], grasp[2], objcm):
#         effect_grasp.append(grasp)
#     return effect_grasp
#
#
# def define_suction_with_rotation(hndfa, grasp_center, finger_normal, hand_normal, objcm,
#                                  rotation_interval=15, rotation_range=(-90, 90)):
#     """
#
#     :param hndfa:
#     :param grasp_center:
#     :param finger_normal:
#     :param hand_normal:
#     :param objcm:
#     :param rotation_interval:
#     :param rotation_range:
#     :param toggleflip:
#     :return:
#
#     author: chenhao, revised by weiwei
#     date: 20200104
#     """
#
#     effect_grasp = []
#     for rotate_angle in range(rotation_range[0], rotation_range[1], rotation_interval):
#         hnd = hndfa.genHand(usesuction=True)
#         hand_normal_rotated = np.dot(rm.rodrigues(finger_normal, rotate_angle), np.asarray(hand_normal))
#         grasp = hnd.approachat(grasp_center[0], grasp_center[1], grasp_center[2],
#                                finger_normal[0], finger_normal[1], finger_normal[2],
#                                hand_normal_rotated[0], hand_normal_rotated[1], hand_normal_rotated[2], jaw_width=0)
#         if not ishndobjcollided(hndfa, grasp[0], grasp[2], objcm) == False:
#             effect_grasp.append(grasp)
#     return effect_grasp


def write_pickle_file(objcm_name, grasp_info_list, root=None, file_name='preannotated_grasps.pickle'):
    """
    if model_name was saved, replace the old grasp info.
    if model_name was never saved, additionally save it.
    :param objcm_name:
    :param grasp_info_list:
    :param root:
    :param file_name:
    :return:
    author: chenhao, revised by weiwei
    date: 20200104
    """
    if root is None:
        directory = "./"
    else:
        directory = root + "/"
    try:
        data = pickle.load(open(directory + file_name, 'rb'))
    except:
        print("load failed, create new data.")
        data = {}
    data[objcm_name] = grasp_info_list
    for k, v in data.items():
        print(k, len(v))
    pickle.dump(data, open(directory + file_name, 'wb'))


def load_pickle_file(objcm_name, root=None, file_name='preannotated_grasps.pickle'):
    """
    :param objcm_name:
    :param root:
    :param file_name:
    :return:
    author: chenhao, revised by weiwei
    date: 20200105
    """
    if root is None:
        directory = "./"
    else:
        directory = root + "/"
    try:
        data = pickle.load(open(directory + file_name, 'rb'))
        for k, v in data.items():
            print(k, len(v))
        grasp_info_list = data[objcm_name]
        return grasp_info_list
    except:
        raise ValueError("File or data not found!")


if __name__ == '__main__':
    import os
    import basis
    import robotsim.grippers.xarm_gripper.xarm_gripper as xag
    import modeling.collisionmodel as cm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_s = xag.XArmGripper(enable_cc=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_localframe()
    grasp_info_list = define_grasp_with_rotation(gripper_s,
                                                 objcm,
                                                 gl_jaw_center=np.array([0,0,0]),
                                                 gl_hndz=np.array([1,0,0]),
                                                 gl_hndx=np.array([0,1,0]),
                                                 jaw_width=.04,
                                                 rotation_ax=np.array([0,0,1]))
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center, pos, rotmat = grasp_info
        gic = gripper_s.copy()
        gic.fix_to(pos, rotmat)
        gic.jaw_to(jaw_width)
        print(pos, rotmat)
        gic.gen_meshmodel().attach_to(base)
    base.run()
