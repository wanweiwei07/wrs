import math
import pickle
import numpy as np
import basis.robot_math as rm


def define_grasp(hnd_s,
                 objcm,
                 gl_jaw_center_pos,
                 gl_jaw_center_z,
                 gl_jaw_center_y,
                 jaw_width,
                 toggle_flip=True,
                 toggle_debug=False):
    """
    :param hnd_s:
    :param objcm:
    :param gl_jaw_center_pos:
    :param gl_jaw_center_z: hand approaching direction
    :param gl_jaw_center_y: normal direction of thumb's contact surface
    :param jaw_width:
    :param objcm:
    :param toggle_flip:
    :return: a list like [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]
    author: chenhao, revised by weiwei
    date: 20200104
    """
    grasp_info_list = []
    collided_grasp_info_list = []
    grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z, gl_jaw_center_y, jaw_width)
    if not hnd_s.is_mesh_collided([objcm]):
        grasp_info_list.append(grasp_info)
    else:
        collided_grasp_info_list.append(grasp_info)
    if toggle_flip:
        grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z, -gl_jaw_center_y, jaw_width)
        if not hnd_s.is_mesh_collided([objcm]):
            grasp_info_list.append(grasp_info)
        else:
            collided_grasp_info_list.append(grasp_info)
    if toggle_debug:
        for grasp_info in collided_grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.jaw_to(jaw_width)
            hnd_s.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
        for grasp_info in grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.jaw_to(jaw_width)
            hnd_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    return grasp_info_list


def define_grasp_with_rotation(hnd_s,
                               objcm,
                               gl_jaw_center_pos,
                               gl_jaw_center_z,
                               gl_jaw_center_y,
                               jaw_width,
                               gl_rotation_ax,
                               rotation_interval=math.radians(60),
                               rotation_range=(math.radians(-180), math.radians(180)),
                               toggle_flip=True,
                               toggle_debug=False):
    """
    :param hnd_s:
    :param objcm: 
    :param gl_jaw_center_pos:
    :param gl_jaw_center_z: hand approaching direction
    :param gl_jaw_center_y: normal direction of thumb's contact surface
    :param jaw_width: 
    :param rotation_interval: 
    :param rotation_range: 
    :param toggle_flip: 
    :return: a list [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]
    author: chenhao, revised by weiwei
    date: 20200104
    """
    grasp_info_list = []
    collided_grasp_info_list = []
    for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
        tmp_rotmat = rm.rotmat_from_axangle(gl_rotation_ax, rotate_angle)
        gl_jaw_center_z_rotated = np.dot(tmp_rotmat, gl_jaw_center_z)
        gl_jaw_center_y_rotated = np.dot(tmp_rotmat, gl_jaw_center_y)
        grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z_rotated, gl_jaw_center_y_rotated,
                                             jaw_width)
        if not hnd_s.is_mesh_collided([objcm]):
            grasp_info_list.append(grasp_info)
        else:
            collided_grasp_info_list.append(grasp_info)
    if toggle_flip:
        for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
            tmp_rotmat = rm.rotmat_from_axangle(gl_rotation_ax, rotate_angle)
            gl_jaw_center_z_rotated = np.dot(tmp_rotmat, gl_jaw_center_z)
            gl_jaw_center_y_rotated = np.dot(tmp_rotmat, -gl_jaw_center_y)
            grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z_rotated, gl_jaw_center_y_rotated,
                                                 jaw_width)
            if not hnd_s.is_mesh_collided([objcm]):
                grasp_info_list.append(grasp_info)
            else:
                collided_grasp_info_list.append(grasp_info)
    if toggle_debug:
        for grasp_info in collided_grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.jaw_to(jaw_width)
            hnd_s.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
        for grasp_info in grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.jaw_to(jaw_width)
            hnd_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    return grasp_info_list


def define_pushing(hnd_s,
                   objcm,
                   gl_surface_pos,
                   gl_surface_normal,
                   cone_angle=math.radians(30),
                   icosphere_level=2,
                   local_rotation_interval=math.radians(45),
                   toggle_debug=False):
    """
    :param hnd_s:
    :param objcm:
    :param gl_surface_pos: used as cone tip
    :param gl_surface_normal: used as cone's main axis
    :param cone_angle: pushing poses will be randomized in this cone
    :param icosphere_levle: 2
    :param local_rotation_interval: discretize the rotation around the local axis of each pushing pose
    :return:
    author: weiwei
    date: 20220308
    """
    push_info_list = []
    collided_push_info_list = []
    pushing_icorotmats = rm.gen_icorotmats(icolevel=icosphere_level,
                                           crop_angle=cone_angle,
                                           crop_normal=gl_surface_normal,
                                           rotation_interval=local_rotation_interval,
                                           toggle_flat=True)
    for pushing_rotmat in pushing_icorotmats:
        push_info = hnd_s.push_at(gl_push_pos=gl_surface_pos, gl_push_rotmat=pushing_rotmat)
        if not hnd_s.is_mesh_collided([objcm]):
            push_info_list.append(push_info)
        else:
            collided_push_info_list.append(push_info)
    if toggle_debug:
        for push_info in collided_push_info_list:
            gl_tip_pos, gl_tip_rotmat, hnd_pos, hnd_rotmat = push_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
        for push_info in push_info_list:
            gl_tip_pos, gl_tip_rotmat, hnd_pos, hnd_rotmat = push_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
        base.run()
    return push_info_list


def write_pickle_file(objcm_name, grasp_info_list, root=None, file_name='preannotated_grasps.pickle', append=False):
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
    if append:
        data[objcm_name].extend(grasp_info_list)
    else:
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
    import robot_sim.end_effectors.gripper.xarm_gripper.xarm_gripper as xag
    import modeling.collision_model as cm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_s = xag.XArmGripper(enable_cc=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_localframe()
    grasp_info_list = define_grasp_with_rotation(gripper_s,
                                                 objcm,
                                                 gl_jaw_center_pos=np.array([0, 0, 0]),
                                                 gl_jaw_center_z=np.array([1, 0, 0]),
                                                 gl_jaw_center_y=np.array([0, 1, 0]),
                                                 jaw_width=.04,
                                                 gl_rotation_ax=np.array([0, 0, 1]))
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        gic = gripper_s.copy()
        gic.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        gic.gen_meshmodel().attach_to(base)
    base.run()
