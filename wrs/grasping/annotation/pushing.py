import wrs.basis.robot_math as rm

def define_actuating_pushing(hnd_s,
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
            hnd_s.gen_mesh_model(rgba=[1, 0, 0, .3]).attach_to(base)
        for push_info in push_info_list:
            gl_tip_pos, gl_tip_rotmat, hnd_pos, hnd_rotmat = push_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.gen_mesh_model(rgba=[0, 1, 0, .3]).attach_to(base)
        base.run()
    return push_info_list
