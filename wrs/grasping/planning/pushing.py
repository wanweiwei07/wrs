import math
from wrs import basis, modeling as cm
import wrs.grasping.annotation.gripping as gag


def plan_pushing(hnd_s,
                 objcm,
                 cone_angle=math.radians(30),
                 icosphere_level=2,
                 local_rotation_interval=math.radians(22.5),
                 max_samples=100,
                 min_dist_between_sampled_contact_points=.005,
                 contact_offset=.002,
                 toggle_debug=False):
    """

    :param hnd_s:
    :param objcm:
    :param cone_angle:
    :param icosphere_level:
    :param local_rotation_interval:
    :param max_samples:
    :param min_dist_between_sampled_contact_points:
    :param contact_offset:
    :return:
    """
    contact_points, contact_normals = objcm.sample_surface(n_samples=max_samples,
                                                           radius=min_dist_between_sampled_contact_points / 2,
                                                           toggle_option='normals')
    push_info_list = []
    for i, cpn in enumerate(zip(contact_points, contact_normals)):
        print(f"{i} of {len(contact_points)} done!")
        push_info_list += gag.define_pushing(hnd_s,
                                            objcm,
                                            gl_surface_pos=cpn[0] + cpn[1] * contact_offset,
                                            gl_surface_normal=cpn[1],
                                            cone_angle=cone_angle,
                                            icosphere_level=icosphere_level,
                                            local_rotation_interval=local_rotation_interval,
                                            toggle_debug=toggle_debug)
    return push_info_list


def write_pickle_file(objcm_name, push_info_list, root=None, file_name='preannotated_push.pickle', append=False):
    if root is None:
        root = './'
    gag.write_pickle_file(objcm_name, push_info_list, root=root, file_name=file_name, append=append)


def load_pickle_file(objcm_name, root=None, file_name='preannotated_push.pickle'):
    if root is None:
        root = './'
    return gag.load_pickle_file(objcm_name, path=root, file_name=file_name)


if __name__ == '__main__':
    import os
    import wrs.robot_sim.end_effectors.grippers.robotiq85_gelsight.robotiq85_gelsight_pusher as rtqp
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_s = rtqp.Robotiq85GelsightPusher()
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_local_frame()
    push_info_list = plan_pushing(gripper_s, objcm, cone_angle=math.radians(60),
                                   local_rotation_interval=math.radians(45), toggle_debug=False)
    for push_info in push_info_list:
        gl_push_pos, gl_push_rotmat, hnd_pos, hnd_rotmat = push_info
        gic = gripper_s.copy()
        gic.fix_to(hnd_pos, hnd_rotmat)
        gic.gen_mesh_model().attach_to(base)
    base.run()
