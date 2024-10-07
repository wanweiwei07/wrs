import os
import math
import numpy as np
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.robot_interface as ri
import wrs.robot_sim._kinematics.jlchain as rkjlc


class CobottaRIPPS(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_plate = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     home_conf=np.zeros(0),
                                     name='base_plate_ripps')
        self.base_plate.jnts[1]['loc_pos'] = np.array([0, 0, 0.01])
        self.base_plate.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base_plate_ripps.stl")
        self.base_plate.lnks[0]['rgba'] = [.55, .55, .55, 1]
        self.base_plate.finalize()
        # arm
        arm_homeconf = np.zeros(6)
        arm_homeconf[1] = -math.pi / 6
        arm_homeconf[2] = math.pi / 2
        arm_homeconf[4] = math.pi / 6
        self.arm = cbta.CobottaArm(pos=self.base_plate.jnts[-1]['gl_posq'],
                                   rotmat=self.base_plate.jnts[-1]['gl_rotmatq'],
                                   home_conf=arm_homeconf,
                                   name='arm', enable_cc=False)
        # grippers
        self.gripper_loc_rotmat = rm.rotmat_from_axangle([0,0,1], np.pi) # 20220607 rotate the pipetting end_type with 180^o.
        self.hnd = cbtp.CobottaPipette(pos=self.arm.jnts[-1]['gl_posq'],
                                       rotmat=self.arm.jnts[-1]['gl_rotmatq'].dot(self.gripper_loc_rotmat),
                                       name='grippers', enable_cc=False)
        # tool center point
        self.arm.jlc.flange_jnt_id = -1
        self.arm.jlc._loc_flange_pos = self.gripper_loc_rotmat.dot(self.hnd.jaw_center_pos)
        self.arm.jlc._loc_flange_rotmat = self.gripper_loc_rotmat.dot(self.hnd.jaw_center_rotmat)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.base_plate, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.jlc, [0, 1, 2, 4, 5, 7])
        active_list = [self.arm.lnks[0],
                       self.arm.lnks[1],
                       self.arm.lnks[2],
                       self.arm.lnks[3],
                       self.arm.lnks[4],
                       self.arm.lnks[5],
                       self.arm.lnks[6],
                       self.hnd.jlc.lnks[0],
                       self.hnd.jlc.lnks[1],
                       self.hnd.jlc.lnks[2],
                       self.hnd.jlc.lnks[4],
                       self.hnd.jlc.lnks[5],
                       self.hnd.jlc.lnks[7]]
        self.cc.set_active_cdlnks(active_list)
        from_list = [self.base_plate.lnks[0],
                     self.arm.lnks[0],
                     self.arm.lnks[1]]
        into_list = [self.arm.lnks[4],
                     self.hnd.jlc.lnks[0],
                     self.hnd.jlc.lnks[1],
                     self.hnd.jlc.lnks[2],
                     self.hnd.jlc.lnks[4],
                     self.hnd.jlc.lnks[5]]
        self.cc.set_cdpair(from_list, into_list)
        from_list = [self.arm.lnks[2],
                     self.arm.lnks[3]]
        into_list = [self.hnd.jlc.lnks[0],
                     self.hnd.jlc.lnks[1],
                     self.hnd.jlc.lnks[2],
                     self.hnd.jlc.lnks[4],
                     self.hnd.jlc.lnks[5]]
        self.cc.set_cdpair(from_list, into_list)
        # TODO is the following update needed?
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_plate.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_plate.jnts[-1]['gl_posq'], rotmat=self.base_plate.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'].dot(self.gripper_loc_rotmat))
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(joint_values=jnt_values)
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'].dot(self.gripper_loc_rotmat))
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name="arm"):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name="arm"):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name: str = 'hnd', jaw_width: float = 0.0):
        self.hnd.change_jaw_width(jaw_width)

    def hold(self, hnd_name, objcm, jaw_width=None):
        """
        the obj_cmodel is added as a part of the robot_s to the cd checker
        :param hnd_name:
        :param jaw_width:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jaw_width is not None:
            self.hnd_dict[hnd_name].change_jaw_width(jaw_width)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_pose_to_tcp(objcm.get_pos(), objcm.get_rotmat())
        into_list = [self.arm.lnks[0],
                     self.arm.lnks[1],
                     self.arm.lnks[2],
                     self.arm.lnks[3],
                     self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, into_list))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the obj_cmodel is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].change_jaw_width(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcp_frame=False,
                       toggle_jnt_frame=False,
                       toggle_connjnt=False,
                       name='cobotta_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_plate.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcp_frame=False,
                                       toggle_jnt_frame=toggle_jnt_frame,
                                       toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcp_frame=toggle_tcp_frame,
                                toggle_jnt_frame=toggle_jnt_frame,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frame).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=None,
                      name='xarm_shuidi_mobile_meshmodel',
                      option='full'):
        """

        :param tcp_jnt_id:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :param toggle_tcp_frame:
        :param toggle_jnt_frame:
        :param rgba:
        :param name:
        :param option: 'full', 'hand_only', 'body_only'
        :return:
        """
        meshmodel = mc.ModelCollection(name=name)
        if option == 'full' or option == 'body_only':
            self.base_plate.gen_mesh_model(tcp_jnt_id=tcp_jnt_id,
                                           tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat,
                                           toggle_tcp_frame=False,
                                           toggle_jnt_frame=toggle_jnt_frame,
                                           rgba=rgba).attach_to(meshmodel)
            self.arm.gen_meshmodel(toggle_tcp_frame=toggle_tcp_frame, toggle_jnt_frames=toggle_jnt_frame,
                                   rgba=rgba).attach_to(meshmodel)
        if option == 'full' or option == 'hand_only':
            self.hnd.gen_meshmodel(toggle_tcp_frame=False,
                                   toggle_jnt_frames=toggle_jnt_frame,
                                   rgba=rgba).attach_to(meshmodel)
        if option == 'full':
            for obj_info in self.oih_infos:
                objcm = obj_info['collision_model'].copy()
                objcm.set_pos(obj_info['gl_pos'])
                objcm.set_rotmat(obj_info['gl_rotmat'])
                if rgba is not None:
                    objcm.set_rgba(rgba)
                objcm.attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    from wrs import basis as rm, robot_sim as jl, robot_sim as cbta, robot_sim as cbtp, modeling as gm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

    gm.gen_frame().attach_to(base)
    robot_s = CobottaRIPPS(enable_cc=True)
    # robot_s.jaw_to(.02)
    # robot_s.gen_meshmodel(toggle_flange_frame=True, toggle_jnt_frames=True).attach_to(base)
    robot_s.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frame=False).attach_to(base)
    # robot_s.gen_stickmodel(toggle_flange_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot_s.show_cdprimit()
    base.run()
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # base.run()
    component_name = 'arm'
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    robot_s.fk(component_name, jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcp_frame=True)
    robot_s_meshmodel.attach_to(base)
    # robot_s.show_cdprimit()
    # robot_s.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = robot_s.is_collided()
    toc = time.time()
    print(result, toc - tic)
    base.run()
