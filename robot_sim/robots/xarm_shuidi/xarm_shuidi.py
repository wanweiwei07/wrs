import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.xarm7.xarm7 as xa
import robot_sim.end_effectors.gripper.xarm_gripper.xarm_gripper as xag
import robot_sim.robots.robot_interface as ri


class XArmShuidi(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="xarm7_shuidi_mobile", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # agv
        self.agv = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(5), name='agv')  # 1-3 x,y,theta; 4-5 dummy
        self.agv.jnts[1]['loc_pos'] = np.zeros(3)
        self.agv.jnts[1]['type'] = 'prismatic'
        self.agv.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.agv.jnts[1]['motion_rng'] = [0.0, 5.0]
        self.agv.jnts[2]['loc_pos'] = np.zeros(3)
        self.agv.jnts[2]['type'] = 'prismatic'
        self.agv.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.agv.jnts[2]['motion_rng'] = [-3.0, 3.0]
        self.agv.jnts[3]['loc_pos'] = np.zeros(3)
        self.agv.jnts[3]['loc_motionax'] = np.array([0, 0, 1])
        self.agv.jnts[3]['motion_rng'] = [-math.pi, math.pi]
        self.agv.jnts[4]['loc_pos'] = np.array([0, .0, .277])  # dummy
        self.agv.jnts[5]['loc_pos'] = np.zeros(3)  # dummy
        self.agv.jnts[6]['loc_pos'] = np.array([0, .0, .168862])
        self.agv.lnks[3]['name'] = 'agv'
        self.agv.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.agv.lnks[3]['mesh_file'] = os.path.join(this_dir, 'meshes', 'shuidi_agv.stl')
        self.agv.lnks[3]['rgba'] = [.7, .7, .7, 1.0]
        self.agv.lnks[4]['mesh_file'] = os.path.join(this_dir, 'meshes', 'battery.stl')
        self.agv.lnks[4]['rgba'] = [.35, .35, .35, 1.0]
        self.agv.lnks[5]['mesh_file'] = os.path.join(this_dir, 'meshes', 'battery_fixture.stl')
        self.agv.lnks[5]['rgba'] = [.55, .55, .55, 1.0]
        self.agv.tgtjnts = [1, 2, 3]
        self.agv.reinitialize()
        # arm
        arm_homeconf = np.zeros(7)
        arm_homeconf[1] = -math.pi / 3
        arm_homeconf[3] = math.pi / 12
        arm_homeconf[5] = -math.pi / 12
        self.arm = xa.XArm7(pos=self.agv.jnts[-1]['gl_posq'],
                            rotmat=self.agv.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf,
                            name='arm', enable_cc=False)
        # ft sensor
        self.ft_sensor = jl.JLChain(pos=self.arm.jnts[-1]['gl_posq'],
                                    rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                    homeconf=np.zeros(0), name='ft_sensor_jl')
        self.ft_sensor.jnts[1]['loc_pos'] = np.array([.0, .0, .065])
        self.ft_sensor.lnks[0]['name'] = "xs_ftsensor"
        self.ft_sensor.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.ft_sensor.lnks[0]['collision_model'] = cm.gen_stick(spos=self.ft_sensor.jnts[0]['loc_pos'],
                                                                 epos=self.ft_sensor.jnts[1]['loc_pos'],
                                                                 thickness=.075, rgba=[.2, .3, .3, 1], sections=24)
        self.ft_sensor.reinitialize()
        # gripper
        self.hnd = xag.XArmGripper(pos=self.ft_sensor.jnts[-1]['gl_posq'],
                                   rotmat=self.ft_sensor.jnts[-1]['gl_rotmatq'],
                                   name='hnd_s', enable_cc=False)
        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_rotmat = self.ft_sensor.jnts[-1]['loc_rotmat'].dot(self.hnd.jaw_center_rotmat)
        self.arm.jlc.tcp_loc_pos = self.ft_sensor.jnts[-1]['loc_pos'] + self.arm.jlc.tcp_loc_rotmat.dot(
            self.hnd.jaw_center_pos)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['ftsensor'] = self.arm
        self.manipulator_dict['hnd'] = self.arm  # specify which hand is a gripper installed to
        self.hnd_dict['arm'] = self.hnd
        self.hnd_dict['ftsensor'] = self.hnd
        self.hnd_dict['hnd'] = self.hnd
        self.ft_sensor_dict['arm'] = self.ft_sensor
        self.ft_sensor_dict['ftsensor'] = self.ft_sensor
        self.ft_sensor_dict['hnd'] = self.ft_sensor

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.agv, [3, 4])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.ft_sensor, [0])
        self.cc.add_cdlnks(self.hnd.lft_outer, [0, 1, 2])
        self.cc.add_cdlnks(self.hnd.rgt_outer, [1, 2])
        activelist = [self.agv.lnks[3],
                      self.agv.lnks[4],
                      self.arm.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.ft_sensor.lnks[0],
                      self.hnd.lft_outer.lnks[0],
                      self.hnd.lft_outer.lnks[1],
                      self.hnd.lft_outer.lnks[2],
                      self.hnd.rgt_outer.lnks[1],
                      self.hnd.rgt_outer.lnks[2]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.agv.lnks[3],
                    self.agv.lnks[4],
                    self.arm.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2]]
        intolist = [self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.ft_sensor.lnks[0],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2]]
        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.agv.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'])
        self.ft_sensor.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.ft_sensor.jnts[-1]['gl_posq'], rotmat=self.ft_sensor.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name='arm', jnt_values=np.zeros(7)):
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
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.ft_sensor_dict[component_name].fix_to(pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                                                       rotmat=self.manipulator_dict[component_name].jnts[-1][
                                                           'gl_rotmatq'])
            self.hnd_dict[component_name].fix_to(
                pos=self.ft_sensor_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.ft_sensor_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        elif component_name == 'agv':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 3:
                raise ValueError("An 1x7 npdarray must be specified to move the agv!")
            status = self.agv.fk(jnt_values)
            self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'])
            self.ft_sensor.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
            self.hnd.fix_to(pos=self.ft_sensor.jnts[-1]['gl_posq'], rotmat=self.ft_sensor.jnts[-1]['gl_rotmatq'])
            # update objects in hand
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat
            return status
        elif component_name == 'agv_arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 10:
                raise ValueError("An 1x10 npdarray must be specified to move both the agv and the arm!")
            status_agv = self.agv.fk(jnt_values)
            status_arm = self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'],
                                         jnt_values=jnt_values[3:10])
            self.ft_sensor.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
            self.hnd.fix_to(pos=self.ft_sensor.jnts[-1]['gl_posq'], rotmat=self.ft_sensor.jnts[-1]['gl_rotmatq'])
            # update objects in hand
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat
            return "succ" if status_agv == "succ" and status_arm == "succ" else "out_of_rng"
        elif component_name == 'all':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 11:
                raise ValueError("An 1x11 npdarray must be specified to move all joints!")
            status_agv = self.agv.fk(jnt_values)
            status_arm = self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'],
                                         jnt_values=jnt_values[3:10])
            self.ft_sensor.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
            self.hnd.fix_to(pos=self.ft_sensor.jnts[-1]['gl_posq'],
                            rotmat=self.ft_sensor.jnts[-1]['gl_rotmatq'],
                            motion_val=jnt_values[10])
            # update objects in hand
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat
            return "succ" if status_agv == "succ" and status_arm == "succ" else "out_of_rng"

    def get_jnt_values(self, component_name="arm"):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        elif component_name == 'agv':
            return self.agv.get_jnt_values()
        elif component_name == 'agv_arm':
            return_val = np.zeros(10)
            return_val[:3] = self.agv.get_jnt_values()
            return_val[3:10] = self.arm.get_jnt_values()
            return return_val
        elif component_name == 'all':
            return_val = np.zeros(11)
            return_val[:3] = self.agv.get_jnt_values()
            return_val[3:10] = self.arm.get_jnt_values()[:]
            return_val[10] = self.hnd.get_jawwidth()
            return return_val

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        elif component_name == 'agv':
            return self.agv.rand_conf()
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name='hnd', jawwidth=0.0):
        self.hnd_dict[hnd_name].jaw_to(jawwidth)

    def get_gl_tcp(self, manipulator_name="arm"):
        return super().get_gl_tcp(manipulator_name=manipulator_name)

    def ik(selfself,
           component_name="arm",
           tgt_pos=np.array([.7, 0, .7]),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=100,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima="accept",
           toggle_debug=False):
        return super().ik(component_name=component_name,
                          tgt_pos=tgt_pos,
                          tgt_rotmat=tgt_rotmat,
                          seed_jnt_values=seed_jnt_values,
                          max_niter=100,
                          tcp_jnt_id=tcp_jnt_id,
                          tcp_loc_pos=tcp_loc_pos,
                          tcp_loc_rotmat=tcp_loc_rotmat,
                          local_minima=local_minima,
                          toggle_debug=toggle_debug)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.agv.lnks[3],
                    self.arm.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def get_gl_pose_from_hio(self, component_name, hio_pos, hio_rotmat):
        """
        get the global pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        if component_name != 'arm':
            raise ValueError("Component name for Xarm7ShuidiRobot must be \'arm\'!")
        hnd_pos = self.arm.jnts[-1]['gl_posq']
        hnd_rotmat = self.arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3]

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.agv.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.ft_sensor.gen_stickmodel(tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.agv.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.ft_sensor.gen_meshmodel(tcp_loc_pos=tcp_loc_pos,
                                     tcp_loc_rotmat=tcp_loc_rotmat,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        self.hnd.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])

    gm.gen_frame().attach_to(base)
    xav = XArmShuidi(enable_cc=True)
    # xav.fk(component_name='all', jnt_values=np.array([0, 0, 0, 0, 0, 0, math.pi, 0, math.pi / 6, 0, 0]))
    xav.jaw_to(jawwidth=.08)
    xav_meshmodel = xav.gen_meshmodel(toggle_tcpcs=False)
    xav_meshmodel.attach_to(base)
    base.run()
    tgt_pos = np.array([.55, 0, .55])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = xav.ik(component_name='arm', tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    print(jnt_values)
    tgt_pos2 = np.array([.45, 0, .07])
    tgt_rotmat2 = rm.rotmat_from_euler(0, math.pi, 0)
    jnt_values2 = xav.ik(component_name='arm', tgt_pos=tgt_pos2, tgt_rotmat=tgt_rotmat2, seed_jnt_values=jnt_values,
                         max_niter=10000)
    print(jnt_values2)
    xav.fk(component_name='arm', jnt_values=jnt_values)
    # xss.fk(component_name='agv', jnt_values=np.array([.2, -.5, math.radians(30)]))
    # xss.show_cdprimit()
    xav.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = xav.is_collided()
    toc = time.time()
    print(result, toc - tic)

    # xav_cpy = xss.copy()
    # xav_cpy.move_to(pos=np.array([.5,.5,0]),rotmat=rm.rotmat_from_axangle([0,0,1],-math.pi/3))
    # xav_meshmodel = xav_cpy.gen_meshmodel()
    # xav_meshmodel.attach_to(base)
    # xav_cpy.show_cdprimit()
    # tic = time.time()
    # result = xav_cpy.is_collided(otherrobot_list=[xss])
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
