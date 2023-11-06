import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim.kinematics.jlchain as jl
import robot_sim.manipulators.rs007l.rs007l as manip
import robot_sim.robots.system_interface as ri
import robot_sim.robots.robot_interface as ai


class KHI_BLQC(ai.RobotInterface):
    """
    author: weiwei
    date: 20230826toyonaka
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="khi_g", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        # arm
        self.manipulator = manip.RS007L(pos=pos,
                                        rotmat=rotmat,
                                        homeconf=np.zeros(6),
                                        name='rs007l', enable_cc=False)
        # tool changer
        self.tc_master = jl.JLChain(pos=self.manipulator.jnts[-1]['gl_posq'],
                                    rotmat=self.manipulator.jnts[-1]['gl_rotmatq'],
                                    home_conf=np.zeros(0),
                                    name='tc_master')
        self.tc_master.lnks[0]['name'] = "tc_master"
        self.tc_master.jnts[-1]['pos_in_loc_tcp'] = np.array([0, 0, .0315])
        self.tc_master.lnks[0]['collision_model'] = cm.gen_stick(self.tc_master.jnts[0]['pos_in_loc_tcp'],
                                                                 # TODO: change to combined model, 20230806
                                                                 self.tc_master.jnts[-1]['pos_in_loc_tcp'],
                                                                 radius=0.05,
                                                                 # rgba=[.2, .2, .2, 1], rgb will be overwritten
                                                                 type='rect',
                                                                 n_sec=36)
        self.tc_master.reinitialize()
        # tool
        self.end_effector = None
        # tool center point
        self.manipulator.tcp_jnt_id = -1
        self.manipulator.tcp_loc_pos = self.tc_master.jnts[-1]['pos_in_loc_tcp']
        self.manipulator.tcp_loc_rotmat = self.tc_master.jnts[-1]['gl_rotmat']
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()

    def _update_oof(self):
        """
        oof = object on flange
        this function is to be implemented by subclasses for updating ft-sensors, tool changers, end_type-effectors, etc.
        :return:
        author: weiwei
        date: 20230807
        """
        self.tc_master.fix_to(pos=self.manipulator.jnts[-1]['gl_posq'],
                              rotmat=self.manipulator.jnts[-1]['gl_rotmatq'])
        if self.is_tool_attached:
            self.end_effector.fix_to(pos=self.tc_master.jnts[-1]['gl_posq'],
                                     rotmat=self.tc_master.jnts[-1]['gl_rotmatq'])

    @property
    def is_tool_attached(self):
        return True if self.end_effector else False

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.manipulator, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.tc_master, [0])
        activelist = [self.manipulator.lnks[0],
                      self.manipulator.lnks[1],
                      self.manipulator.lnks[2],
                      self.manipulator.lnks[3],
                      self.manipulator.lnks[4],
                      self.manipulator.lnks[5],
                      self.manipulator.lnks[6],
                      self.tc_master.lnks[0]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.manipulator.lnks[0],
                    self.manipulator.lnks[1]]
        intolist = [self.manipulator.lnks[4],
                    self.manipulator.lnks[5],
                    self.manipulator.lnks[6],
                    self.tc_master.lnks[0]]
        self.cc.set_cdpair(fromlist, intolist)

    def fk(self, jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
            raise ValueError("An 1x6 npdarray must be specified to move the arm!")
        self.manipulator.fk(jnt_values)
        self._update_oof()

    def attach_tool(self, end_effector):
        """
        :param end_effector: an instance of robot_sim.end_effectors
        :return:
        """
        if self.is_tool_attached:
            raise Exception("A tool has been attached!")
        self.manipulator.tcp_loc_pos = self.tc_master.jnts[-1]['pos_in_loc_tcp'] + end_effector.action_center_pos
        self.manipulator.tcp_loc_rotmat = self.tc_master.jnts[-1]['gl_rotmat']
        self.end_effector = end_effector

    def detach_tool(self):
        """
        there is no need to specify a tool instance since only the currently attached tool can be detached
        :return:
        """
        if not self.is_tool_attached:
            raise Exception("Cannot detach a tool since nothing is attached!")
        self.manipulator.tcp_loc_pos = self.tc_master.jnts[-1]['gl_posq']
        self.manipulator.tcp_loc_rotmat = self.tc_master.jnts[-1]['gl_rotmatq']
        self.end_effector = None

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='khi_g_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.manipulator.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                        toggle_tcpcs=toggle_tcpcs,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.tc_master.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        if self.is_tool_attached:
            self.end_effector.gen_stickmodel(toggle_tcpcs=False,
                                             toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='khi_g_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.manipulator.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcpcs=toggle_tcpcs,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)
        self.tc_master.gen_mesh_model(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      rgba=rgba).attach_to(meshmodel)
        if self.is_tool_attached:
            self.end_effector.gen_mesh_model(toggle_tcpcs=False,
                                             toggle_jntscs=toggle_jntscs,
                                             rgba=rgba).attach_to(meshmodel)
            for obj_info in self.oih_infos:
                objcm = obj_info['collision_model']
                objcm.set_pos(obj_info['gl_pos'])
                objcm.set_rotmat(obj_info['gl_rotmat'])
                objcm.copy().attach_to(meshmodel)
        return meshmodel

    def ik(self,
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           tcp_loc_pos=None,
           tcp_loc_rotmat=None):
        return self.manipulator.analytical_ik(tgt_pos,
                                              tgt_rotmat,
                                              tcp_loc_pos=tcp_loc_pos,
                                              tcp_loc_rotmat=tcp_loc_rotmat)


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import robot_sim.end_effectors.gripper.or2fg7.or2fg7 as org
    import robot_sim.end_effectors.single_contact.screw_driver.orsd.orsd as ors
    import motion.probabilistic.rrt_connect as rrtc

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)
    robot_s = KHI_BLQC(enable_cc=True)
    rrtc_planner = rrtc.RRTConnect_v2(robot_s)

    ee_g_pos = np.array([0, .8, .19])
    ee_g_rotmat = rm.rotmat_from_euler(0, np.radians(180), 0)
    ee_g = org.OR2FG7(pos=ee_g_pos,
                      rotmat=ee_g_rotmat,
                      coupling_offset_pos=np.array([0, 0, 0.0314]), enable_cc=True)
    ee_g_meshmodel = ee_g.gen_meshmodel()
    ee_g_meshmodel.attach_to(base)
    jv_attach_eeg = robot_s.ik(ee_g_pos, ee_g_rotmat)

    ee_sd_pos = np.array([.15, .8, .19])
    ee_sd_rotmat = rm.rotmat_from_euler(0, np.radians(180), np.radians(-90))
    ee_sd = ors.ORSD(pos=ee_sd_pos,
                     rotmat=ee_sd_rotmat,
                     coupling_offset_pos=np.array([0, 0, 0.0314]),
                     enable_cc=True)
    ee_sd_meshmodel = ee_g.gen_meshmodel()
    ee_sd_meshmodel.attach_to(base)
    jv_attach_eesd = robot_s.ik(ee_g_pos, ee_g_rotmat)

    goal_pos = np.array([.3, -.8, .19])
    goal_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
    jv_goal = robot_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)

    robot_path_attach_ee_g = rrtc_planner.plan(start_conf=robot_s.get_jnt_values(),
                                               goal_conf=jv_attach_eeg,
                                               obstacle_list=[],
                                               ext_dist=.1,
                                               max_time=300)
    print(len(robot_path_attach_ee_g))

    # for joint_values in robot_path:
    #     robot_s.fk(joint_values)
    #     robot_s.gen_meshmodel().attach_to(base)
    # base.run()

    robot_path = robot_path_attach_ee_g

    counter = [0]
    robot_attached_list = []
    object_attached_list = []


    def update(robot_s, ee_g, ee_sd, robot_path, counter, robot_attached_list, object_attached_list, task):
        if counter[0] >= len(robot_path):
            counter[0] = 0
        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
        jnt_values = robot_path[counter[0]]
        robot_s.fk(jnt_values)
        robot_meshmodel = robot_s.gen_mesh_model()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        counter[0] += 1
        print(counter[0])
        return task.again


    taskMgr.doMethodLater(0.01, update, 'udpate',
                          extraArgs=[robot_s, ee_g, ee_sd, robot_path, counter, robot_attached_list,
                                     object_attached_list],
                          appendTask=True)
    base.run()

    robot_s = KHI_BLQC(enable_cc=True)
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=True).attach_to(base)

    ee_g_pos = np.array([0, .8, .19])
    ee_g_rotmat = rm.rotmat_from_euler(0, np.radians(180), 0)
    ee_g = org.OR2FG7(pos=ee_g_pos,
                      rotmat=ee_g_rotmat,
                      coupling_offset_pos=np.array([0, 0, 0.0314]), enable_cc=True)
    ee_g.gen_meshmodel().attach_to(base)

    ee_sd_pos = np.array([.15, .8, .19])
    ee_sd_rotmat = rm.rotmat_from_euler(0, np.radians(180), np.radians(-90))
    ee_sd = ors.ORSD(pos=ee_sd_pos,
                     rotmat=ee_sd_rotmat,
                     coupling_offset_pos=np.array([0, 0, 0.0314]),
                     enable_cc=True)
    ee_sd.gen_meshmodel().attach_to(base)

    jnt_values = robot_s.ik(ee_g_pos, ee_g_rotmat)
    robot_s.fk(jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_s_meshmodel.attach_to(base)

    robot_s.attach_tool(ee_g)
    goal_pos = np.array([.3, -.8, .19])
    goal_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
    jnt_values = robot_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    robot_s.fk(jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_s_meshmodel.attach_to(base)
    base.run()

    tgt_pos = np.array([.45, .2, .35])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # base.run()
    component_name = 'arm'
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    robot_s.fk(component_name, jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_s_meshmodel.attach_to(base)
    # robot_s.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = robot_s.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
