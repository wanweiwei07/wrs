import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur3.ur3 as ur
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq
import robot_sim.end_effectors.gripper.robotiq85_gelsight.robotiq85_gelsight as rtq_gs
# import robot_sim.end_effectors.gripper.robotiq85_gelsight.robotiq85_gelsight_pusher as rtq_gs
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.robots.robot_interface as ri


class UR3Dual(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.lft_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(13), name='lft_body_jl')
        self.lft_body.jnts[0]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[1]['loc_pos'] = np.array([-0.0, 0.0, 0.0])
        self.lft_body.jnts[2]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[3]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[4]['loc_pos'] = np.array([-0.0, 0.0, 0.0])
        self.lft_body.jnts[5]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[6]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[7]['loc_pos'] = np.array([-0.0, 0.0, 0.0])
        self.lft_body.jnts[8]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[9]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[10]['loc_pos'] = np.array([-0.0, 0.0, 0.0])
        self.lft_body.jnts[11]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[12]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[13]['loc_pos'] = np.array([0.0, 0.0, 0.0])
        self.lft_body.jnts[14]['loc_pos'] = np.array([.0, .258485281374, 1.61051471863])
        self.lft_body.jnts[14]['loc_rotmat'] = rm.rotmat_from_euler(-3.0 * math.pi / 4.0, 0, math.pi, 'rxyz')
        # body
        self.lft_body.lnks[0]['name'] = "ur3_dual_lft_body"
        self.lft_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_base.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.lft_body.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        # columns
        self.lft_body.lnks[1]['name'] = "ur3_dual_back_rgt_column"
        self.lft_body.lnks[1]['loc_pos'] = np.array([-0.41, -0.945, 0])
        self.lft_body.lnks[1]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column2400x60x60.stl"))
        self.lft_body.lnks[2]['name'] = "ur3_dual_back_lft_column"
        self.lft_body.lnks[2]['loc_pos'] = np.array([-0.41, 0.945, 0])
        self.lft_body.lnks[2]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column2400x60x60.stl"))
        self.lft_body.lnks[3]['name'] = "ur3_dual_front_rgt_column"
        self.lft_body.lnks[3]['loc_pos'] = np.array([0.73, -0.945, 0])
        self.lft_body.lnks[3]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column2400x60x60.stl"))
        self.lft_body.lnks[4]['name'] = "ur3_dual_front_lft_column"
        self.lft_body.lnks[4]['loc_pos'] = np.array([0.73, 0.945, 0])
        self.lft_body.lnks[4]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column2400x60x60.stl"))
        # x_rows
        self.lft_body.lnks[5]['name'] = "ur3_dual_up_rgt_xrow"
        self.lft_body.lnks[5]['loc_pos'] = np.array([-0.38, -0.945, 2.37])
        self.lft_body.lnks[5]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi / 2, 0)
        self.lft_body.lnks[5]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column1080x60x60.stl"))
        self.lft_body.lnks[6]['name'] = "ur3_dual_bottom_rgt_xrow"
        self.lft_body.lnks[6]['loc_pos'] = np.array([-0.38, -0.945, 1.07])
        self.lft_body.lnks[6]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi / 2, 0)
        self.lft_body.lnks[6]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column1080x60x60.stl"))
        self.lft_body.lnks[7]['name'] = "ur3_dual_up_lft_xrow"
        self.lft_body.lnks[7]['loc_pos'] = np.array([-0.38, 0.945, 2.37])
        self.lft_body.lnks[7]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi / 2, 0)
        self.lft_body.lnks[7]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column1080x60x60.stl"))
        self.lft_body.lnks[8]['name'] = "ur3_dual_bottom_lft_xrow"
        self.lft_body.lnks[8]['loc_pos'] = np.array([-0.38, 0.945, 1.07])
        self.lft_body.lnks[8]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi / 2, 0)
        self.lft_body.lnks[8]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column1080x60x60.stl"))
        # y_rows
        self.lft_body.lnks[9]['name'] = "ur3_dual_back_up_yrow"
        self.lft_body.lnks[9]['loc_pos'] = np.array([-0.41, -0.915, 2.37])
        self.lft_body.lnks[9]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi / 2, 0, 0)
        self.lft_body.lnks[9]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column1830x60x60.stl"))
        self.lft_body.lnks[10]['name'] = "ur3_dual_back_bottom_yrow"
        self.lft_body.lnks[10]['loc_pos'] = np.array([-0.41, -0.915, 0.35])
        self.lft_body.lnks[10]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi / 2, 0, 0)
        self.lft_body.lnks[10]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column1830x60x60.stl"))
        self.lft_body.lnks[11]['name'] = "ur3_dual_front_up_yrow"
        self.lft_body.lnks[11]['loc_pos'] = np.array([0.73, -0.915, 2.37])
        self.lft_body.lnks[11]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi / 2, 0, 0)
        self.lft_body.lnks[11]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_column1830x60x60.stl"))
        # table TODO update using vision sensors
        self.lft_body.lnks[12]['name'] = "ur3_dual_table"
        self.lft_body.lnks[12]['loc_pos'] = np.array([0.36, 0.0, 1.046])
        self.lft_body.lnks[12]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi / 2)
        self.lft_body.lnks[12]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_table1820x54x800.stl"))
        self.lft_body.lnks[12]['rgba'] = [.9, .77, .52, 1.0]
        # mounter
        self.lft_body.lnks[13]['name'] = "ur3_dual_mounter"
        self.lft_body.lnks[13]['loc_pos'] = np.array([0.0, 0.0, 1.439])
        self.lft_body.lnks[13]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "mounter.stl"))
        self.lft_body.lnks[13]['rgba'] = [.55, .55, .55, 1.0]
        self.lft_body.reinitialize()
        lft_arm_homeconf = np.zeros(6)
        lft_arm_homeconf[0] = math.pi / 12.0
        lft_arm_homeconf[1] = -math.pi * 1.0 / 3.0
        lft_arm_homeconf[2] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[3] = -math.pi
        lft_arm_homeconf[4] = -math.pi / 2.0
        self.lft_arm = ur.UR3(pos=self.lft_body.jnts[-1]['gl_posq'],
                              rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                              homeconf=lft_arm_homeconf,
                              enable_cc=False)
        # lft hand ftsensor
        self.lft_ft_sensor = jl.JLChain(pos=self.lft_arm.jnts[-1]['gl_posq'],
                                        rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                        homeconf=np.zeros(0), name='lft_ft_sensor_jl')
        self.lft_ft_sensor.jnts[1]['loc_pos'] = np.array([.0, .0, .0484])
        self.lft_ft_sensor.lnks[0]['name'] = "ur3_dual_lft_ft_sensor"
        self.lft_ft_sensor.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_ft_sensor.lnks[0]['collision_model'] = cm.gen_stick(spos=self.lft_ft_sensor.jnts[0]['loc_pos'],
                                                                     epos=self.lft_ft_sensor.jnts[1]['loc_pos'],
                                                                     thickness=.067, rgba=[.2, .3, .3, 1], sections=24)
        self.lft_ft_sensor.reinitialize()
        # lft hand
        # self.lft_hnd = rtq_gs.Robotiq85GelsightPusher(pos=self.lft_ft_sensor.jnts[-1]['gl_posq'],
        #                                               rotmat=self.lft_ft_sensor.jnts[-1]['gl_rotmatq'],
        #                                               enable_cc=False)
        self.lft_hnd = rtq_gs.Robotiq85Gelsight(pos=self.lft_ft_sensor.jnts[-1]['gl_posq'],
                                                rotmat=self.lft_ft_sensor.jnts[-1]['gl_rotmatq'],
                                                enable_cc=False)
        # rigth side
        self.rgt_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='rgt_body_jl')
        self.rgt_body.jnts[1]['loc_pos'] = np.array([.0, -.258485281374, 1.61051471863])  # right from robot_s view
        self.rgt_body.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(3.0 * math.pi / 4.0, .0,
                                                                   .0)  # left from robot_s view
        self.rgt_body.lnks[0]['name'] = "ur3_dual_rgt_body"
        self.rgt_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_body.lnks[0]['mesh_file'] = None
        self.rgt_body.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.rgt_body.reinitialize()
        rgt_arm_homeconf = np.zeros(6)
        rgt_arm_homeconf[0] = -math.pi / 12.0
        rgt_arm_homeconf[1] = -math.pi * 2.0 / 3.0
        rgt_arm_homeconf[2] = math.pi * 2.0 / 3.0
        lft_arm_homeconf[3] = math.pi
        rgt_arm_homeconf[4] = math.pi / 2.0
        self.rgt_arm = ur.UR3(pos=self.rgt_body.jnts[-1]['gl_posq'],
                              rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                              homeconf=rgt_arm_homeconf,
                              enable_cc=False)
        # rgt hand ft sensor
        self.rgt_ft_sensor = jl.JLChain(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                                        rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'],
                                        homeconf=np.zeros(0), name='rgt_ft_sensor_jl')
        self.rgt_ft_sensor.jnts[1]['loc_pos'] = np.array([.0, .0, .0484])
        self.rgt_ft_sensor.lnks[0]['name'] = "ur3_dual_rgt_ft_sensor"
        self.rgt_ft_sensor.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_ft_sensor.lnks[0]['collision_model'] = cm.gen_stick(spos=self.rgt_ft_sensor.jnts[0]['loc_pos'],
                                                                     epos=self.rgt_ft_sensor.jnts[1]['loc_pos'],
                                                                     thickness=.067, rgba=[.2, .3, .3, 1], sections=24)
        self.rgt_ft_sensor.reinitialize()
        # TODO replace using copy
        self.rgt_hnd = rtq.Robotiq85(pos=self.rgt_ft_sensor.jnts[-1]['gl_posq'],
                                     rotmat=self.rgt_ft_sensor.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # tool center point
        # lft
        self.lft_arm.tcp_jnt_id = -1
        self.lft_arm.tcp_loc_rotmat = self.lft_ft_sensor.jnts[-1]['loc_rotmat'].dot(self.lft_hnd.jaw_center_rotmat)
        self.lft_arm.tcp_loc_pos = self.lft_ft_sensor.jnts[-1]['loc_pos'] + self.lft_arm.tcp_loc_rotmat.dot(
            self.lft_hnd.jaw_center_pos)
        # rgt
        self.rgt_arm.tcp_jnt_id = -1
        self.rgt_arm.tcp_loc_rotmat = self.rgt_ft_sensor.jnts[-1]['loc_rotmat'].dot(self.rgt_hnd.jaw_center_rotmat)
        self.rgt_arm.tcp_loc_pos = self.rgt_ft_sensor.jnts[-1]['loc_pos'] + self.rgt_arm.tcp_loc_rotmat.dot(
            self.rgt_hnd.jaw_center_pos)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.lft_oih_infos = []
        self.rgt_oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['rgt_arm'] = self.rgt_arm
        self.manipulator_dict['lft_arm'] = self.lft_arm
        self.manipulator_dict['rgt_hnd'] = self.rgt_arm  # specify which hand is a gripper installed to
        self.manipulator_dict['lft_hnd'] = self.lft_arm  # specify which hand is a gripper installed to
        self.manipulator_dict['rgt_ftsensor'] = self.rgt_arm  # specify which hand is a gripper installed to
        self.manipulator_dict['lft_ftsensor'] = self.lft_arm  # specify which hand is a gripper installed to
        self.hnd_dict['rgt_hnd'] = self.rgt_hnd
        self.hnd_dict['lft_hnd'] = self.lft_hnd
        self.hnd_dict['rgt_arm'] = self.rgt_hnd
        self.hnd_dict['lft_arm'] = self.lft_hnd
        self.hnd_dict['rgt_ftsensor'] = self.rgt_hnd
        self.hnd_dict['lft_ftsensor'] = self.lft_hnd
        self.ft_sensor_dict['rgt_ftsensor'] = self.rgt_ft_sensor
        self.ft_sensor_dict['lft_ftsensor'] = self.lft_ft_sensor
        self.ft_sensor_dict['rgt_arm'] = self.rgt_ft_sensor
        self.ft_sensor_dict['lft_arm'] = self.lft_ft_sensor
        self.ft_sensor_dict['rgt_hnd'] = self.rgt_ft_sensor
        self.ft_sensor_dict['lft_hnd'] = self.lft_ft_sensor

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.18, 0.0, 0.105),
                                              x=.61 + radius, y=.41 + radius, z=.105 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, 0.4445),
                                              x=.321 + radius, y=.321 + radius, z=.2345 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.0, 0.0, 0.8895),
                                              x=.05 + radius, y=.05 + radius, z=.6795 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_c3 = CollisionBox(Point3(0.0, 0.0, 1.619),
                                              x=.1 + radius, y=.275 + radius, z=.05 + radius)
        collision_node.addSolid(collision_primitive_c3)
        collision_primitive_l0 = CollisionBox(Point3(0.0, 0.300, 1.669),
                                              x=.1 + radius, y=.029 + radius, z=.021 + radius)
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0.0, -0.300, 1.669),
                                              x=.1 + radius, y=.029 + radius, z=.021 + radius)
        collision_node.addSolid(collision_primitive_r0)
        return collision_node

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.lft_body, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.cc.add_cdlnks(self.lft_arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.lft_ft_sensor, [0])
        self.cc.add_cdlnks(self.lft_hnd.lft_outer, [0, 1, 2, 3])
        self.cc.add_cdlnks(self.lft_hnd.rgt_outer, [1, 2, 3])
        self.cc.add_cdlnks(self.rgt_arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.rgt_ft_sensor, [0])
        self.cc.add_cdlnks(self.rgt_hnd.lft_outer, [0, 1, 2, 3, 4])
        self.cc.add_cdlnks(self.rgt_hnd.rgt_outer, [1, 2, 3, 4])
        # lnks used for cd with external stationary objects
        activelist = [self.lft_arm.lnks[2],
                      self.lft_arm.lnks[3],
                      self.lft_arm.lnks[4],
                      self.lft_arm.lnks[5],
                      self.lft_arm.lnks[6],
                      self.lft_ft_sensor.lnks[0],
                      self.lft_hnd.lft_outer.lnks[0],
                      self.lft_hnd.lft_outer.lnks[1],
                      self.lft_hnd.lft_outer.lnks[2],
                      self.lft_hnd.lft_outer.lnks[3],
                      self.lft_hnd.rgt_outer.lnks[1],
                      self.lft_hnd.rgt_outer.lnks[2],
                      self.lft_hnd.rgt_outer.lnks[3],
                      self.rgt_arm.lnks[2],
                      self.rgt_arm.lnks[3],
                      self.rgt_arm.lnks[4],
                      self.rgt_arm.lnks[5],
                      self.rgt_arm.lnks[6],
                      self.rgt_ft_sensor.lnks[0],
                      self.rgt_hnd.lft_outer.lnks[0],
                      self.rgt_hnd.lft_outer.lnks[1],
                      self.rgt_hnd.lft_outer.lnks[2],
                      self.rgt_hnd.lft_outer.lnks[3],
                      self.rgt_hnd.lft_outer.lnks[4],
                      self.rgt_hnd.rgt_outer.lnks[1],
                      self.rgt_hnd.rgt_outer.lnks[2],
                      self.rgt_hnd.rgt_outer.lnks[3],
                      self.rgt_hnd.rgt_outer.lnks[4]]
        self.cc.set_active_cdlnks(activelist)
        # lnks used for arm-body collision detection
        fromlist = [self.lft_body.lnks[0],  # body
                    self.lft_body.lnks[1],  # back-rgt column
                    self.lft_body.lnks[2],  # back-lft column
                    self.lft_body.lnks[3],  # head-rgt row
                    self.lft_body.lnks[4],  # head-lft row
                    self.lft_body.lnks[5],  # up right x_row
                    self.lft_body.lnks[6],  # bottom right x_row
                    self.lft_body.lnks[7],  # up left row
                    self.lft_body.lnks[8],  # bottom left row
                    self.lft_body.lnks[9],  # back up y_row
                    self.lft_body.lnks[10],  # back bottom y_row
                    self.lft_body.lnks[11],  # head up y_row
                    self.lft_body.lnks[12],  # table
                    self.lft_arm.lnks[1],
                    self.rgt_arm.lnks[1]]
        intolist = [self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_ft_sensor.lnks[0],
                    self.lft_hnd.lft_outer.lnks[0],
                    self.lft_hnd.lft_outer.lnks[1],
                    self.lft_hnd.lft_outer.lnks[2],
                    self.lft_hnd.lft_outer.lnks[3],
                    self.lft_hnd.rgt_outer.lnks[1],
                    self.lft_hnd.rgt_outer.lnks[2],
                    self.lft_hnd.rgt_outer.lnks[3],
                    self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_ft_sensor.lnks[0],
                    self.rgt_hnd.lft_outer.lnks[0],
                    self.rgt_hnd.lft_outer.lnks[1],
                    self.rgt_hnd.lft_outer.lnks[2],
                    self.rgt_hnd.lft_outer.lnks[3],
                    self.rgt_hnd.lft_outer.lnks[4],
                    self.rgt_hnd.rgt_outer.lnks[1],
                    self.rgt_hnd.rgt_outer.lnks[2],
                    self.rgt_hnd.rgt_outer.lnks[3],
                    self.rgt_hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)
        # lnks used for arm-body collision detection -- extra
        fromlist = [self.lft_body.lnks[0]]  # body
        intolist = [self.lft_arm.lnks[2],
                    self.rgt_arm.lnks[2]]
        self.cc.set_cdpair(fromlist, intolist)
        # arm-arm collision
        fromlist = [self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_ft_sensor.lnks[0],
                    self.lft_hnd.lft_outer.lnks[0],
                    self.lft_hnd.lft_outer.lnks[1],
                    self.lft_hnd.lft_outer.lnks[2],
                    self.lft_hnd.lft_outer.lnks[3],
                    self.lft_hnd.rgt_outer.lnks[1],
                    self.lft_hnd.rgt_outer.lnks[2],
                    self.lft_hnd.rgt_outer.lnks[3]]
        intolist = [self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_ft_sensor.lnks[0],
                    self.rgt_hnd.lft_outer.lnks[0],
                    self.rgt_hnd.lft_outer.lnks[1],
                    self.rgt_hnd.lft_outer.lnks[2],
                    self.rgt_hnd.lft_outer.lnks[3],
                    self.rgt_hnd.lft_outer.lnks[4],
                    self.rgt_hnd.rgt_outer.lnks[1],
                    self.rgt_hnd.rgt_outer.lnks[2],
                    self.rgt_hnd.rgt_outer.lnks[3],
                    self.rgt_hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)

    def get_hnd_on_manipulator(self, manipulator_name):
        if manipulator_name == 'rgt_arm':
            return self.rgt_hnd
        elif manipulator_name == 'lft_arm':
            return self.lft_hnd
        else:
            raise ValueError("The given jlc does not have a hand!")

    def fix_to(self, pos, rotmat):
        super().fix_to(pos, rotmat)
        self.pos = pos
        self.rotmat = rotmat
        self.lft_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'], rotmat=self.lft_body.jnts[-1]['gl_rotmatq'])
        self.lft_ft_sensor.fix_to(pos=self.lft_arm.jnts[-1]['gl_posq'], rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])
        self.lft_hnd.fix_to(pos=self.lft_ft_sensor.jnts[-1]['gl_posq'],
                            rotmat=self.lft_ft_sensor.jnts[-1]['gl_rotmatq'])
        self.rgt_body.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'], rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        self.rgt_ft_sensor.fix_to(pos=self.rgt_arm.jnts[-1]['gl_posq'], rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])
        self.rgt_hnd.fix_to(pos=self.rgt_ft_sensor.jnts[-1]['gl_posq'],
                            rotmat=self.rgt_ft_sensor.jnts[-1]['gl_rotmatq'])

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: nparray 1x6 or 1x12 depending on component_names
        :hnd_name 'lft_arm', 'rgt_arm', 'both_arm'
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka, 20210403osaka
        """

        def update_oih(component_name='rgt_arm'):
            # inline function for update objects in hand
            if component_name == 'rgt_arm':
                oih_info_list = self.rgt_oih_infos
            elif component_name == 'lft_arm':
                oih_info_list = self.lft_oih_infos
            for obj_info in oih_info_list:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.ft_sensor_dict[component_name].fix_to(pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                                                       rotmat=self.manipulator_dict[component_name].jnts[-1][
                                                           'gl_rotmatq'])
            self.get_hnd_on_manipulator(component_name).fix_to(
                pos=self.ft_sensor_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.ft_sensor_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        super().fk(component_name, jnt_values)
        # examine length
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move a single arm!")
            return update_component(component_name, jnt_values)
        elif component_name == 'both_arm':
            if (jnt_values.size != 12):
                raise ValueError("A 1x12 npdarrays must be specified to move both arm!")
            status_lft = update_component('lft_arm', jnt_values[0:6])
            status_rgt = update_component('rgt_arm', jnt_values[6:12])
            return "succ" if status_lft == "succ" and status_rgt == "succ" else "out_of_rng"
        elif component_name == 'all':
            raise NotImplementedError
        else:
            raise ValueError("The given component name is not available!")

    def jaw_to(self, hnd_name, jaw_width):
        self.hnd_dict[hnd_name].jaw_to(jaw_width=jaw_width)
        # update arm tcp
        self.lft_arm.tcp_loc_pos = self.lft_ft_sensor.jnts[-1]['loc_pos'] + self.lft_hnd.jaw_center_pos
        self.rgt_arm.tcp_loc_pos = self.rgt_ft_sensor.jnts[-1]['loc_pos'] + self.rgt_hnd.jaw_center_pos

    def rand_conf(self, component_name):
        """
        override robot_interface.rand_conf
        :param component_name:
        :return:
        author: weiwei
        date: 20210406
        """
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            return super().rand_conf(component_name)
        elif component_name == 'both_arm':
            return np.hstack((super().rand_conf('lft_arm'), super().rand_conf('rgt_arm')))
        else:
            raise NotImplementedError

    def hold(self, objcm, jaw_width=None, hnd_name='lft_hnd'):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jaw_width:
        :param objcm:
        :return:
        """
        if hnd_name == 'lft_hnd':
            rel_pos, rel_rotmat = self.lft_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.lft_body.lnks[0],  # body
                        self.lft_body.lnks[1],  # back-rgt column
                        self.lft_body.lnks[2],  # back-lft column
                        self.lft_body.lnks[3],  # head-rgt row
                        self.lft_body.lnks[4],  # head-lft row
                        self.lft_body.lnks[5],  # up right x_row
                        self.lft_body.lnks[6],  # bottom right x_row
                        self.lft_body.lnks[7],  # up left row
                        self.lft_body.lnks[8],  # bottom left row
                        self.lft_body.lnks[9],  # back up y_row
                        self.lft_body.lnks[10],  # back bottom y_row
                        self.lft_body.lnks[11],  # head up y_row
                        self.lft_body.lnks[12],  # table
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_ft_sensor.lnks[0],
                        self.rgt_hnd.rgt_outer.lnks[1],
                        self.rgt_hnd.rgt_outer.lnks[2],
                        self.rgt_hnd.rgt_outer.lnks[3],
                        self.rgt_hnd.rgt_outer.lnks[4]]
            self.lft_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        elif hnd_name == 'rgt_hnd':
            rel_pos, rel_rotmat = self.rgt_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.lft_body.lnks[0],  # body
                        self.lft_body.lnks[1],  # back-rgt column
                        self.lft_body.lnks[2],  # back-lft column
                        self.lft_body.lnks[3],  # head-rgt row
                        self.lft_body.lnks[4],  # head-lft row
                        self.lft_body.lnks[5],  # up right x_row
                        self.lft_body.lnks[6],  # bottom right x_row
                        self.lft_body.lnks[7],  # up left row
                        self.lft_body.lnks[8],  # bottom left row
                        self.lft_body.lnks[9],  # back up y_row
                        self.lft_body.lnks[10],  # back bottom y_row
                        self.lft_body.lnks[11],  # head up y_row
                        self.lft_body.lnks[12],  # table
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_ft_sensor.lnks[0],
                        self.lft_hnd.rgt_outer.lnks[1],
                        self.lft_hnd.rgt_outer.lnks[2],
                        self.lft_hnd.rgt_outer.lnks[3],
                        self.lft_hnd.rgt_outer.lnks[4],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4]]
            self.rgt_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        return rel_pos, rel_rotmat

    def get_loc_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
        """
        get the loc pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        if component_name == 'lft_arm':
            arm = self.lft_arm
        elif component_name == 'rgt_arm':
            arm = self.rgt_arm
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return self.cvt_gl_to_loc_tcp(component_name, gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3])

    def get_gl_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
        """
        get the loc pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        if component_name == 'lft_arm':
            arm = self.lft_arm
        elif component_name == 'rgt_arm':
            arm = self.rgt_arm
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3]

    def get_oih_cm_list(self, hnd_name='lft_hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        return_list = []
        for obj_info in oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def get_oih_glhomomat_list(self, hnd_name='lft_hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        return_list = []
        for obj_info in oih_infos:
            return_list.append(rm.homomat_from_posrot(obj_info['gl_pos']), obj_info['gl_rotmat'])
        return return_list

    def get_oih_relhomomat(self, objcm, hnd_name='lft_hnd'):
        """
        TODO: useless? 20210320
        oih = object in hand list
        :param objcm
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        if hnd_name == 'lft_hnd':
            oih_info_list = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_info_list = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        for obj_info in oih_info_list:
            if obj_info['collision_model'] is objcm:
                return rm.homomat_from_posrot(obj_info['rel_pos']), obj_info['rel_rotmat']

    def release(self, hnd_name, objcm, jaw_width=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jaw_width:
        :param objcm:
        :param hnd_name:
        :return:
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                oih_infos.remove(obj_info)
                break

    def release_all(self, jaw_width=None, hnd_name='lft_hnd'):
        """
        release all objects from the specified hand
        :param jaw_width:
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210125
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            self.cc.delete_cdobj(obj_info)
        oih_infos.clear()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='ur3dual'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_ft_sensor.gen_stickmodel(toggle_tcpcs=toggle_tcpcs,
                                          toggle_jntscs=toggle_jntscs,
                                          toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_ft_sensor.gen_stickmodel(toggle_tcpcs=toggle_tcpcs,
                                          toggle_jntscs=toggle_jntscs,
                                          toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.lft_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.lft_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.lft_ft_sensor.gen_meshmodel(toggle_tcpcs=toggle_tcpcs,
                                         toggle_jntscs=toggle_jntscs,
                                         rgba=rgba).attach_to(meshmodel)
        self.rgt_ft_sensor.gen_meshmodel(toggle_tcpcs=toggle_tcpcs,
                                         toggle_jntscs=toggle_jntscs,
                                         rgba=rgba).attach_to(meshmodel)
        for obj_info in self.lft_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        for obj_info in self.rgt_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 3], lookat_pos=[0, 0, 1])
    gm.gen_frame().attach_to(base)
    u3d = UR3Dual()
    # u3d.show_cdprimit()
    # u3d.fk(.85)
    u3d_meshmodel = u3d.gen_meshmodel(toggle_tcpcs=True)
    u3d_meshmodel.attach_to(base)
    # u3d.gen_stickmodel().attach_to(base)
    base.run()
