import os
import math
import numpy as np
from wrs import basis as rm, robot_sim as jl, robot_sim as mi, modeling as gm


class DobotMagician(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(3), name='cobotta', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # the last joint is passive
        new_homeconf = self._mimic_jnt_values(homeconf)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=new_homeconf, name=name)
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.024])
        self.jlc.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, -1.570796325)
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[1]['motion_range'] = [-3.14159265, 3.14159265]
        self.jlc.jnts[2]['loc_pos'] = np.array([-0.01175, 0, 0.114])
        self.jlc.jnts[2]['gl_rotmat'] = rm.rotmat_from_euler(1.570796325, -0.20128231967888244, -1.570796325)
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['motion_range'] = [0, 1.570796325]
        self.jlc.jnts[3]['loc_pos'] = np.array([0.02699, 0.13228, -0.01175])
        self.jlc.jnts[3]['gl_rotmat'] = rm.rotmat_from_euler(0, 3.14159265, -1.24211575528)
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, -1])
        self.jlc.jnts[3]['motion_range'] = [0, 1.570796325]
        self.jlc.jnts[4]['loc_pos'] = np.array([0.07431, -0.12684, 0.0])
        self.jlc.jnts[4]['gl_rotmat'] = rm.rotmat_from_euler(0, 3.14159265, 0)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])
        # self.jlc.joints[5]['loc_pos'] = np.array([-0.0328, 0.02871, 0])
        self.jlc.jnts[5]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, 3.14159265)
        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base_link.stl")
        self.jlc.lnks[0]['rgba'] = [.5, .5, .5, 1.0]
        self.jlc.lnks[1]['name'] = "link1"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "link_1.stl")
        self.jlc.lnks[1]['rgba'] = [.55, .55, .55, 1.0]
        self.jlc.lnks[2]['name'] = "link2"
        self.jlc.lnks[2]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "link_2.stl")
        self.jlc.lnks[2]['rgba'] = [.15, .15, .15, 1]
        self.jlc.lnks[3]['name'] = "link3"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "link_5.stl")
        self.jlc.lnks[3]['rgba'] = [.55, .55, .55, 1]
        self.jlc.lnks[4]['name'] = "link4"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "link_6.stl")
        self.jlc.lnks[4]['rgba'] = [.35, .35, .35, 1.0]
        self.jlc.finalize()
        # prepare parameters for analytical ik
        upper_arm_vec = self.jlc.jnts[3]['gl_posq'] - np.array([0, 0, 0.138])
        lower_arm_vec = self.jlc.jnts[4]['gl_posq'] - self.jlc.jnts[3]['gl_posq']
        self._init_j2_angle = math.asin(np.sqrt(upper_arm_vec[0]**2+upper_arm_vec[1]**2)/0.135)
        self._init_j3_angle = rm.angle_between_vectors(lower_arm_vec, -upper_arm_vec)
        print(self._init_j3_angle, self._init_j2_angle)
        # collision detection
        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4]]
        self.cc.set_active_cdlnks(activelist)

    def _mimic_jnt_values(self, jnt_values):
        """
        always set j4 to be j2+j3
        :param jnt_values:
        :return:
        author: weiwei
        date: 20220214
        """
        new_jnt_values = np.zeros(4)
        new_jnt_values[:3] = jnt_values
        new_jnt_values[3] = jnt_values[1] + jnt_values[2]
        return new_jnt_values

    def fk(self, jnt_values=None):
        jnt_values = self._mimic_jnt_values(jnt_values)
        super().fk(jnt_values=jnt_values)

    def ik(self,
           tgt_pos,
           tgt_theta=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None):
        """
        analytical ik, override the numerical one
        :param tgt_pos:
        :param tgt_theta: radian single angle, TODO
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :return:
        """
        j1_angle = math.atan(tgt_pos[0] / tgt_pos[1])
        j2_to_j4_distance = math.sqrt((tgt_pos[2] - 0.138) ** 2 + tgt_pos[0] ** 2 + tgt_pos[1] ** 2)
        print(math.acos((0.147 ** 2 + 0.135 ** 2 - j2_to_j4_distance ** 2) / (2 * 0.147 * 0.135)))
        j3_angle = math.pi/2-math.acos((0.147 ** 2 + 0.135 ** 2 - j2_to_j4_distance ** 2) / (2 * 0.147 * 0.135))
        print(j3_angle)
        j2_angle = math.pi/2-(math.acos(
            (0.135 ** 2 + j2_to_j4_distance ** 2 - 0.147 ** 2) / (2 * 0.135 * j2_to_j4_distance)) + math.atan(
            (tgt_pos[2] - 0.138) / math.sqrt(tgt_pos[0] ** 2 + tgt_pos[1] ** 2)))
        # j2_angle = math.pi/2+self._init_j2_angle-j2_angle
        return np.array([j1_angle, j2_angle, j3_angle])


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)
    robot_s = DobotMagician(enable_cc=True)
    robot_s.fk(jnt_values=np.array([math.radians(0), math.radians(0), math.radians(0)]))
    # arm_mesh = robot_s.gen_meshmodel()
    # arm_mesh.attach_to(base)
    # arm_mesh.show_cdprimit()
    # robot_s.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    goal_pos = np.array([.1,-.1,.15])
    goal_theta = .1
    goal_rotmat = rm.rotmat_from_axangle([0,0,1], goal_theta)
    solved_jnt_values = robot_s.ik(tgt_pos=goal_pos, tcp_loc_pos=np.array([0,0.1,0]))
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    robot_s.fk(jnt_values=solved_jnt_values)
    print(solved_jnt_values)
    # robot_s.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    manipulator_meshmodel = robot_s.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    # tic = time.time()
    # print(arm.is_collided())
    # toc = time.time()
    # print(toc - tic)

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # mgm.GeometricModel("./meshes/base.dae").attach_to(base)
    # mgm.gen_frame().attach_to(base)
    base.run()
