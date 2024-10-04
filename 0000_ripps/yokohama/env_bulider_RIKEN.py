import copy

import numpy as np
from wrs import basis as rm, robot_sim as cbtr, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd
import utils


class Env(object):
    def __init__(self, rbt_s, armname="arm"):
        self.rbt_s = rbt_s
        self.armname = armname
        self.pipette_pos = self.rbt_s.arm.jlc._loc_flange_pos - np.array([-0.008, -0.14085, 0.06075])
        self.tcp_loc_pos = self.rbt_s.arm.jlc._loc_flange_pos
        self.env_build()
        # self.load_pipette()

    def load_pipette(self):
        self.outside = cm.CollisionModel("./meshes/model_base.stl", cdprim_type="box")
        pipettemat4 = rm.homomat_from_posrot(self.pipette_pos, np.eye(3))
        eepos, eerot = self.rbt_s.get_gl_tcp(manipulator_name=self.armname)
        tcpmat4 = np.dot(rm.homomat_from_posrot(eepos, eerot), np.linalg.inv(pipettemat4))
        self.outside.set_rgba([0., 0.4, 0.8, 0.6])
        self.outside.set_pos(tcpmat4[:3, 3])
        self.outside.set_rotmat(tcpmat4[:3, :3])
        self.rbt_s.hold(self.armname, self.outside)
        self.pipette = cm.CollisionModel("./meshes/model_tip.stl", cdprim_type="box", ex_radius=0.01)
        pipettemat4 = rm.homomat_from_posrot(self.pipette_pos, np.eye(3))
        eepos, eerot = self.rbt_s.get_gl_tcp(manipulator_name=self.armname)
        tcpmat4 = np.dot(rm.homomat_from_posrot(eepos, eerot), np.linalg.inv(pipettemat4))
        self.pipette.set_rgba([0, 0.4, 0.8, 0.8])
        self.pipette.set_pos(tcpmat4[:3, 3])
        self.pipette.set_rotmat(tcpmat4[:3, :3])
        self.rbt_s.hold(self.armname, self.pipette)

    # def load_tip(self,id):
    #     self.current_tip = self.tip_dict["mcm"][id].copy()
    #     self.rbt_s.hold(self.armname, self.current_tip)

    # def release_tip(self):
    #     self.rbt_s.release(self.armname, self.current_tip)

    def env_build(self):
        table_plate = cm.gen_box(xyz_lengths=[.405, .26, .003])
        table_plate.set_pos([0.07 + 0.2025, .055, .0015])
        table_plate.set_rgba([.87, .87, .87, 1])
        table_plate.attach_to(base)

        self.frame_bottom = cm.CollisionModel("./meshes/frame_bottom.stl")
        self.frame_bottom.set_rgba([.55, .55, .55, 1])
        self.frame_bottom.attach_to(base)

        # box_tip = mcm.CollisionModel("./meshes/box_mbp.stl", ex_radius=0.005)  # 115*80
        self.tip_rack = utils.Base96("./meshes/box_mbp.stl")
        rack_pos = np.array([0.266, -0.0257, 0.00])
        # rack_pos = np.array([.0495, .0315, 0])
        self.tip_rack.set_rgba([140 / 255, 110 / 255, 170 / 255, 1])
        rack_point1 = np.array([0.2300, 0.08])
        rack_point2 = np.array([0.2300, 0.08])
        rack_point3 = np.array([0.2300, 0.08])
        self.tip_rack.set_pose_from_2d_points(rack_point1, rack_point2, rack_point3)
        # self.tip_rack.set_pos(rack_pos)
        # self.tip_rack.set_rotmat(rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2))
        # self.tip_rack.attach_to(base)

        # self.tip_cm = mcm.gen_box(np.array([0.105, 0.141, 0.022]), rgba=[0, 0, 0, 0.1])
        # self.tip_cm.set_pos(rack_pos + np.array([0, 0, 0.075]))
        # self.tip_cm.set_rotmat(rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2))
        # self.tip_cm.attach_to(base)

        dispose_box = cm.CollisionModel("./meshes/tip_rack_cover.stl", ex_radius=.007)
        dispose_box.set_rgba([140 / 255, 110 / 255, 170 / 255, 1])
        dispose_box.set_pos(pos=np.array([.12, 0.12, .003]))
        dispose_box.attach_to(base)

        self.dispose_box_cm = cm.CollisionModel("./meshes/tip_rack_cover.stl", ex_radius=.007)
        self.dispose_box_cm.set_pos(pos=np.array([.16, 0.12, .015]))

        # box_chemical = mcm.CollisionModel("./meshes/96_well.stl", ex_radius=0.)  # 128*85
        self.deep_plate = utils.Dish6("./meshes/96_well.stl")
        che_pos = np.array([0.2825, 0.08, 0.015])
        # che_pos = np.array([.0495, .0315, 0])
        self.deep_plate.set_rgba([200 / 255, 180 / 255, 140 / 255, 1])
        che_point1 = np.array([0.2300, 0.08])
        che_point2 =np.array([0.2300, 0.08])
        che_point3 =np.array([0.2300, 0.08])
        self.deep_plate.set_pose_from_2d_points(che_point1, che_point2, che_point3)
        # self.deep_plate.set_pos(che_pos)
        # self.deep_plate.set_rotmat(rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2))
        # self.deep_plate.attach_to(base)

        self.plant_pose = np.array([0.2270, 0.2405, 0.2045, np.pi / 2, 0, np.pi / 2, 5])

    def update_env(self, rack_pos=None, rack_rotmat=None, microplate_pos=None, microplate_rotmat=None):
        self.tip_rack.detach()
        self.deep_plate.detach()
        self.tip_cm.detach()
        if rack_pos is not None:
            self.tip_rack.set_pos(rack_pos)
            self.tip_cm.set_pos(rack_pos + np.array([0, 0, 0.075]))
        if rack_rotmat is not None:
            self.tip_rack.set_rotmat(rack_rotmat)
            self.tip_cm.set_rotmat(rack_rotmat)
        if microplate_pos is not None:
            self.deep_plate.set_pos(microplate_pos)
        if microplate_rotmat is not None:
            self.deep_plate.set_rotmat(microplate_rotmat)
        if (rack_pos is not None) or (rack_rotmat is not None):
            self.tip_rack.attach_to(base)
            # self.tip_cm.attach_to(base)
        if (microplate_pos is not None) or (microplate_rotmat is not None):
            self.deep_plate.attach_to(base)

    def update_env_from_file(self, file_name):
        [rack_pos, rack_euler, microplate_pos, microplate_euler] = np.loadtxt(file_name)
        [ai, aj, ak] = rack_euler
        rack_rotmat = rm.rotmat_from_euler(ai, aj, ak)
        [ai, aj, ak] = microplate_euler
        microplate_rotmat = rm.rotmat_from_euler(ai, aj, ak)
        self.update_env(rack_pos, rack_rotmat, microplate_pos, microplate_rotmat)


if __name__ == '__main__':
    base = wd.World(cam_pos=[-0.331782, 1.2348, 0.634336], lookat_pos=[-0.0605939, 0.649106, 0.311471])
    gm.gen_frame().attach_to(base)
    component_name = 'arm'
    robot_s = cbtr.CobottaRIPPS()
    env = Env(robot_s)

    # pos_list = np.loadtxt("init_poses.txt")
    # for pos in pos_list:
    #     mgm.gen_sphere(np.array([pos[0], pos[1], 0.1])).attach_to(base)
    # num_id = 6
    # env.update_env_from_file(f"./data/random_place/position{num_id}/env_mat_{num_id}.txt")
    # env.update_rbt_from_file(f"./data/random_place/position{num_id}/env_mat_rbt_{num_id}.txt")
    env.tip_rack.attach_to(base)
    env.deep_plate.attach_to(base)

    tip = cm.CollisionModel("./meshes/tip.stl")
    tip.set_rgba([200 / 255, 180 / 255, 140 / 255, 1])
    for tip_id in range(96):
        pos_rack = env.tip_rack._hole_pos_list[tip_id]
        tip_new = copy.deepcopy(tip)
        tip_new.set_pos(pos_rack)
        tip_new.attach_to(base)
        # mgm.gen_sphere(pos_rack,major_radius=0.001 + 0.0001 * tip_id).attach_to(base)
        # pos_plate = env.deep_plate._hole_pos_list[tip_id]
        # mgm.gen_sphere(pos_plate,major_radius=0.001 + 0.0001 * tip_id).attach_to(base)

    box0 = env.tip_rack
    # box0.show_cdprimit()
    box1 = env.deep_plate
    # box1.show_cdprimit()
    # env.tip_cm.show_cdprimit()

    # chemical_pos_list = env.tip_rack._hole_pos_list
    # for id, pos in enumerate(chemical_pos_list):
    #     mgm.gen_sphere(pos, major_radius=0.001 + 0.0001 * id).attach_to(base)
    #
    # plant_pos_list = env.microplate_list[0]._hole_pos_list
    # for id, pos in enumerate(plant_pos_list):
    #     mgm.gen_sphere(pos, major_radius=0.001 + 0.0002 * id).attach_to(base)

    # current_jnts = np.array([0.08685238, 0.72893128, 1.2966003, 1.90433666, 1.02620525, -0.51833472])
    eject_jnt_values = np.array([1.37435462, 0.98535585, 0.915062, 1.71130978, -1.23317083, -0.93993529])
    robot_s.fk(component_name, jnt_values=eject_jnt_values)
    robot_s.gen_meshmodel(toggle_tcpcs=False).attach_to(base)
    # robot_s.show_cdprimit()
    base.run()
