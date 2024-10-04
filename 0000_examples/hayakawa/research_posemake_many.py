#### Program for my research
#### generate many pose of graspping in both hands(for pulling rope) and pushing in left hand(for pushing object)
import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as rtq85, robot_sim as rbts, modeling as gm
import math
import numpy as np
from pyquaternion import Quaternion
import copy


class PoseMaker(object):

    def __init__(self):
        self.rtq85 = rtq85.Robotiq85()
        self.rbt = rbts.UR3Dual()
        # # import manipulation.grip.robotiq85.robotiq85 as rtq85
        # # self.base = pandactrl.World(camp=[5000, 3000, 3000], lookatp=[0, 0, 700])
        # self.hndfa = rtq85.Robotiq85Factory()
        # self.rtq85 = self.hndfa.genHand()
        # self.rgthnd = self.hndfa.genHand()
        # self.lfthnd = self.hndfa.genHand()
        # self.robot_s = robot_s.Ur3DualRobot(self.rgthnd, self.lfthnd)
        # self.rbtmg = robotmesh.Ur3DualMesh()
        # # self.obj = mcm.CollisionModel(initializer="./objects/research_box.stl")

    def lftgrasppose(self):
        lftdirstart = 250
        lftverticalstart = lftdirstart + 90
        handrotrange = 5
        predefined_grasps_lft = []
        handdirect_lft = []
        loc_z = np.array([0, -1, 0])
        rotmat = rm.rotmat_from_axangle(loc_z, math.radians(-90))
        predefined_grasps_lft.append(
            self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(np.array([1, 0, 0])),
                                          jaw_width=self.rtq85.jaw_range[1]))
        handdirect_lft.append([0, -1, 0])
        for i in range(8):
            loc_z = np.array([math.cos(math.radians(lftdirstart + i * handrotrange)),
                              math.sin(math.radians(lftdirstart + i * handrotrange)), -.2])
            rotmat = rm.rotmat_from_axangle(loc_z, math.radians(-90))
            predefined_grasps_lft.append(
                self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(
                    np.array([math.cos(math.radians(lftverticalstart + i * handrotrange)),
                              math.sin(math.radians(lftverticalstart + i * handrotrange)),
                              0])), jaw_width=self.rtq85.jaw_range[0]))
            handdirect_lft.append([math.cos(math.radians(lftdirstart + i * handrotrange)),
                                   math.sin(math.radians(lftdirstart + i * handrotrange)), -.2])

        for i in range(8):
            loc_z = np.array([math.cos(math.radians(lftdirstart + i * handrotrange)),
                              math.sin(math.radians(lftdirstart + i * handrotrange)), 0])
            rotmat = rm.rotmat_from_axangle(loc_z, math.radians(-90))
            predefined_grasps_lft.append(
                self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(
                    np.array([math.cos(math.radians(lftverticalstart + i * handrotrange)),
                              math.sin(math.radians(lftverticalstart + i * handrotrange)),
                              0])), jaw_width=self.rtq85.jaw_range[0]))
            handdirect_lft.append([math.cos(math.radians(lftdirstart + i * handrotrange)),
                                   math.sin(math.radians(lftdirstart + i * handrotrange)), 0])

        for i in range(8):
            loc_z = np.array([math.cos(math.radians(lftdirstart + i * handrotrange)),
                              math.sin(math.radians(lftdirstart + i * handrotrange)), 0])
            rotmat = rm.rotmat_from_axangle(loc_z, math.radians(-90))
            predefined_grasps_lft.append(
                self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(
                    np.array([math.cos(math.radians(lftverticalstart + i * handrotrange)),
                              math.sin(math.radians(lftverticalstart + i * handrotrange)),
                              0])), jaw_width=self.rtq85.jaw_range[0]))
            handdirect_lft.append([math.cos(math.radians(lftdirstart + i * handrotrange)),
                                   math.sin(math.radians(lftdirstart + i * handrotrange)), .2])
        return predefined_grasps_lft, handdirect_lft

    def rgtgrasppose(self):
        rgtdirstart = 90  # hand approach motion_vec
        rgtverticalstart = rgtdirstart - 90  # thumb motion_vec
        handrotrange = 5
        predefined_grasps_rgt = []
        handdirect_rgt = []
        loc_z = np.array([0, 1, 0])
        rotmat = rm.rotmat_from_axangle(loc_z, math.radians(90))
        predefined_grasps_rgt.append(
            self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(np.array([1, 0, 0])),
                                          jaw_width=self.rtq85.jaw_range[1]))
        handdirect_rgt.append([0, 1, 0])
        for i in range(4):
            loc_z = np.array([math.cos(math.radians(rgtdirstart - i * handrotrange)),
                              math.sin(math.radians(rgtdirstart - i * handrotrange)), -.1])
            rotmat = rm.rotmat_from_axangle(loc_z, math.radians(90))
            predefined_grasps_rgt.append(
                self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(
                    np.array([math.cos(math.radians(rgtverticalstart - i * handrotrange)),
                              math.sin(math.radians(rgtverticalstart - i * handrotrange)),
                              0])), jaw_width=self.rtq85.jaw_range[0]))
            handdirect_rgt.append([math.cos(math.radians(rgtdirstart - i * handrotrange)),
                                   math.sin(math.radians(rgtdirstart - i * handrotrange)), -.1])
        for i in range(4):
            loc_z = np.array([math.cos(math.radians(rgtdirstart - i * handrotrange)),
                              math.sin(math.radians(rgtdirstart - i * handrotrange)), 0])
            rotmat = rm.rotmat_from_axangle(loc_z, math.radians(90))
            predefined_grasps_rgt.append(
                self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(
                    np.array([math.cos(math.radians(rgtverticalstart - i * handrotrange)),
                              math.sin(math.radians(rgtverticalstart - i * handrotrange)),
                              0])), jaw_width=self.rtq85.jaw_range[0]))
            handdirect_rgt.append([math.cos(math.radians(rgtdirstart - i * handrotrange)),
                                   math.sin(math.radians(rgtdirstart - i * handrotrange)), 0])

            for i in range(4):
                loc_z = np.array([math.cos(math.radians(rgtdirstart - i * handrotrange)),
                                  math.sin(math.radians(rgtdirstart - i * handrotrange)), .1])
                rotmat = rm.rotmat_from_axangle(loc_z, math.radians(90))
                predefined_grasps_rgt.append(
                    self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), loc_z, rotmat.dot(
                        np.array([math.cos(math.radians(rgtverticalstart - i * handrotrange)),
                                  math.sin(math.radians(rgtverticalstart - i * handrotrange)),
                                  0])), jaw_width=self.rtq85.jaw_range[0]))
            handdirect_rgt.append([math.cos(math.radians(rgtdirstart - i * handrotrange)),
                                   math.sin(math.radians(rgtdirstart - i * handrotrange)), .1])
        return predefined_grasps_rgt, handdirect_rgt

    def pushpose(self, axisvec, pushpoint, toggle_debug=False):
        pushposelist = []
        pushpose_rotmatlist = []
        zaxis = np.array([0, 0, 1])
        axisvec_norm = np.linalg.norm(axisvec)  ## 円錐の中心のベクトル
        theta = 5
        degree = 90  ## 30
        handrotate = 180  ## 30
        thetamax = 30  ## 60
        thetarange = int(thetamax / theta)
        degreerange = int(360 / degree)
        handrotaterange = int(360 / handrotate)
        for i in range(thetarange):
            referencevec = axisvec + (axisvec_norm * math.tan(math.radians(theta * (i + 1)))) * zaxis
            referencepoint = pushpoint + referencevec
            ## プッシングする点からの相対座標に変換して、クォータニオンを求める
            q_refvec = Quaternion(0, referencepoint[0] - pushpoint[0], referencepoint[1] - pushpoint[1],
                                  referencepoint[2] - pushpoint[2])
            for j in range(degreerange):
                q_axis = Quaternion(axis=rm.unit_vector(axisvec), degrees=degree * (j + 1))  ## 回転クォータニオン
                q_new = q_axis * q_refvec * q_axis.inverse
                ## 絶対座標に戻す
                point = np.array([q_new[1] + pushpoint[0], q_new[2] + pushpoint[1], q_new[3] + pushpoint[2]])
                # base.pggen.plotSphere(base.render, pos=point, major_radius=10, rgba=[0,0,1,1])
                handdir = pushpoint - point
                handdir_projection = copy.copy(handdir)  ## xy平面への正射影
                handdir_projection[2] = 0
                handdir_projection = rm.unit_vector(handdir_projection)
                ## ハンド座標系の各要素となるベクトル
                handdir = rm.unit_vector(handdir)  ## z
                thumb_verticalvec = np.cross(zaxis, handdir_projection)  ## x
                zaxis_hand = np.cross(handdir, thumb_verticalvec)  ## y
                # pushposelist.append(self.rtq85.approachAt(5,5,5,thumb_verticalvec[0], thumb_verticalvec[1], thumb_verticalvec[2],
                #                                       handdir[0], handdir[1], handdir[2], ee_values=0))
                ## ハンドの方向を軸に-90度ずつ回転した姿勢を生成
                for k in range(handrotaterange):
                    handrotmat = np.empty((0, 3))
                    ## test
                    # handrotmat = np.append(handrotmat, np.array([handdir]), axis=0)
                    # handrotmat = np.append(handrotmat, np.array([thumb_verticalvec]), axis=0)
                    # handrotmat = np.append(handrotmat, np.array([zaxis_hand]), axis=0)
                    handrotmat = np.append(handrotmat, np.array([thumb_verticalvec]), axis=0)
                    handrotmat = np.append(handrotmat, np.array([zaxis_hand]), axis=0)
                    handrotmat = np.append(handrotmat, np.array([handdir]), axis=0)
                    handrotmat = handrotmat.T
                    handrotmat = np.dot(rm.rotmat_from_axangle(handrotmat[:, 2], - math.radians(handrotate * k)),
                                        handrotmat)
                    pushposelist.append(
                        self.rtq85.grip_at_by_twovecs(np.array([.005, .005, .005]), np.array(
                            [handrotmat[:, 2][0], handrotmat[:, 2][1], handrotmat[:, 2][2]]), np.array(
                            [handrotmat[:, 0][0], handrotmat[:, 0][1], handrotmat[:, 0][2]]),
                                                      jaw_width=self.rtq85.jaw_range[0]))
                    if toggle_debug:
                        self.rtq85.copy().gen_mesh_model().attach_to(base)
                    pushpose_rotmatlist.append(handrotmat)
        return pushpose_rotmatlist


if __name__ == "__main__":
    base = wd.World(cam_pos=[.3, 0, .3], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    ## プッシング時の姿勢を生成
    axisvec = np.array([0, 1, 0])
    pushpoint = np.array([0, 0, 0])
    p_maker = PoseMaker()
    rotmatlist = p_maker.pushpose(axisvec, pushpoint, toggle_debug=True)
    print("len", len(rotmatlist))
    base.run()
