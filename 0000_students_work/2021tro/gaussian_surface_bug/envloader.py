import os

import numpy as np

import config
# import localenv.item as item
import item
# import manipulation.grip.robotiqhe.robotiqhe as rtqhe
# import pandaplotutils.pandactrl as pc
# import robotcon.ur3edual as ur3ex
# import robotsim.robots.dualarm.ur3edual.ur3edual as ur3edual
# import robotsim.robots.dualarm.ur3edual.ur3edualball as robotball
# import robotsim.robots.dualarm.ur3edual.ur3edualmesh as robotmesh
# import trimesh.sample as ts
from wrs import basis as rm, modeling as cm


# import utiltools.thirdparty.o3dhelper as o3d_helper
# from trimesh.primitives import Box


class Env_wrs(object):
    def __init__(self, boundingradius=10.0, betransparent=False):
        """
        load obstacles model
        separated by category

        :param base:
        author: weiwei
        date: 20181205
        """

        self.__this_dir, _ = os.path.split(__file__)

        # table
        self.__tablepath = os.path.join(self.__this_dir, "../obstacles", "ur3edtable.stl")
        self.__tablecm = cm.CollisionModel(self.__tablepath, ex_radius=boundingradius, betransparency=betransparent)
        self.__tablecm.setPos(180, 0, 0)
        self.__tablecm.setColor(.32, .32, .3, 1.0)

        self.__battached = False
        self.__changableobslist = []

    def reparentTo(self, nodepath):
        if not self.__battached:
            # table
            self.__tablecm.reparentTo(nodepath)
            # housing
            self.__battached = True

    def loadobj(self, name):
        self.__objpath = os.path.join(self.__this_dir, "../../0000_srl/objects", name)
        self.__objcm = cm.CollisionModel(self.__objpath, type="ball")
        return self.__objcm

    def getstationaryobslist(self):
        """
        generate the collision model for stationary obstacles

        :return:

        author: weiwei
        date: 20180811
        """

        stationaryobslist = [self.__tablecm]
        return stationaryobslist

    def getchangableobslist(self):
        """
        get the collision model for changable obstacles

        :return:

        author: weiwei
        date: 20190313
        """
        return self.__changableobslist

    def addchangableobs(self, nodepath, objcm, pos, rot):
        """

        :param objcm: CollisionModel
        :param pos: nparray 1x3
        :param rot: nparray 3x3
        :return:

        author: weiwei
        date: 20190313
        """

        self.__changableobslist.append(objcm)
        objcm.reparentTo(nodepath)
        objcm.setMat(base.pg.npToMat4(rot, pos))

    def addchangableobscm(self, objcm):
        self.__changableobslist.append(objcm)

    def removechangableobs(self, objcm):
        if objcm in self.__changableobslist:
            objcm.remove()


# def loadEnv_wrs(camp=[4000, 0, 1700], lookat_pos=[0, 0, 1000]):
#     # Table width: 120
#     # Table long: 1080
#
#     base = pc.World(camp=camp, lookat_pos=lookat_pos)
#     env = Env_wrs(boundingradius=7.0)
#     env.reparentTo(base.render)
#     # obstacle = mcm.CollisionModel(objinit=Box(box_extents=[30, 298, 194]))
#     # env.addchangableobs(base.render, obstacle, [1080 + 30 / 2, -600 + 200 + 298 / 2 - 20, 780 + 97], np.eye(3))
#     # obstacle = mcm.CollisionModel(objinit=Box(box_extents=[60, 298, 15]))
#     # env.addchangableobs(base.render, obstacle, [1080, -600 + 200 + 298 / 2 - 20, 780 + 130], np.eye(3))
#     # obstacle = mcm.CollisionModel(objinit=Box(box_extents=[60, 40, 194]))
#     # env.addchangableobs(base.render, obstacle, [1080, -600 + 200 + 105 + 298 / 2, 780 + 97], np.eye(3))
#     # phonix
#     phoxicam = mcm.CollisionModel(objinit=Box(box_extents=[550, 200, 100]))
#     phoxicam.setColor(.32, .32, .3, 1)
#     env.addchangableobs(base.render, phoxicam, [650, 0, 1760], np.eye(3))
#     # desk
#     desk = mcm.CollisionModel(objinit=Box(box_extents=[1080, 400, 760]))
#     desk.setColor(0.7, 0.7, 0.7, 1)
#     env.addchangableobs(base.render, desk, [540, 800, 380], np.eye(3))
#     # penframe
#     # penframe = mcm.CollisionModel(objinit=Box(box_extents=[200, 320, 100]))
#     # penframe.setColor(0.7, 0.7, 0.7, 0.8)
#     # env.addchangableobs(base.render, penframe, [1080 - 300 + 100, 600 - 175, 795], np.eye(3))
#
#     return base, env


def __pcd_trans(pcd, amat):
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(amat, homopcd).T
    return realpcd[:, :3]


# def loadUr3e(showrbt=False):
#     hndfa = rtqhe.HandFactory()
#     rgthnd = hndfa.genHand(ftsensoroffset=36)
#     lfthnd = hndfa.genHand(ftsensoroffset=36)
#
#     rbtball = robotball.Ur3EDualBall()
#     rbt = ur3edual.Ur3EDualRobot(rgthnd, lfthnd)
#
#     rbt.opengripper(armname="lft")
#     rbt.opengripper(armname="rgt")
#     rbtmg = robotmesh.Ur3EDualMesh()
#     rbt.goinitpose()
#     if showrbt:
#         rbtmg.genmnp(rbt, toggleendcoord=False).reparentTo(base.render)
#
#     return rbt, rbtmg, rbtball


# def loadUr3ex(rbt):
#     rbtx = ur3ex.Ur3EDualUrx(rbt)
#
#     return rbtx


def loadObj(f_name, pos=(0, 0, 0), rot=(0, 0, 0), color=(1, 1, 1), transparency=0.5):
    obj = cm.CollisionModel(objinit=os.path.join(config.ROOT, "obstacles", f_name))
    obj.setPos(pos[0], pos[1], pos[2])
    obj.setColor(color[0], color[1], color[2], transparency)
    obj.setRPY(rot[0], rot[1], rot[2])

    return obj


def loadObjpcd(f_name, pos=(0, 0, 0), rot=(0, 0, 0), sample_num=100000, toggledebug=False):
    obj = cm.CollisionModel(objinit=os.path.join(config.ROOT, "obstacles", f_name))
    rotmat4 = np.zeros([4, 4])
    rotmat4[:3, :3] = rm.rotmat_from_euler(rot[0], rot[1], rot[2], axes="sxyz")
    rotmat4[:3, 3] = pos
    obj_surface = np.asarray(obj.sample_surface(n_samples=sample_num))
    # obj_surface = obj_surface[obj_surface[:, 2] > 2]
    obj_surface_real = __pcd_trans(obj_surface, rotmat4)
    if toggledebug:
        # obj_surface = o3d_helper.nparray2o3dpcd(copy.deepcopy(obj_surface))
        # obj_surface.paint_uniform_color([1, 0.706, 0])
        # o3d.visualization.draw_geometries([obj_surface], window_name='loadObjpcd')
        # pcddnp = base.pg.genpointcloudnp(obj_surface_real)
        # pcddnp.reparentTo(base.render)
        pass
    return obj_surface_real


def update(rbtmnp, motioncounter, robot, path, armname, robotmesh, robotball, task):
    if motioncounter[0] < len(path):
        if rbtmnp[0] is not None:
            rbtmnp[0].detachNode()
        pose = path[motioncounter[0]]
        robot.movearmfk(pose, armname)
        rbtmnp[0] = robotmesh.genmnp(robot)
        rbtmnp[0].reparentTo(base.render)
        motioncounter[0] += 1
    else:
        motioncounter[0] = 0
    return task.again


def loadObjitem(f_name, pos=(0, 0, 0), rot=(0, 0, 0), sample_num=10000, type="box"):
    if f_name[-3:] != 'stl':
        f_name += '.stl'
    objcm = cm.CollisionModel(objinit=os.path.join(config.ROOT, "obstacles", f_name), type=type)
    objmat4 = np.zeros([4, 4])
    objmat4[:3, :3] = rm.rotmat_from_euler(rot[0], rot[1], rot[2], axes="sxyz")
    objmat4[:3, 3] = pos
    objcm.sethomomat(objmat4)
    return item.Item(objcm=objcm, objmat4=objmat4, sample_num=sample_num)


if __name__ == '__main__':
    base, env = loadEnv_wrs()
    objcm = loadObj('cylinder.stl', pos=(700, 0, 780), rot=(0, 0, 0), transparency=1)
    objcm.reparentTo(base.render)
    # loadObjpcd('obstacles/cylinder.stl', pos=(800, 400, 780), rotmat=(0, -90, 0))
    # loadObjpcd('obstacles/pen.stl', pos=(800, 400, 785), rotmat=(0, -90, 0))
    rbt, rbtmg, rbtball = loadUr3e()
    # rbtmg.genmnp(rbt, togglejntscoord=True).reparentTo(base.render)
    rbtmg.gensnp(rbt, togglejntscoord=True).reparentTo(base.render)

    # rbtball.showfullcn(rbt)
    base.run()
