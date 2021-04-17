from motion import smoother as sm
from motion import checker as ctcb
from motion import collisioncheckerball as cdck
from motion.rrt import rrtconnect as rrtc
import tro.tro_robothelper as robothelper
from pandaplotutils import pandactrl as pc
import numpy as np
import utiltools.robotmath as rm
import tro.tro_animationgenerator as anime
import os
import copy
import tubepuzzlefaster as tp
import tubepuzzle_newstand as tp_nst
import tro.tro_locator as loc
import locatorfixed_newstand as locfixed_nst
import cv2
import environment.collisionmodel as cm
import utiltools.thirdparty.p3dhelper as p3dh
import utiltools.thirdparty.o3dhelper as o3dh
import pickle
import tro.tro_pickplaceplanner as ppp
import manipulation.grip.yumiintegrated.yumiintegrated as yi

def load_pickle_file_grip(model_name, root=None):
    """

    :param model_name:
    :param effect_grasps:
    :return:

    author: chenhao, revised by weiwei
    date: 20200105
    """

    if root is None:
        directory = "./"
    else:
        directory = root+"/"

    try:
        data = pickle.load(open(directory+'tro_predefinedgrippings.pickle', 'rb'))
        for k, v in data.items():
            print(k, len(v))
        effect_grasps = data[model_name]
        return effect_grasps
    except:
        print("load failed, create new graqsp file or grasp first.")
        raise ValueError("File or data not found!")

def load_pickle_file_suction(model_name, root=None):
    """

    :param model_name:
    :param effect_grasps:
    :return:

    author: chenhao, revised by weiwei
    date: 20200105
    """

    if root is None:
        directory = "./"
    else:
        directory = root+"/"

    try:
        data = pickle.load(open(directory+'tro_predefinedsuctions.pickle', 'rb'))
        for k, v in data.items():
            print(k, len(v))
        effect_grasps = data[model_name]
        return effect_grasps
    except:
        print("load failed, create new graqsp file or grasp first.")
        raise ValueError("File or data not found!")

if __name__ == '__main__':
    import manipulation.grip.freegrip as fg
    import environment.bulletcdhelper as bcd

    base = pc.World(camp=[2000, -2000, 1500], lookatpos=[0, 0, 100], up=[0,-1,1], autocamrotate=False)
    hndfa = yi.YumiIntegratedFactory()
    objcm = cm.CollisionModel(objinit="../objects/" + "vacuumhead.stl")
    gp = fg.Freegrip(objinit="../objects/" + "vacuumhead.stl", hand=hndfa.genHand())
    bmc = bcd.MCMchecker(toggledebug=True)

    gp.segShow(base, togglesamples=False, togglenormals=False,
                togglesamples_ref=False, togglenormals_ref=False,
                togglesamples_refcls=False, togglenormals_refcls=False, alpha =1)
    predefinedgrasps = load_pickle_file_grip(objcm.name)

    counter = [0]
    hndnps = [None]
    def update(predefinedgrasps, counter, hndnps, bmc, objcm, task):
        if hndnps[0] != None:
            hndnps[0].removeNode()
        if counter[0] >= len(predefinedgrasps):
            counter[0] = 0
            return task.again
        jawwidth, finger_center, hand_homomat = predefinedgrasps[counter[0]]
        hndnps[0] = hndfa.genHand()
        hndnps[0].setjawwidth(30)
        hndnps[0].set_homomat(hand_homomat)
        hndnps[0].reparentTo(base.render)
        iscollided = bmc.isMeshListMeshListCollided(hndnps[0].cmlist, [objcm])
        if iscollided:
            hndnps[0].setColor(.5, 0, 0, 1)
        print(iscollided)
        counter[0]+=1
        return task.again
    # objcm.reparentTo(base.render)
    taskMgr.doMethodLater(.1, update, "update", extraArgs=[predefinedgrasps, counter, hndnps, bmc, objcm], appendTask = True)
    base.run()