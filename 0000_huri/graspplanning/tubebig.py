import graspplanning.grasp_utils as gu
import robothelper as yh
import numpy as np

if __name__ == "__main__":
    rhx = yh.RobotHelper()
    hndfa = rhx.rgthndfa
    objcm = rhx.mcm.CollisionModel(objinit="../objects/" + "tubebig_capped.stl")

    predefinedgrasps = []
    c0nvec = rhx.np.array([0, -1, 0])
    approachvec = rhx.np.array([1, 0, 0])
    for z in [80, 95]:
    # for z in [95]:
        for anglei in range(0,360,45):
        # for anglei in range(180, 271, 90):
            newcv = rhx.np.dot(rhx.rm.rodrigues((0, 0, 1), anglei), c0nvec)
            tempav = rhx.np.dot(rhx.rm.rodrigues((0, 0, 1), anglei), approachvec)
            # for anglej in [210,240,270]:
            for anglej in [250, 265]:
            # for anglej in [240, 270]:
                newav = rhx.np.dot(rhx.rm.rodrigues(newcv, anglej), tempav)
                predefinedgrasps+=gu.define_gripper_grasps(hndfa, rhx.np.array([0, 0, z]), newcv, newav, jawwidth=14, cmodel=objcm)

    gu.write_pickle_file(objcm.name, predefinedgrasps, path=rhx.path + "/grasp" + hndfa.name)

    # show
    predefinedgrasps = gu.load_pickle_file(objcm.name, path=rhx.path + "/grasp" + hndfa.name)
    for eachgrasp in predefinedgrasps:
        print(eachgrasp)
        jawwidth, finger_center, hand_homomat = eachgrasp
        newhnd = hndfa.genHand()
        newhnd.setjawwidth(jawwidth)
        newhnd.set_homomat(hand_homomat)
        newhnd.reparentTo(rhx.base.render)
    objcm.reparentTo(rhx.base.render)
    rhx.base.run()
