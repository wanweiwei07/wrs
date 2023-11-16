import graspplanning.grasp_utils as gu
import robothelper as yh
import numpy as np
import pickle
import random

def write_pickle_file(model_name, effect_grasps, root=None):
    """

    :param model_name:
    :param effect_grasps:
    :return:

    author: chenhao, revised by weiwei
    date: 20200104
    """

    if root is None:
        directory = "./"
    else:
        directory = root+"/"

    try:
        data = pickle.load(open(directory+'tro_predefinedsuctions.pickle', 'rb'))
    except:
        print("load failed, create new file.")
        data = {}

    data[model_name] = effect_grasps
    for k, v in data.items():
        print(k, len(v))
    pickle.dump(data, open(directory+'tro_predefinedsuctions.pickle', 'wb'))

def load_pickle_file(model_name, root=None):
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

if __name__ == "__main__":
    rhx = yh.RobotHelper()
    hndfa = rhx.rgthndfa
    objcm = rhx.mcm.CollisionModel(objinit="../objects/" + "vacuumhead.stl")

    predefinedgrasps = []
    c0nvec = rhx.np.array([0,1,0])
    approachvec = rhx.np.array([1,0,0])
    for z in [120+random.randint(0,10), 160+random.randint(0,10),  200+random.randint(-30,-20)]:
    # for z in [95]:
        x = random.randint(-2,8)
        for anglei in range(0,360,45):
        # for anglei in range(180, 271, 90):
        #     newcv = rhx.np.dot(rhx.rm.rodrigues(approachvec, anglei), c0nvec)
            newav = rhx.np.dot(rhx.rm.rodrigues(c0nvec, anglei), approachvec)
            # for anglej in [210,240,270]:
            # for anglej in [250, 265]:
            # for anglej in [240, 270]:
            #     newav = rhx.np.dot(rhx.rm.rodrigues(newcv, anglej), tempav)
            predefinedgrasps+=gu.define_suction(hndfa,rhx.np.array([x,0,z]),c0nvec, newav, objcm=objcm)

    c0nvec = rhx.np.array([0,-1,0])
    approachvec = rhx.np.array([1,0,0])
    for z in [120+random.randint(0,10), 160+random.randint(0,10),  200+random.randint(-30,-20)]:
    # for z in [95]:
        x = random.randint(-2,8)
        for anglei in range(0,360,45):
        # for anglei in range(180, 271, 90):
        #     newcv = rhx.np.dot(rhx.rm.rodrigues(approachvec, anglei), c0nvec)
            newav = rhx.np.dot(rhx.rm.rodrigues(c0nvec, anglei), approachvec)
            # for anglej in [210,240,270]:
            # for anglej in [250, 265]:
            # for anglej in [240, 270]:
            #     newav = rhx.np.dot(rhx.rm.rodrigues(newcv, anglej), tempav)
            predefinedgrasps+=gu.define_suction(hndfa,rhx.np.array([x,0,z]),c0nvec, newav, objcm=objcm)

    write_pickle_file(objcm.name, predefinedgrasps)

    # show
    predefinedgrasps = load_pickle_file(objcm.name)
    for eachgrasp in predefinedgrasps:
        print(eachgrasp)
        jawwidth, finger_center, hand_homomat = eachgrasp
        newhnd = hndfa.genHand()
        newhnd.setjawwidth(jawwidth)
        newhnd.set_homomat(hand_homomat)
        newhnd.reparentTo(rhx.base.render)
    objcm.reparentTo(rhx.base.render)
    rhx.base.run()