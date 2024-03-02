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
        data = pickle.load(open(directory+'tro_predefinedgrippings.pickle', 'rb'))
    except:
        print("load failed, create new file.")
        data = {}

    data[model_name] = effect_grasps
    for k, v in data.items():
        print(k, len(v))
    pickle.dump(data, open(directory+'tro_predefinedgrippings.pickle', 'wb'))

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
        data = pickle.load(open(directory+'tro_predefinedgrippings.pickle', 'rb'))
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
    c0nvec = rhx.np.array([0, -1, 0])
    approachvec = rhx.np.array([1, 0, 0])
    for z in [90+random.randint(0,20), 130+random.randint(-10,20), 170+random.randint(-10,10), 210+random.randint(-10,20)]:
        x = random.randint(-2,8)
        for anglei in range(0, 360, 15):
        # for anglej in [240, 270]:
            newav = rhx.np.dot(rhx.rm.rodrigues(c0nvec, anglei), approachvec)
            predefinedgrasps+=gu.define_gripper_grasps(hndfa, rhx.np.array([x, 0, z]), c0nvec, newav, jawwidth=30, cmodel=objcm, toggleflip=False)
    c0nvec = rhx.np.array([0, 1, 0])
    approachvec = rhx.np.array([1, 0, 0])
    for z in [90+random.randint(0,20), 130+random.randint(-10,20), 170+random.randint(-10,10), 210+random.randint(-10,20)]:
        x = random.randint(-2,8)
        for anglei in range(0, 360, 15):
        # for anglej in [240, 270]:
            newav = rhx.np.dot(rhx.rm.rodrigues(c0nvec, anglei), approachvec)
            predefinedgrasps+=gu.define_gripper_grasps(hndfa, rhx.np.array([x, 0, z]), c0nvec, newav, jawwidth=30, cmodel=objcm, toggleflip=False)

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