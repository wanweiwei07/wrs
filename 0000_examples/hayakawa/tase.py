import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as rtq85, robot_sim as ur3ds, motion as rrtc, modeling as cm, modeling as gm
import numpy as np
import copy
import open3d as o3d
import random
from skimage.measure import LineModelND, ransac
import research_posemake_many as pose
import math
import socket
import wrs.robot_con.ur.program_builder as pb
import pickle
import time
import sympy as sp
from scipy.optimize import basinhopping
import wrs.motion.optimization_based.incremental_nik as inik

rotatedegree = 5
endthreshold = 3
objpointrange = [300, 900, -400, 800, 1051, 1500]
objpos_finalmax_lft = np.array([250, 250, 1600])
objpos_finalmax_rgt = np.array([250, -250, 1600])
## ToDo : Change the param according to the object
## param(acrylic board) ----------------------------------
objpath = "./research_flippingboard2_mm.stl"
l = 300
w = 300
h = 40
M = 4.0
g = 9.8
myu0 = 0.5
myu1 = 0.4
vmax = 30
anglemax = 20
timestep = 1.0
thetathreshold = 50
# limitdegree = 70
limitdegree = 90 + math.degrees(math.atan(h / l)) + 10
print(limitdegree)

objpos_start = np.array([.381, .250, 1.1], dtype=float)
pushpose_pre = np.array([15.46510215, -124.31216495, -22.21501633, -68.25934326, 108.02513127, 39.89826658])
pushrot = np.array([[0.02974146, -0.74159545, 0.67018776],
                    [0.06115005, -0.66787857, -0.74175392],
                    [0.99768538, 0.06304286, 0.02548492]])
## ---------------------------------------------------------

## param(stainless box) ----------------------------------------
# obj_path = "./objects/TCbox.stl"
# l = 300
# w = 400
# h = 150
# M = 6.0
# g = 9.8
# myu0 = 0.4
# myu1 = 0.1
# vmax = 30
# anglemax = 20
# timestep = 1.0
# thetathreshold = 49
# # limitdegree = 125
# limitdegree = 90 + math.degrees(math.atan(h / l)) + 10
# print(limitdegree)
#
# objpos_start = np.array([381, 250, 1035], dtype=float)
# pushpose_pre = np.array([15.46510215, -124.31216495, -22.21501633, -68.25934326, 108.02513127, 39.89826658])
# pushrot = np.array([[ 0.02974146, -0.74159545, 0.67018776],
#                     [ 0.06115005, -0.66787857, -0.74175392],
#                     [ 0.99768538, 0.06304286, 0.02548492]])
## -------------------------------------------------------------

## param(plywood board) ----------------------------------------
# obj_path = "./objects/400×500×44.stl"
# l = 500
# w = 400
# h = 44
# M = 6.4
# g = 9.8
# myu0 = 0.6
# myu1 = 0.3
# vmax = 45
# anglemax = 20
# timestep = 1.0
# thetathreshold = 57
# # limitdegree = 100
# limitdegree = 90 + math.degrees(math.atan(h / l)) + 10
# print(limitdegree)
#
# objpos_start = np.array([240, 140-30, 1035], dtype=float)
# pushpose_pre = np.array([12.840271549966547, -92.64224433679576, -39.088370300126584, 112.36556622471164, -92.64626048802772, 35.67784488430386])
# pushrot = np.array([[ 0.02437668,  0.74389354,  0.66785341],
#                     [-0.16925718,  0.66147852, -0.73061493],
#                     [-0.98527041, -0.09522902,  0.14203398]])
## --------------------------------------------------------------

Mg = [0, -M * g]
pulleypos = np.array([580, 370, 2500])
ropetoppos = np.array([.25, 0, 2.5])
rotate_axis = np.array([1, 0, 0])

## calibration_matrix 2020-0818
calibration_matrix = np.array([[3.95473025e-02, -8.94575014e-01, -4.45164638e-01, 7.62553715e+02],
                               [-9.98624616e-01, -2.00371608e-02, -4.84498644e-02, 6.67240739e+01],
                               [3.44222026e-02, 4.46468426e-01, -8.94137045e-01, 2.12149540e+03],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


def gethangedpos(objpos, objrot):
    ## ベニヤ板の場合
    hangedpos = copy.copy(objpos) + (w / 2) * objrot[:, 0] + l * objrot[:, 1] + h * objrot[:, 2]
    return hangedpos


def getobjcenter(objpos, objrot):
    ## ベニヤ板の場合
    objcenter = copy.copy(objpos) + (w / 2) * objrot[:, 0] + (l / 2) * objrot[:, 1] + (h / 2) * objrot[:, 2]
    return objcenter


def getrotatecenter(objpos, objrot):
    ## ベニヤ板の場合
    rotatecenter = copy.copy(objpos) + (w / 2) * objrot[:, 0]
    return rotatecenter


def getrotatecenter_after(objpos, objrot):
    ## ベニヤ板の場合
    rotatecenter = copy.copy(objpos) + (w / 2) * objrot[:, 0] + h * objrot[:, 2]
    return rotatecenter


def getrefpoint(objpos, objrot):
    ## ベニヤ板の場合
    refpoint = copy.copy(objpos) + h * objrot[:, 2]
    return refpoint


def getpointcloudkinect(pointrange=[]):
    pcd = client.getpcd()
    pcd2 = np.ones((len(pcd), 4))
    pcd2[:, :3] = pcd
    newpcd = np.dot(calibration_matrix, pcd2.T).T[:, :3]
    if len(pointrange) > 0:
        x0, x1, y0, y1, z0, z1 = pointrange
        newpcd = np.array([x for x in newpcd if (x0 < x[0] < x1) and (y0 < x[1] < y1) and (z0 < x[2] < z1)])
    return newpcd


## 2020_0722作成
def getpointcloudkinectforrope_up(rbt, armname, initialpoint, pointrange):
    # mph = client.getpcd()
    # pcd2 = np.ones((len(mph), 4))
    # pcd2[:, :3] = mph
    # newpcd = np.dot(calibration_matrix, pcd2.T).T[:, :3]
    newpcd = getpointcloudkinect(pointrange)
    finalpoint = rbt.get_gl_tcp(manipulator_name=armname)[0]
    tostartvec = copy.copy(initialpoint - finalpoint)
    newpcd = np.array([x for x in newpcd if x[2] < 1700])
    newpcd = np.array([x for x in newpcd if rm.angle_between_vectors(tostartvec, x - finalpoint) < math.radians(30)])
    return newpcd


def getpointcloudkinectforrope_down(rbt, armname, pointrange=[]):
    # mph = client.getpcd()
    # pcd2 = np.ones((len(mph), 4))
    # pcd2[:, :3] = mph
    # newpcd = np.dot(calibration_matrix, pcd2.T).T[:, :3]
    newpcd = getpointcloudkinect(pointrange)
    initialpoint = rbt.get_gl_tcp(manipulator_name=armname)[0]
    # eepos_under = copy.copy(initialpoint)
    # eepos_under[2] -= 250
    # refvec = copy.copy(eepos_under - initialpoint)
    base.pggen.plotSphere(base.render, pos=initialpoint, radius=10, rgba=[1, 0, 0, 1])
    minuszaxis = np.array([0, 0, -1])
    newpcd = np.array([x for x in newpcd if 1100 < x[2] < initialpoint[2]])
    newpcd = np.array([x for x in newpcd if rm.angle_between_vectors(minuszaxis, x - initialpoint) < math.radisn(40)])
    return newpcd


## RANSACでロープを検出
def doRANSAC(newpcd, threshold):
    model_robust, inliers = ransac(newpcd, LineModelND, min_samples=100, residual_threshold=threshold, max_trials=1000)
    outliers = inliers == False
    ## 検出した直線の表示
    ropeline = []  # ロープの点群のみ取り出す
    for i, eachpoint in enumerate(newpcd):
        if inliers[i] == True:
            # base.pggen.plotSphere(base.render, pos=newpcd[numberofrope], major_radius=10, rgba=[1, 0, 0, .5])
            ropeline.append(newpcd[i])
    return ropeline


## リストを昇順に並べ替える(デフォルト:z座標)
def ascendingorder(array, axis=2):
    array = np.asarray(array)
    array_ascend = array[array[:, axis].argsort(), :]
    return array_ascend


## リストをz座標で降順に並べ替える
def descendingorder(array, axis):
    array_ascend = ascendingorder(array, axis)
    array_descend = array_ascend[::-1]
    return array_descend


def create_candidate_points(arm_name, initialhandpos, obstacles=None, limitation=None):
    if arm_name == "lft_arm":
        pointlistrange = np.array([.15, .3, .05, .3, 1.3, initialhandpos[2]])
    elif arm_name == "rgt_arm":
        pointlistrange = np.array([.15, .3, -.2, -.05, 1.3, initialhandpos[2]])
    if obstacles is not None and arm_name == "lft":
        for obs in obstacles:
            ## 3dモデルを点群化し、原点に配置
            obs_points = obs.sample_surface(8000)
            homomat = obs.get_homomat()
            obs_points_converted = np.ones((len(obs_points), 4))
            obs_points_converted[:, :3] = obs_points
            obs_points_converted = np.dot(homomat, obs_points_converted.T).T[:, :3]
            zmax = max(obs_points_converted[:, 2]) + .15
            pointlistrange[4] = zmax
    # print("pointrange", pointlistrange)
    if limitation is not None:
        pointlistrange[3] = limitation
    points = []
    number = 30
    for i in range(number):
        x = random.uniform(pointlistrange[0], pointlistrange[1])
        y = random.uniform(pointlistrange[2], pointlistrange[3])
        z = random.uniform(pointlistrange[4], pointlistrange[5])
        point = [x, y, z]
        # print("point", point)
        points.append(point)
    return points


## 始点での把持姿勢を探索
def decidestartpose(armname, ropelinesorted, predefined_grasps, fromjnt, startpointid):
    IKpossiblelist_start = []
    while True:
        objpos_initial = ropelinesorted[startpointid]
        objrot_initial = np.eye(3)
        objmat4_initial = rm.homomat_from_posrot(objpos_initial, objrot_initial)
        obj_initial = copy.deepcopy(ropeobj)  # ->早川：変数の定義はどこですか？また、obj.copy()を使ってください．
        obj_initial.set_rgba(rgba=[1, 0, 0, .5])
        obj_initial.set_homomat(objmat4_initial)
        for i, eachgrasp in enumerate(predefined_grasps):
            prejawwidth, prehndfc, prehndpos, prehndrotmat = eachgrasp
            prehndhomomat = rm.homomat_from_posrot(prehndpos, prehndrotmat)
            hndmat4_initial = np.dot(objmat4_initial, prehndhomomat)
            eepos_initial = rm.transform_points_by_homomat(objmat4_initial, prehndfc)[:3]
            eerot_initial = hndmat4_initial[:3, :3]
            start = robot_s.ik(component_name=armname,
                               tgt_pos=eepos_initial,
                               tgt_rotmat=eerot_initial,
                               seed_jnt_values=fromjnt)
            if start is not None:
                original_jnt_values = robot_s.get_jnt_values(component_name=armname)
                robot_s.fk(component_name=armname, jnt_values=start)
                objrelmat = robot_s.cvt_gl_to_loc_tcp(armname, objpos_initial, objrot_initial)
                ## 衝突検出
                cd_result = robot_s.is_collided(obscmlist)
                if not cd_result:
                    IKpossiblelist_start.append([start, objrelmat, i])
                robot_s.fk(component_name=armname, jnt_values=original_jnt_values)
        if len(IKpossiblelist_start) > 0:
            return IKpossiblelist_start, objpos_initial, objrot_initial, startpointid
        startpointid = startpointid + 1
        if startpointid == len(ropelinesorted):
            print("始点が存在しませんでした")
            return [False, False, False, False]
        print("startpointid = ", startpointid)


## 終点での把持姿勢を探索(終点を1つにしたとき)
def decidegoalpose_onepoint(arm_name,
                            IKpossiblelist_start,
                            hold_pos_final,
                            predefined_grasps,
                            obscmlist):
    IKpossiblelist_startgoal = []
    objrot_final = np.eye(3)
    objmat4_final = rm.homomat_from_posrot(hold_pos_final, objrot_final)
    # obj_final = copy.deepcopy(ropeobj)
    # obj_final.set_rgba(rgba=[1, 0, 0, .5])
    # obj_final.set_homomat(objmat4_final)
    for i in IKpossiblelist_start:
        prejawwidth, prehndfc, prehndpos, prehndrotmat = predefined_grasps[i[2]]
        prehndhomomat = rm.homomat_from_posrot(prehndpos, prehndrotmat)
        hndmat4_final = np.dot(objmat4_final, prehndhomomat)
        eepos_final = rm.transform_points_by_homomat(objmat4_final, prehndfc)[:3]
        eerot_final = hndmat4_final[:3, :3]
        fromjnt = i[0]
        goal = robot_s.ik(component_name=arm_name,
                          tgt_pos=eepos_final,
                          tgt_rotmat=eerot_final,
                          seed_jnt_values=fromjnt)
        # mgm.gen_frame(pos=eepos_final, rotmat=eerot_final).attach_to(base)
        # robot_s.fk(arm_name, fromjnt)
        # robot_s.gen_meshmodel().attach_to(base)
        # base.run()
        if goal is not None:
            original_jnt_values = robot_s.get_jnt_values(component_name=arm_name)
            robot_s.fk(component_name=arm_name, jnt_values=goal)
            cd_result = robot_s.is_collided(obscmlist)
            if not cd_result:
                IKpossiblelist_startgoal.append([i[0], goal, i[1], [2]])
            robot_s.fk(component_name=arm_name, jnt_values=original_jnt_values)
    if len(IKpossiblelist_startgoal) > 0:
        return IKpossiblelist_startgoal
    else:
        print("終点での姿勢が存在しません")
        return False


## 最初の一回の引き動作のみ
def getsuitablegoalpos_first(arm_name,
                             IKpossiblelist_start,
                             objpos_initial,
                             objpos_finallist,
                             predefined_grasps):
    ## 重み
    w_length = 1
    w_FT = 1
    w_manip = 1
    pullinglengthlist = []
    for i, selected_objpos_final in enumerate(objpos_finallist):
        pullinglength = np.linalg.norm(objpos_initial - selected_objpos_final)
        pullinglengthlist.append(pullinglength)
    pullinglength_ref = min(pullinglengthlist)
    ## 評価要素を計算
    totalIKpossiblelist_startgoal = []
    costlist = []
    assessment_value_list = []
    for i, selected_objpos_final in enumerate(objpos_finallist):
        ## pullinglength
        pullinglength = pullinglengthlist[i]
        pullinglength_cost = 1 - pullinglength_ref / pullinglength
        ## FT
        zaxis = np.array([0, 0, 1])
        tostartvec = objpos_initial - selected_objpos_final
        theta = rm.angle_between_vectors(rm.unit_vector(tostartvec), zaxis)
        FT_cost = math.cos(theta)
        ## manipulability
        IKpossiblelist_startgoal = decidegoalpose_onepoint(arm_name,
                                                           IKpossiblelist_start,
                                                           objpos_initial,
                                                           selected_objpos_final,
                                                           predefined_grasps,
                                                           obscmlist)
        if IKpossiblelist_startgoal is not False and IKpossiblelist_start is not False:
            manipulability_cost = len(IKpossiblelist_startgoal) / len(IKpossiblelist_start)
        else:
            manipulability_cost = -100
        ## 各コストのリスト
        costlist.append([pullinglength_cost, FT_cost, manipulability_cost])
        ## 使用可能なIKのリスト
        totalIKpossiblelist_startgoal.append(IKpossiblelist_startgoal)
        ## tostartvec, togoalvecのリスト
        # veclist.append([tostartvec, togoalvec])
        ## 評価関数値のリスト
        assessment_value = w_length * pullinglength_cost + w_manip * manipulability_cost + w_FT * FT_cost
        ## [assessment_value, chosen_objpos_final]
        assessment_value_list.append([assessment_value, i])
    assessment_value_list = descendingorder(assessment_value_list, axis=0)
    print("assessment_value_list", assessment_value_list)
    return assessment_value_list, totalIKpossiblelist_startgoal, costlist


def getsuitablegoalpos_second(arm_name,
                              IKpossiblelist_start,
                              objpos_initial,
                              objpos_finallist,
                              predefined_grasps,
                              predictlist):
    # objpos_final_under = np.array([250, 0, 1650])
    ## 重み
    w_length = 1
    w_FT = 0
    w_manip = 0
    pullinglengthlist = []
    for i, use_objpos_final in enumerate(objpos_finallist):
        pullinglength = np.linalg.norm(objpos_initial - use_objpos_final)
        pullinglengthlist.append(pullinglength)
    pullinglength_ref = min(pullinglengthlist)
    ## 評価要素を計算
    totalIKpossiblelist_startgoal = []
    costlist = []
    ## 各点における予測値の要素を計算
    elements_for_predictlist = []
    for i, use_objpos_final in enumerate(objpos_finallist):
        flag = 0
        togoalvec = copy.copy(use_objpos_final - objpos_initial)
        d_next = np.linalg.norm(objpos_initial - use_objpos_final)
        d_before, theta_before, theta_beforebefore = predictlist
        ## 次の角度の予測値
        theta_next = theta_before + (theta_before - theta_beforebefore) * (d_next / d_before)
        if theta_next > thetathreshold:
            d_next = (thetathreshold - theta_before) * (d_before / (theta_before - theta_beforebefore))
            use_objpos_final = copy.copy(objpos_initial) + d_next * rm.unit_vector(togoalvec)
            togoalvec = copy.copy(use_objpos_final - objpos_initial)
            flag = 1
        elements_for_predictlist.append([d_next, theta_next, use_objpos_final, flag, togoalvec])
    ## 評価値の計算
    value_plus_element = []
    for i, eachpos in enumerate(objpos_finallist):
        use_element = elements_for_predictlist[i]
        use_objpos_final = use_element[2]
        ## pullinglength
        pullinglength = pullinglengthlist[i]
        pullinglength_cost = 1 - pullinglength_ref / pullinglength
        print("axis_length cost = ", pullinglength_cost)
        ## FT
        zaxis = np.array([0, 0, 1])
        togoalvec = use_element[4]
        tostartvec = copy.copy(togoalvec) * (-1)
        degree = rm.angle_between_vectors(rm.unit_vector(tostartvec), zaxis)
        FT_cost = math.cos(degree)
        print("force cost = ", FT_cost)
        ## 予測位置での物体の情報
        obj_predict = copy.deepcopy(obj)
        objectpos = copy.copy(objpos_start)
        objectrot = rm.rotmat_from_axangle(rotate_axis, math.radians(use_element[1]))
        objmat_predict = rm.homomat_from_posrot(objectpos, objectrot)
        obj_predict.set_rotmat(objmat_predict)
        ## 予測位置での物体を障害物として追加
        obscmlist.append(obj_predict)
        pickle.dump(obscmlist, open("obscmlist.pickle", "wb"))
        ## manipulability
        IKpossiblelist_startgoal = decidegoalpose_onepoint(arm_name,
                                                           IKpossiblelist_start,
                                                           objpos_initial,
                                                           use_objpos_final,
                                                           predefined_grasps,
                                                           obscmlist)
        if IKpossiblelist_startgoal is not False and IKpossiblelist_start is not False:
            manipulability_cost = len(IKpossiblelist_startgoal) / len(IKpossiblelist_start)
        else:
            manipulability_cost = -100
        obscmlist.pop(-1)
        print("manipulation cost = ", manipulability_cost)
        ## 各コストのリスト
        costlist.append([pullinglength_cost, FT_cost, manipulability_cost])
        ## 使用可能なIKのリスト
        totalIKpossiblelist_startgoal.append(IKpossiblelist_startgoal)
        ## 評価関数値のリスト
        assessment_value = w_length * pullinglength_cost + w_manip * manipulability_cost + w_FT * FT_cost
        ## value_plus_element : [assessment_value, i, d_next, theta_next, use_objpos_final, flag, togoalvec]
        value_plus_element.append([assessment_value, i] + use_element)
        # ## [assessment_value, chosen_objpos_final]
        # assessment_value_list.append([assessment_value, i])
    # assessment_value_list = descendingorder(assessment_value_list, axis=0)
    value_plus_element = descendingorder(value_plus_element, axis=0)
    assessment_value_list = value_plus_element[:, :2]  ## assessment_value, i
    print("assessment_value_list", assessment_value_list)
    elements_for_predictlist = value_plus_element[:, 2:6]  ## d_next, theta_next, use_objpos_final, flag
    togoalveclist = value_plus_element[:, 6]  ## togoalvec
    return assessment_value_list, totalIKpossiblelist_startgoal, costlist, elements_for_predictlist, togoalveclist


## 終点での把持姿勢を探索(0203作成：左右で引く方向を変換)
def decidegoalpose(arm_name,
                   IKpossiblelist_start,
                   objpos_initial,
                   predefined_grasps,
                   objpos_final=np.array([260, 0, 1200]),
                   diff=None,
                   label="down"):
    # tic = time.time()
    IKpossiblelist_startgoal = []
    # if label == "down":
    #     if arm_name == "lft":
    #         hold_pos_final = np.array([260, 100, 1400])
    #     else:
    #         hold_pos_final = np.array([260, -100, 1400])
    objrot_final = np.eye(3)
    tostartvec = objpos_initial - objpos_final  ## 終点から始点への方向ベクトル(非正規化)
    togoalvec = objpos_final - objpos_initial  ## 始点から終点への方向ベクトル(非正規化)
    togoalvec_len = np.linalg.norm(togoalvec)
    togoalvec_normalize = rm.unit_vector(togoalvec)
    pullinglength = copy.copy(togoalvec_len)
    if label == "down":
        if diff is not None:  ## 一回目の引き動作のための条件
            if diff < togoalvec_len:
                print("pass")
                pullinglength = copy.copy(diff)
    while True:
        if label == "down":
            objpos_final = objpos_initial + pullinglength * togoalvec_normalize
        else:
            pass
        togoalvec = objpos_final - objpos_initial
        print("hold_pos_final", objpos_final)
        objmat4_final = rm.homomat_from_posrot(objpos_final, objrot_final)
        obj_final = copy.deepcopy(ropeobj)
        obj_final.set_rgba([1, 0, 0, .5])
        obj_final.set_rotmat(objmat4_final)
        for i in IKpossiblelist_start:
            prejawwidth, prehndfc, prehndpos, prehndrotmat = predefined_grasps[i[2]]
            prehndhomomat = rm.homomat_from_posrot(prehndpos, prehndrotmat)
            hndmat4_final = np.dot(objmat4_final, prehndhomomat)
            eepos_final = rm.transform_points_by_homomat(objmat4_final, prehndfc)[:3]
            eerot_final = hndmat4_final[:3, :3]
            # goal = robot_s.numik(eepos_final, eerot_final, arm_name)
            fromjnt = i[0]
            goal = robot_s.ik(component_name=arm_name,
                              tgt_pos=eepos_final,
                              tgt_rotmat=eerot_final,
                              seed_jnt_values=fromjnt)
            if goal is not None:
                original_jnt_values = robot_s.get_jnt_values(component_name=arm_name)
                robot_s.fk(component_name=arm_name, jnt_values=goal)
                cd_result = robot_s.is_collided(obscmlist)
                if not cd_result:
                    IKpossiblelist_startgoal.append([i[0], goal, i[1], [2]])
                robot_s.fk(manipulator_name=arm_name, jnt_values=original_jnt_values)
        if len(IKpossiblelist_startgoal) > 0:
            print(str(pullinglength) + "mm引きます")
            return IKpossiblelist_startgoal, objpos_final, tostartvec, togoalvec
        pullinglength -= 1
        if pullinglength < 0:
            print("終点が存在しません")
            return [False, False, False, False]


## 中継点での把持姿勢を探索
def decidemidpose(arm_name, IKpossiblelist_startgoal, handdir, objpos_final=None):
    centerflag = 0
    if objpos_final is not None:
        if objpos_final[1] == 0:
            centerflag = 1
    print("中継点での姿勢を探索します")
    IKpossiblelist = []
    for i in IKpossiblelist_startgoal:
        direction = rm.unit_vector(handdir[i[3]]) * (-1)
        distance = .08
        while True:
            if objpos_final is None or centerflag == 1:  ## 終点が中心のとき(hold_pos_final = Noneを設定)終点からの中継点も計算
                ## 始点に対する中継点の経路
                midpathstart = robot_inik_solver.gen_rel_linear_motion_with_given_conf(arm_name,
                                                                                       i[0],
                                                                                       direction,
                                                                                       distance,
                                                                                       obscmlist,
                                                                                       type="source")
                midpathgoal = robot_inik_solver.gen_rel_linear_motion_with_given_conf(arm_name,
                                                                                       i[1],
                                                                                       direction,
                                                                                       distance,
                                                                                       obscmlist,
                                                                                       type="source")
                if len(midpathstart) > 0 and len(midpathgoal) > 0:
                    # robot_s.movearmfk(midpath[-1], arm_name)
                    # mideepos, mideerot = robot_s.getee(arm_name)
                    midpathstart = midpathstart[::-1]
                    midjntstart = copy.copy(midpathstart[0])
                    midjntgoal = copy.copy(midpathgoal[0])
                    #### list[startjnt, goaljnt, midjntlist, midpathlist, objrelmat, id]
                    IKpossiblelist.append(
                        [i[0], i[1], [midjntstart, midjntgoal], [midpathstart, midpathgoal], i[2], i[3]])
                    break
                else:
                    distance -= 1
                    if distance <= 30:
                        print(str(i[3]) + "番目の姿勢は中継点が見つかりません")
                        break
            else:
                ## 始点に対する中継点の経路
                midpathstart = robot_inik_solver.gen_rel_linear_motion_with_given_conf(arm_name,
                                                                                       i[0],
                                                                                       direction,
                                                                                       distance,
                                                                                       [],
                                                                                       type="source")
                if len(midpathstart) > 0:
                    midpathstart = midpathstart[::-1]
                    midjntstart = copy.copy(midpathstart[0])
                    goaljnt = i[1]
                    #### list[startjnt, goaljnt, midjntlist, midpathlist, objrelmat, id]
                    IKpossiblelist.append([i[0], i[1], [midjntstart, goaljnt], [midpathstart, []], i[2], i[3]])
                    break
                else:
                    distance -= 1
                    if distance <= 30:
                        print(str(i[3]) + "番目の姿勢は始点に対する中継点が見つかりません")
                        break
    return IKpossiblelist


def ropepullingmotion(IKpossiblelist, togoalvec, ctcallback, theta=None, theta_next=None):
    for i in range(len(IKpossiblelist)):
        useid = random.randint(0, len(IKpossiblelist) - 1)
        use_startjnt = IKpossiblelist[useid][0]
        use_objrelmat = IKpossiblelist[useid][4]
        pullinglength = np.linalg.norm(togoalvec)
        print("pullinglength : ", pullinglength)
        togoalvec_copy = copy.copy(togoalvec)
        direction = rm.unit_vector(togoalvec_copy)
        obstacles_forpullingrope = copy.deepcopy(obscmlist)
        if theta is not None and theta_next is not None:
            currentobj = copy.deepcopy(obj)
            currentrot = rm.rotmat_from_axangle(rotate_axis, theta)
            currentmat = rm.homomat_from_posrot(objpos_start, currentrot)
            currentobj.set_homomat(currentmat)
            nextobj = copy.deepcopy(obj)
            nextrot = rm.rotmat_from_axangle(rotate_axis, theta_next)
            nextmat = rm.homomat_from_posrot(objpos_start, nextrot)
            nextobj.set_homomat(nextmat)
            i = 0.1
            while True:
                appendobj = copy.deepcopy(obj)
                appendrot = rm.rotmat_from_euler(rotate_axis, theta + i)
                appendmat = rm.homomat_from_posrot(objpos_start, appendrot)
                appendobj.set_homomat(appendmat)
                obstacles_forpullingrope.append(appendobj)
                i += 0.1
                if theta + i >= theta_next:
                    break
        ropepulling = robot_inik_solver.gen_rel_linear_motion_with_given_conf(arm_name,
                                                                              use_startjnt,
                                                                              direction,
                                                                              pullinglength,
                                                                              obstacles_forpullingrope,
                                                                              type="source")
        ropepulling = ctcallback.getLinearPrimitive(use_startjnt, direction, pullinglength, [ropeobj], [use_objrelmat],
                                                    obstacles_forpullingrope, type="source")
        if len(ropepulling) > 0:
            print("ropepulling motion planning success!")
            return ropepulling, IKpossiblelist[useid], useid
    print("ropepulling motion not found!")
    return [False, False, False]
    # return ropepulling, IKpossiblelist[useid], useid


def RRTmotion(startjoint, goaljoint, ctcallback, obscmlist, expanddis, maxtime):
    tic = time.time()
    smoother = sm.Smoother()
    pathplanner = rrtc.RRTConnect(start=startjoint, goal=goaljoint, ctcallback=ctcallback,
                                  starttreesamplerate=30,
                                  endtreesamplerate=30, expanddis=expanddis,
                                  maxiter=2000, maxtime=maxtime)

    path, _ = pathplanner.planning(obscmlist)
    if path is not False:
        path = smoother.pathsmoothing(path, pathplanner)
        return path
    else:
        return False


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Estimate normal with search major_radius %.3f." % 10)
    o3d.geometry.PointCloud.estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search major_radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd,
                                                     o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=100))
    return pcd, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = 30
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal linear_distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, result_ransac):
    distance_threshold = 30
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   linear_distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


def objectfitting(newpcd, fitobjpcd, refpoint_fitting):
    samplepoint = copy.copy(newpcd)
    targetpoint = sample_volume(fitobjpcd, 20000)
    targetpointnew = copy.copy(targetpoint)
    while True:
        targetpoint = targetpointnew
        voxel_size = 30
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(samplepoint)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(targetpoint)
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        print("RANSAC start")
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, source_fpfh, voxel_size)
        print(result_ransac)
        print("ICP start")
        result_icp = refine_registration(source, target, result_ransac)
        print(result_icp)
        transformation = result_icp.transformation
        transmatrix = np.linalg.inv(transformation)
        # print("transmatrix = ", transmatrix)
        targetpointnew = np.concatenate((targetpoint, np.ones((targetpoint.shape[0], 1))), axis=1)
        targetpointnew = np.dot(transmatrix, targetpointnew.T)
        targetpointnew = targetpointnew.T[:, :3]
        ## refpointを変換し、z座標で降順に並べる
        refpoint_fitting = np.dot(transmatrix, refpoint_fitting.T).T
        refpoint_fitting = descendingorder(refpoint_fitting, axis=2)
        print("diff:", abs(refpoint_fitting[0][2] - refpoint_fitting[1][2]))
        print("refpoint", refpoint_fitting)
        # for i in refpoint_fitting:
        #     base.pggen.plotSphere(base.render, pos=i[:3], major_radius=15, rgba=[1,1,0,1])
        # break
        toppoints_zdiff = abs(refpoint_fitting[0][2] - refpoint_fitting[1][2])
        toppoints_length = abs(refpoint_fitting[0] - refpoint_fitting[1])
        if 0 < toppoints_zdiff < 10 and 300 < np.linalg.norm(toppoints_length) < 450:
            print("----------- fitting end_type ------------")
            break
    return targetpointnew, refpoint_fitting


def getobjaxis(targetpointnew, refpoint_fitting, flag=0):
    ## 点群の重心を求める
    cog = np.mean(targetpointnew, axis=0)

    ## 法線推定
    targetpcd = o3d.geometry.PointCloud()
    targetpcd.points = o3d.utility.Vector3dVector(targetpointnew)
    o3d.geometry.PointCloud.estimate_normals(targetpcd, o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    print("has normals?", targetpcd.has_normals())
    normal_array = np.asarray(targetpcd.normals)
    normal_array_use = np.array([x for x in normal_array if (x[2] > 0)])
    ## 法線ベクトルを求める
    zaxis_obj = np.mean(normal_array_use, axis=0)
    normz = np.linalg.norm(zaxis_obj)
    zaxis_obj /= normz

    ## 床面からの傾きを求める
    zaxis = np.array([0, 0, 1])
    norm1 = np.linalg.norm(zaxis)
    norm2 = np.linalg.norm(zaxis_obj)
    dot = np.dot(zaxis, zaxis_obj)
    theta = math.acos(dot / (norm1 * norm2)) * 180 / math.pi

    norm1 = np.linalg.norm(refpoint_fitting[0][:3] - refpoint_fitting[2][:3])
    norm2 = np.linalg.norm(refpoint_fitting[0][:3] - refpoint_fitting[3][:3])
    if norm1 < norm2:
        yaxis_obj = refpoint_fitting[0][:3] - refpoint_fitting[2][:3]
    else:
        yaxis_obj = refpoint_fitting[0][:3] - refpoint_fitting[3][:3]

    normy = np.linalg.norm(yaxis_obj)
    yaxis_obj /= normy

    xaxis_obj = np.cross(yaxis_obj, zaxis_obj)
    normx = np.linalg.norm(xaxis_obj)
    xaxis_obj /= normx

    return xaxis_obj, yaxis_obj, zaxis_obj, theta, cog


def getobjposandrot(zaxis_obj):
    objpos_initial = copy.copy(objpos_start)
    theta = rm.degree_betweenvector(zaxis_obj, [0, 0, 1])
    objrot_initial = rm.rodrigues(rotate_axis, theta)

    return objpos_initial, objrot_initial


def getobjposandrot_after(cog, xaxis_obj, yaxis_obj, zaxis_obj):
    ## ベニヤ板の場合
    objpos_initial = copy.copy(cog)
    objpos_initial -= (w / 2) * xaxis_obj
    objpos_initial -= (l / 2) * yaxis_obj
    objpos_initial -= (h / 2) * zaxis_obj

    objrot_initial = np.empty((0, 3))
    objrot_initial = np.append(objrot_initial, np.array([xaxis_obj]), axis=0)
    objrot_initial = np.append(objrot_initial, np.array([yaxis_obj]), axis=0)
    objrot_initial = np.append(objrot_initial, np.array([zaxis_obj]), axis=0)
    objrot_initial = objrot_initial.T

    return objpos_initial, objrot_initial


def getlimitdegree(objpos, rotateaxis):
    objpos_vertical = copy.copy(objpos)
    objrot_vertical = rm.rodrigues(rotateaxis, 90)

    criteriapoint = getrefpoint(objpos_vertical, objrot_vertical)

    i = 0
    n = [0, 0, 1]  ## 地面の法線ベクトル
    rotatecenter = getrotatecenter_after(objpos_vertical, objrot_vertical)
    while True:
        objpos_after = copy.copy(objpos)
        objrot_after = rm.rodrigues(rotateaxis, 90 + i)
        refpoint = getrefpoint(objpos_after, objrot_after)
        objpos_after += criteriapoint - refpoint

        hangedpos_after = gethangedpos(objpos_after, objrot_after)
        objcenter = getobjcenter(objpos_after, objrot_after)

        # a0 = sp.Symbol('a0')
        t = sp.Symbol('t')
        fx = sp.Symbol('fx')
        fy = sp.Symbol('fy')
        fz = sp.Symbol('fz')

        T = np.dot(t, rm.unit_vector(pulleypos - hangedpos_after))
        F = np.array([fx, fy, fz])

        rt = hangedpos_after - rotatecenter
        rg = objcenter - rotatecenter

        force_equation = Mg + F + T
        print("force_equation : ", force_equation)
        moment_equation = np.dot(rotateaxis, np.cross(rg, Mg) + np.cross(rt, T))
        print("moment_equation : ", moment_equation)

        answer = sp.solve([force_equation[0], force_equation[1], force_equation[2], moment_equation])
        print("answer = ", answer)

        if len(answer) != 0:
            if answer[t] > 0 and answer[fz] > 0:
                if answer[fx] ** 2 + answer[fy] ** 2 < answer[fz] ** 2:
                    break

        i += 1

    return i + 90


def getpushingpath(theta, rotateaxis):
    ## Objective Function
    def func(p):
        f1, f0, t, alpha0, alpha1 = p
        n1 = np.array([-math.sin(math.radians(theta)), math.cos(math.radians(theta))])
        l1 = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
        n0 = np.array([0, 1])
        l0 = np.array([1, 0])
        F1 = np.dot(f1, n1) + np.dot(alpha1 * myu1 * f1, l1)
        F0 = np.dot(f0, n0) + np.dot(alpha0 * myu0 * f0, l0)
        T = t * t_dir
        opt = 150 * np.dot(F1, F1) + np.dot(T, T) + np.dot(F0, F0)
        return opt

    ## Constraints
    def force_eq1(p):
        f1, f0, t, alpha0, alpha1 = p
        n1 = np.array([-math.cos(math.radians(theta)), math.sin(math.radians(theta))])
        l1 = np.array([math.sin(math.radians(theta)), math.cos(math.radians(theta))])
        n0 = np.array([0, 1])
        l0 = np.array([1, 0])
        F1 = np.dot(f1, n1) + np.dot(alpha1 * myu1 * f1, l1)
        F0 = np.dot(f0, n0) + np.dot(alpha0 * myu0 * f0, l0)
        T = t * t_dir
        return T + Mg + F0 + F1

    def torque_eq1(p):
        f1, f0, t, alpha0, alpha1 = p
        n1 = np.array([-math.cos(math.radians(theta)), math.sin(math.radians(theta))])
        l1 = np.array([math.sin(math.radians(theta)), math.cos(math.radians(theta))])
        # r1_3d = copy.copy(objpos) + rdir * rotmat[:, 1]
        r1_3d = chosenpos
        r1 = np.array([r1_3d[1], r1_3d[2]])
        F1 = np.dot(f1, n1) + np.dot(alpha1 * myu1 * f1, l1)
        return np.cross(rt, t * t_dir) + np.cross(rg, Mg) + np.cross(r1, F1)

    def f1ineq(p):
        f1, f0, t, alpha0, alpha1 = p
        n1 = np.array([-math.cos(math.radians(theta)), math.sin(math.radians(theta))])
        l1 = np.array([math.sin(math.radians(theta)), math.cos(math.radians(theta))])
        # r1 = copy.copy(objpos) + r * rotmat[:, 1]
        F1 = np.dot(f1, n1) + np.dot(alpha1 * myu1 * f1, l1)
        return 30 ** 2 - np.dot(F1, F1)

    pushpath_total = []
    pushpos_total = []
    obj_total = []
    hangedpos_total = []
    pushposlist1 = []
    pushposlist2 = []
    degreelist1 = []
    degreelist2 = []

    i = 0
    endflag = 0
    pos_i = []
    pos_iminus1 = []
    pushpose_iminus1 = [pushpose_pre]
    while True:
        print("theta = ", theta)
        if theta <= 90:
            degreelist1.append(theta)
        else:
            degreelist2.append(theta)
        box = copy.deepcopy(obj)
        objpos = copy.copy(objpos_start)
        rot = rm.rodrigues(rotateaxis, theta)
        rot_ver = rm.rodrigues(rotateaxis, 90)
        refpos_ver = getrefpoint(objpos_start, rot_ver)
        if theta > 90:
            refpos = getrefpoint(objpos, rot)
            objpos += refpos_ver - refpos
        mat = rm.homobuild(objpos, rot)
        box.setMat(base.pg.np4ToMat4(mat))
        box.setColor(.8, .6, .3, .1)
        obj_total.append(box)
        # box.reparentTo(base.render)
        pushposlist = []
        nextpushlenlist = []
        for num in range(10):
            pushpos = copy.copy(objpos) + (w / 2) * rot[:, 0] + (l - 10 - num * 10) * rot[:, 1]
            pushposlist.append(pushpos)
            # base.pggen.plotSphere(base.render, pos=pushpos, major_radius=5, rgba=[0, 1, 0, 1])
            ## ------ for plotting arrow ----------------------------------------------
            # if i == 1:
            #     print("pre to pos : ", np.linalg.norm(pushpos - pos_iminus1[0]))
            #     nextpushlenlist.append(np.linalg.norm(pushpos - pos_iminus1[0]))
            #     base.pggen.plotArrow(base.render, spos=pos_iminus1[0], epos=pushpos,
            #                          axis_length=np.linalg.norm(pushpos - pos_iminus1[0]), major_radius=1.0, rgba=[0, 0, 1, 1])
            # elif i >= 2:
            #     print("pre to pos : ", np.linalg.norm(pushpos - pos_i[0]))
            #     nextpushlenlist.append(np.linalg.norm(pushpos - pos_i[0]))
            #     base.pggen.plotArrow(base.render, spos=pos_i[0], epos=pushpos,
            #                          axis_length=np.linalg.norm(pushpos - pos_i[0]), major_radius=1.0, rgba=[0, 0, 1, 1])

            # base.pggen.plotSphere(base.render, pos=pushpos, rgba=[0,1,0,1], major_radius=5)
            ## --------------------------------------------------------------------------
        hangedpos = gethangedpos(objpos, rot)
        hangedpos_total.append(hangedpos)
        # base.pggen.plotSphere(base.render, pos=hangedpos, major_radius=10, rgba=[0,1,0,1])
        rotatecenter = getrotatecenter(objpos, rot)
        # base.pggen.plotSphere(base.render, pos=rotatecenter, major_radius=10, rgba=[0,0,1,1])
        if theta > 90:
            rotatecenter = getrotatecenter_after(objpos, rot)
        objcenter = getobjcenter(objpos, rot)
        # base.pggen.plotSphere(base.render, pos=objcenter, major_radius=10, rgba=[1,0,0,1])
        rg_3d = objcenter - rotatecenter
        rt_3d = hangedpos - rotatecenter
        rg = np.array([rg_3d[1], rg_3d[2]])
        rt = np.array([rt_3d[1], rt_3d[2]])
        optimizationlist = np.empty((0, 2))
        t_dir_3d = pulleypos - hangedpos
        t_dir = rm.unit_vector([t_dir_3d[1], t_dir_3d[2]])
        bounds = ((0, 30), (0, np.inf), (0, np.inf), (-1, 1), (-1, 1))
        conds = ({'end_type': 'eq', 'fun': force_eq1},
                 {'end_type': 'eq', 'fun': torque_eq1},
                 {'end_type': 'ineq', 'fun': f1ineq})
        p0 = [30, 30, 30, 0, 0]
        minimizer_kwargs = {"method": "SLSQP", "constraints": conds, "bounds": bounds}
        for chosenpos in pushposlist:
            if i == 0:
                o = basinhopping(func, p0, minimizer_kwargs=minimizer_kwargs)
                value = func(o.x)
                print(o.x)
                optimizationlist = np.append(optimizationlist, np.array([np.array([chosenpos, value])]), axis=0)
            elif i == 1:
                if np.linalg.norm(chosenpos - pos_iminus1[0]) <= vmax * timestep:
                    o = basinhopping(func, p0, minimizer_kwargs=minimizer_kwargs)
                    value = func(o.x)
                    print(o.x)
                    optimizationlist = np.append(optimizationlist, np.array([np.array([chosenpos, value])]), axis=0)
            else:
                vec_current = rm.unit_vector(chosenpos - pos_i[0])
                vec_previous = rm.unit_vector(pos_i[0] - pos_iminus1[0])
                angle = rm.degree_betweenvector(vec_current, vec_previous)
                print("angle=", angle)
                # if np.linalg.norm(chosenpos - pos_iminus1[0]) * 0.001 <= vmax * timestep and angle <= anglemax:
                if np.linalg.norm(chosenpos - pos_i[0]) <= vmax * timestep and angle <= anglemax:
                    o = basinhopping(func, p0, minimizer_kwargs=minimizer_kwargs)
                    value = func(o.x)
                    print(o.x)
                    optimizationlist = np.append(optimizationlist, np.array([np.array([chosenpos, value])]), axis=0)
        ## sorting the pushposlist by value
        optimizationlist = ascendingorder(optimizationlist, axis=1)
        for j, eachpos in enumerate(optimizationlist):
            maximumvaluepos = eachpos[0]
            # if i == 0:
            #     pos_iminus1 = [maximumvaluepos]
            # elif i == 1:
            #     pos_i = [maximumvaluepos]
            # else:
            #     pos_iminus1[0] = pos_i[0]
            #     pos_i[0] = maximumvaluepos
            ## kinematic constraint
            pushpose = robot_s.ik("lft_arm", maximumvaluepos, pushrot, seed_jnt_values=pushpose_iminus1[0])
            if pushpose is not None:
                if i == 0:
                    pos_iminus1 = [maximumvaluepos]
                    pushpose_iminus1 = [pushpose]
                    pushpos_total.append(maximumvaluepos)
                    break
                else:
                    vec = rm.unit_vector(maximumvaluepos - pos_iminus1[0])
                    length = np.linalg.norm(maximumvaluepos - pos_iminus1[0])
                    pushpath = ctcallback_lft.getLinearPrimitivenothold(pushpose_iminus1[0], vec, length, obscmlist,
                                                                        type="source")
                    if len(pushpath) > 0:
                        if i == 1:
                            pos_i = [maximumvaluepos]
                        else:
                            pos_iminus1 = pos_i[0]
                            pos_i = [maximumvaluepos]
                        pushpose_iminus1 = [pushpose]
                        pushpath_total.append(pushpath)
                        pushpos_total.append(maximumvaluepos)
                        break

                # if theta <= 90:
                #     pushposlist1.append(eachpos[0])
                # else:
                #     pushposlist2.append(eachpos[0])
                # break

        if endflag == 1:
            break

        ## renew theta
        rotatedegree = 5
        if 90 <= theta + rotatedegree < 95:
            rotatedegree = 90 - theta

        if theta + rotatedegree > limitdegree:
            rotatedegree = limitdegree - theta
            endflag = 1

        theta += rotatedegree
        i += 1

    return pushpath_total, pushpos_total, obj_total, hangedpos_total


def getforce(armname):
    if armname == "rgt":
        rgtarm_ftsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rgtarm_ftsocket.connect(uc.rgtarm_ftsocket_ipad)
        targetftsocket = rgtarm_ftsocket
    else:
        lftarm_ftsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lftarm_ftsocket.connect(uc.lftarm_ftsocket_ipad)
        targetftsocket = lftarm_ftsocket
    rawft = targetftsocket.recv(1024)
    rawft_letter = str(rawft)
    rawft_letter_split = rawft_letter.split("(")[1].split(")")[0].split(" , ")
    rawft_final = [float(i) for i in rawft_letter_split]
    print("force ", rawft_final)

    def getforcenorm(force_array):
        force_norm = force_array[0] * force_array[0] + force_array[1] * force_array[1] + force_array[2] * force_array[2]
        # force_norm = np.linalg.norm(force_array)
        force_norm = math.sqrt(force_norm)
        return force_norm

    targetftsocket.close()
    return getforcenorm(rawft_final)


def zeroforce(armname):
    pblder = pb.ProgramBuilder()
    pblder.loadprog("zerosensor.script")
    progzerosensor = pblder.ret_program_to_run()
    if armname == "lft":
        uc.lftarm.send_program(progzerosensor)
    elif armname == "rgt":
        uc.rgtarm.send_program(progzerosensor)


if __name__ == "__main__":

    base = wd.World(cam_pos=[7, 0, 2.2], lookat_pos=[0, 0, .7])
    gm.gen_frame().attach_to(base)
    board = cm.gen_box(xyz_lengths=[.4, .5, .01])  # RANSAC用
    rtq85_s = rtq85.Robotiq85()
    robot_s = ur3ds.UR3Dual()
    robot_inik_solver = inik.IncrementalNIK(robot_s)
    ## 物体の読み込み
    ropeobj = cm.CollisionModel(initor="./research_box_mm.stl")
    obj = cm.CollisionModel(initor=objpath)
    ## 事前定義把持とハンド姿勢の呼び出し
    handpose = pose.PoseMaker()
    predefined_grasps_lft, handdirlist_lft = handpose.lftgrasppose()  # 予備の把持姿勢と把持方向
    predefined_grasps_rgt, handdirlist_rgt = handpose.rgtgrasppose()
    # for grasp_pose in predefined_grasps_rgt:
    #     ee_values, jaw_center_pos, gripper_root_pos, gripper_root_rotmat = grasp_pose
    #     mgm.gen_frame(gripper_root_pos, gripper_root_rotmat).attach_to(base)
    # base.run()
    # objects
    test = copy.deepcopy(obj)
    rotmat = rm.rotmat_from_axangle(rotate_axis, math.radians(45))
    homomat = rm.homomat_from_posrot(objpos_start, rotmat)
    test.set_homomat(homomat)
    test.set_rgba([.8, .6, .3, .4])
    test.attach_to(base)
    next = copy.deepcopy(obj)
    rotmat = rm.rotmat_from_axangle(rotate_axis, math.radians(58))
    homomat = rm.homomat_from_posrot(objpos_start, rotmat)
    next.set_homomat(homomat)
    next.set_rgba([0, 1, 0, .4])
    next.attach_to(base)
    obscmlist = []
    obscmlist.append(test)
    obscmlist.append(next)

    # ## 確認用の者　右手の引き動作と対象物の移動
    arm_name = 'rgt_arm'
    rgt_jnt_values = np.radians(
        np.array([-16.40505261, -52.96523856, 91.11206022, 36.08211617, 132.71248608, 67.39504932]))
    robot_s.fk(component_name=arm_name, jnt_values=rgt_jnt_values)
    robot_s.gen_meshmodel().attach_to(base)
    # rgt_pos, rgt_rotmat = robot_s.get_gl_tcp(hnd_name=arm_name)
    # ropelinesorted = []
    # dir = rm.unit_vector(ropetoppos - rgt_pos)
    # path = inik_solver.gen_rel_linear_motion(hnd_name=arm_name,
    #                                                goal_tcp_pos=rgt_pos,
    #                                                goal_tcp_rotmat=rgt_rotmat,
    #                                                motion_vec=-dir,
    #                                                linear_distance=.15,
    #                                                end_type='source')
    # # for conf in path:
    # #     robot_s.fk(arm_name, conf)
    # #     robot_s.gen_meshmodel().attach_to(base)
    # # base.run()
    # robot_s.fk(hnd_name=arm_name, jnt_values=path[-1])
    # # robot_s.gen_meshmodel().attach_to(base)
    # # base.run()
    # rgt_pos, rgt_rotmat = robot_s.get_gl_tcp(hnd_name=arm_name)
    # counter = 0
    # while True:
    #     rgt_append_pos = rgt_pos + dir * counter * 1e-3
    #     ropelinesorted.append(rgt_append_pos)
    #     counter += 1
    #     if rgt_append_pos[2] > 1.7:
    #         break
    # ropelinesorted = ropelinesorted[::-1]
    # for rope_point in ropelinesorted:
    #     mgm.gen_sphere(rope_point).attach_to(base)
    # # base.run()
    # 
    # # 左手の引き動作
    # startpointid = 0  # 1.7
    # hold_pos_final = np.array([.25, -.15, 1.4])  # 仮設置
    # arm_name = "rgt_arm"
    # IKpossiblelist_start, hold_pos_init, hold_rot_init, startpointid = decidestartpose(ropelinesorted,
    #                                                                                    arm_name,
    #                                                                                    predefined_grasps_rgt,
    #                                                                                    robot_s.rgt_arm.home,
    #                                                                                    startpointid)
    # # for data in IKpossiblelist_start:
    # #     robot_s.fk(arm_name, data[0])
    # #     robot_s.gen_meshmodel().attach_to(base)
    # # base.run()
    # IKpossiblelist_startgoal = decidegoalpose_onepoint(arm_name,
    #                                                    IKpossiblelist_start,
    #                                                    hold_pos_final,
    #                                                    predefined_grasps_rgt,
    #                                                    obscmlist)
    # rgtstart = IKpossiblelist_startgoal[0][0]
    # rgtgoal = IKpossiblelist_startgoal[0][1]
    # robot_s.fk(arm_name, rgtstart)
    # rgtstart_pos, rgtstart_rotmat = robot_s.get_gl_tcp(arm_name)
    # robot_s.gen_meshmodel().attach_to(base)
    # robot_s.fk(arm_name, rgtgoal)
    # robot_s.gen_meshmodel().attach_to(base)
    # # base.run()
    # 
    # dir = rm.unit_vector(hold_pos_final - ropelinesorted[0])
    # axis_length = np.linalg.norm(hold_pos_final - ropelinesorted[0])
    # path = inik_solver.gen_rel_linear_motion(arm_name, rgtstart_pos, rgtstart_rotmat, dir, axis_length, [],
    #                                                end_type="source")
    # for conf in path:
    #     robot_s.fk(arm_name, conf)
    #     robot_s.gen_meshmodel().attach_to(base)
    # # base.run()

    ## ------ シミュレーション用 ------
    ropelinesorted = []
    for i in range(200):
        ropelinesorted.append(np.array([.25, 0, 1.65 - i * 1e3]))
    # ropelinesorted = np.load('RopeVertex_test.npy')
    np.save('RopeVertex_test.npy', ropelinesorted)
    ## ------------------------------

    ## ------ 実際のロープの点群 ------
    # pointrange = [.1, .4, -.15, .15, 1.2, 1.8]
    # newpcd = getpointcloudkinect(pointrange)
    # ropelinesorted = doRANSAC(newpcd, 5)
    # np.save('RopeVertex_data.npy', ropelinesorted)
    # -----------------------

    ## todo プッシングのシミュレーションの際はコメントアウト
    ## 開始(右手で引く)
    startpointid = 0
    endflag = 0
    while True:
        arm_name = 'rgt_arm'
        print("startpointid", startpointid)
        IKpossiblelist_start_rgt, objpos_initial_rgt, objrot_initial_rgt, startpointid = decidestartpose(arm_name,
                                                                                                         ropelinesorted,
                                                                                                         predefined_grasps_rgt,
                                                                                                         robot_s.rgt_arm.home_conf,
                                                                                                         startpointid)
        objpos_finallist_rgt = create_candidate_points(arm_name=arm_name, initialhandpos=objpos_initial_rgt)
        np.save('finalposlist.npy', objpos_finallist_rgt)
        # for point in objpos_finallist_rgt:
        #     mgm.gen_sphere(pos=point).attach_to(base)
        # base.run()
        assessment_value_list_rgt, totalIKpossiblelist_startgoal_rgt, costlist_rgt = \
            getsuitablegoalpos_first(arm_name,
                                     IKpossiblelist_start_rgt,
                                     objpos_initial_rgt,
                                     objpos_finallist_rgt,
                                     predefined_grasps_rgt)
        for i, each_assessment_value_list in enumerate(assessment_value_list_rgt):
            useid = each_assessment_value_list[1]  ## 評価値が高いものから順に選択
            print("useid", useid)
            IKpossiblelist_startgoal_rgt = totalIKpossiblelist_startgoal_rgt[int(useid)]
            togoalvec_rgt = objpos_finallist_rgt[int(useid)] - objpos_initial_rgt
            tostartvec_rgt = copy.copy(togoalvec_rgt) * (-1)
            use_objpos_final_rgt = copy.copy(objpos_initial_rgt) + togoalvec_rgt
            IKpossiblelist_rgt = decidemidpose(arm_name,
                                               IKpossiblelist_startgoal_rgt,
                                               handdirlist_rgt,
                                               objpos_final=use_objpos_final_rgt)
            ropepulling_rgt, usingposelist_rgt, usingposeid_rgt = ropepullingmotion(IKpossiblelist_rgt, togoalvec_rgt,
                                                                                    ctcallback_rgt)
            print("ropepulling_rgt", ropepulling_rgt)
            gotoinitialrgtpoint = RRTmotion(rbt.initjnts[3:9], usingposelist_rgt[2][0], ctcallback_rgt, obscmlist, 30,
                                            10)

            if len(ropepulling_rgt) > 0 and gotoinitialrgtpoint is not False:
                endflag = 1
                ## はじめにロープを引く長さ
                d_0 = np.linalg.norm(objpos_initial_rgt - use_objpos_final_rgt)
                theta_0 = 0
                theta_1 = 0
                break

        if endflag == 1:
            break

        startpointid += 1
        # if startpointid == len(ropelinesorted):

    ## 終点に対する中継点と、経路を保存
    keeplist = [usingposelist_rgt[2][1], usingposelist_rgt[3][1], tostartvec_rgt]
    pickle.dump(keeplist, open("keeplist.pickle", "wb"))
    # # -------------------------------------------------------------------------------
    # ## todo プッシングのシミュレーションの際はコメントアウト

    ### todo pushingのシミュレーション用
    ## 実環境上
    # newpcd = getpointcloudkinect(objpointrange)
    # base.pg.genPntsnp(newpcd).reparentTo(base.render)
    # # base.run()
    # refpoint_fitting = np.array([[-200, -250, 0, 1],
    #                              [-200, 250, 0, 1],
    #                              [200, 250, 0, 1],
    #                              [200, -250, 0, 1]])
    # targetpointnew, refpoint_fitting = objectfitting(newpcd, board, refpoint_fitting)
    # xaxis_obj, yaxis_obj, zaxis_obj, theta, cog = getobjaxis(targetpointnew, refpoint_fitting)
    # objpos_initial_board, objrot_initial_board = getobjposandrot(zaxis_obj)
    # objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
    # currentobj = copy.deepcopy(obj)
    # currentobj.setMat(base.pg.np4ToMat4(objmat4))
    # currentobj.setColor(1, 0, 0, .5)
    # currentobj.reparentTo(base.render)
    # print(theta)
    # base.run()
    # pickle.dump(currentobj, open("obj.pickle", "wb"))

    ## シミュレータ上
    # theta_sim = [66]
    # theta = theta_sim[0]
    # # experimentlist = [[gotoinitialrgtpoint + usingposelist_rgt[3][0], 85, 0, "rgt", "gotoinitialrgtpoint", 2.0]]
    #
    # hold_pos_init = copy.copy(objpos_start)
    # hold_rot_init = rm.rodrigues(rotate_axis, theta)
    # # test = copy.deepcopy(obj)
    # # mat = rm.homobuild(hold_pos_init, hold_rot_init)
    # # test.setMat(base.pg.np4ToMat4(mat))
    # # test.reparentTo(base.render)
    # # robot_s.movearmfk(pushpose_pre, "lft")
    # # rbtmg.genmnp(robot_s, togglejntscoord=False).reparentTo(base.render)
    # # base.run()
    #
    # ## pushingの動作計画
    # # limitdegree = getlimitdegree(hold_pos_init, rotate_axis)
    # limitdegree = 100
    # ## test
    # pushpath_total, pushpos_total, obj_total, hangedpos_total = getpushingpath_first(obj, hold_pos_init,  hold_rot_init,  theta, rotate_axis)
    # print("pushpath_total", pushpath_total)
    #
    # obj_ver = copy.deepcopy(obj_total[-1])
    # objmat4_ver = obj_ver.gethomomat()
    # objpos_ver = objmat4_ver[:3, 3]
    # objrot_ver = objmat4_ver[:3, :3]
    #
    # pushpos = pushpos_total[-1]
    # pushpath_total2, pushpos_total2, obj_total2, hangedpos_total2 = getpushingpath_second(obj_ver, objpos_ver, objrot_ver, pushpos, limitdegree, rotate_axis)
    # print("pushpath_total2", pushpath_total2)
    #
    # # pushpath_all = np.concatenate([pushpath_total, pushpath_total2])
    # # pushpos_all = np.concatenate([pushpos_total, pushpos_total2])
    # # obj_all = np.concatenate([obj_total, obj_total2])
    # # looselength_all = np.concatenate([looselength_total, looselength_total2])
    # pushpath_all = list(pushpath_total) + list(pushpath_total2)
    # pushpos_all = list(pushpos_total) + list(pushpos_total2)
    # obj_all = list(obj_total) + list(obj_total2)
    # hangedpos_all = list(hangedpos_total) + list(hangedpos_total2)
    #
    # pickle.dump(obj_all[0], open("obj.pickle", "wb"))
    # pickle.dump([pushpath_all, pushpos_all, obj_all, hangedpos_all], open("pushinglist.pickle", "wb"))
    ## todo pushingのシミュレーション用
    # base.run()

    ### シミュレーション　& 実機
    ### Order of "experimentlist":[path, startjawwidth, endjawwidth, arm_name, pathlable, timepathstep]
    predictlist = [d_0, theta_1, theta_0]
    experimentlist = [[gotoinitialrgtpoint + usingposelist_rgt[3][0], 85, 0, "rgt", "gotoinitialrgtpoint", 2.0],
                      [ropepulling_rgt, 0, 0, "rgt", "ropepulling_rgt", 1.0]]

    ## シミュレーション用
    theta_sim = [0]

    pushstartpoint = [0]
    motioncounter = [0]
    pathcounter = [0]
    activelist = experimentlist[pathcounter[0]]
    ## ロープ引きの際のカウンタ
    pullingcounter = [0]
    pullingcounter_rgt = [0]
    pullingcounter_lft = [0]
    forcelist = []
    ropepullingforcelist = []
    used_pushpathlist = []
    ## 押し動作の際のカウンタ
    forcecounter = [0]
    pushcounter = [0]
    forceflag = [0]
    stopflag = [0]
    rbtmnpani = [None]
    objmnpani = [None]
    pntani = [None]
    finalposani = [None]
    endpoint = [1000]

    ropepullingflag = [1]
    pushingflag = [0]
    ropeloosningflag = [0]
    ropepullendflag = [0]
    rgtregraspflag = [0]
    onearmflag = [0]
    pushendflag = [0]
    getgoalcounter = [0]

    # obj_current = None
    finalpos = [None]
    rot = rm.rodrigues(rotate_axis, theta_sim[0])
    mat = rm.homobuild(objpos_start, rot)
    obj_current = copy.deepcopy(obj)
    obj_current.setMat(base.pg.np4ToMat4(mat))
    pickle.dump(obj_current, open("obj.pickle", "wb"))


    def updatemotionsec(activelist, rbtmnp, objmnp, motioncounter, rbt, pnt, finalpos, task):
        if motioncounter[0] < len(activelist[0]):
            if rbtmnp[0] is not None:
                rbtmnp[0].detachNode()
            if objmnp[0] is not None:
                objmnp[0].detachNode()
            if pnt[0] is not None:
                pnt[0].detachNode()
            if finalpos[0] is not None:
                finalpos[0].detachNode()
            ## ロボットの表示
            pose = activelist[0][motioncounter[0]]
            armname = activelist[3]
            rbt.movearmfk(pose, armname)
            rbtmnp[0] = rbtmg.genmnp(rbt)
            rbtmnp[0].reparentTo(base.render)
            ## ロープの点群の表示
            # rope = np.load('RopeVertex_test.npy')  ## テスト用
            rope = np.load('RopeVertex_data.npy')  ## 実際の点群
            pnt[0] = base.pg.genPntsnp(rope, colors=[[1, 0, 0, 1] for x in rope], pntsize=5)
            pnt[0].reparentTo(base.render)
            ## 終点のリストの表示
            poslist = np.load('finalposlist.npy')
            if poslist != []:
                finalpos[0] = base.pg.genPntsnp(poslist, colors=[[0, 1, 0, 1] for x in poslist], pntsize=5)
                finalpos[0].reparentTo(base.render)
            ## 吊るした物体の表示
            obj_current = pickle.load(open("obj.pickle", "rb"))
            objmnp[0] = obj_current
            objmnp[0].setColor(.8, .6, .3, .5)
            objmnp[0].reparentTo(base.render)
            motioncounter[0] += 1

        return task.again


    taskMgr.doMethodLater(0.1, updatemotionsec, "updatemotionsec",
                          extraArgs=[activelist, rbtmnpani, objmnpani, motioncounter, rbt, pntani, finalposani],
                          appendTask=True)


    def updatesection(rbtmnp, objmnp, motioncounter, rbt, pnt, finalpos, task):
        if base.inputmgr.keyMap['space'] is True:
            activelist_real = experimentlist[pathcounter[0]]
            ## ---- 実機 ----
            # timepathstep = activelist_real[5]
            # zeroforce(arm_name=activelist_real[3])
            # uc.opengripper(arm_name=activelist_real[3], fingerdistance=activelist_real[1])
            # # time.sleep(1)
            # zeroforce(arm_name=activelist_real[3])
            # if len(activelist_real[0]) > 0:
            #     uc.movejntssgl(activelist_real[0][0],arm_name=activelist_real[3])
            #     uc.movejntssgl_cont(activelist_real[0], arm_name=activelist_real[3], timepathstep=timepathstep)
            # uc.opengripper(arm_name=activelist_real[3], fingerdistance=activelist_real[2])
            #
            # ###
            # # if activelist_real[4] == "lftpush" and 70 < rotatecheck[0] + rotatedegree < 75:
            # rgt_pos = robot_s.getee("rgt")[0]
            # if activelist_real[4] == "lftpush" and rgt_pos[2] >= 1700:
            #     uc.opengripper("rgt")
            #     rgtregraspflag[0] = 1
            # ###
            #
            # # if activelist_real[4] == "ropepulling_lft" or activelist_real[4] == "ropepulling_rgt":
            # #     # time.sleep(1)
            # #     # zeroforce(arm_name=activelist_real[3])
            # #     time.sleep(1)
            # #     ropepullingforce = getforce(arm_name=activelist_real[3])
            # #     print("引いた際の力：", ropepullingforce)
            # #     ropepullingforcelist.append(ropepullingforce)
            # #     print("力のリスト：", ropepullingforcelist)
            # if pathcounter[0] == endpoint[0]:
            #     print("実験終了です")
            #     return False
            ## -------------

            ## シミュレーション用
            # print("label:", activelist_real[4])
            # if activelist_real[4] == "lftpush" and robot_s.getee("rgt")[0][2] > 1700:
            #     rgtregraspflag[0] = 1

            ## ----------ロープを引く---------
            if ropepullingflag[0] == 1:
                if activelist_real[4] == "ropepulling_lft":
                    pullingcounter[0] += 1
                    pullingcounter_lft[0] += 1
                    print("左手で" + str(pullingcounter_lft) + "回引きました")
                    currentrgtjnt = rbt.getarmjnts("rgt")
                    currentlftjnt = rbt.getarmjnts("lft")
                    print("currentrgtjnt", currentrgtjnt)
                    print("curentlftjnt", currentlftjnt)

                    ## 実際の点群 TODO シミュレーションの際はコメントアウト
                    newpcd = getpointcloudkinect(objpointrange)

                    ## ransac and icp, θを計算(実機)
                    refpoint_fitting = np.array([[-w / 2, -l / 2, 0, 1],
                                                 [-w / 2, l / 2, 0, 1],
                                                 [w / 2, l / 2, 0, 1],
                                                 [w / 2, -l / 2, 0, 1]])
                    targetpointnew, refpoint_fitting = objectfitting(newpcd, board, refpoint_fitting)
                    pickle.dump(targetpointnew, open("targetpointnew.pickle", "wb"))
                    xaxis_obj, yaxis_obj, zaxis_obj, theta, cog = getobjaxis(targetpointnew, refpoint_fitting)
                    print("theta = ", theta)
                    if theta > thetathreshold:
                        ropepullendflag[0] = 1
                    ## thetaの結果を代入
                    predictlist[1] = theta
                    print("predictlist = ", predictlist)
                    # objpos_initial_board, objrot_initial_board = getobjposandrot(cog, xaxis_obj, yaxis_obj, zaxis_obj)
                    objpos_initial_board, objrot_initial_board = getobjposandrot(zaxis_obj)
                    objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    obj_current = copy.deepcopy(obj)
                    obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    obj_current.setColor(1, 0, 0, .5)
                    # objmnp[0] = obj_current
                    pickle.dump(obj_current, open("obj.pickle", "wb"))
                    print("傾き：", theta)
                    ## ------------------------------------

                    ## 物体の情報を更新(シミュレーション用)
                    # theta_sim[0] += 5
                    # pickle.dump(theta_sim[0], open("thetacheck.pickle", "wb"))
                    # print("theta = ", theta_sim[0])
                    # predictlist[1] = theta_sim[0]
                    # print("predictlist = ", predictlist)
                    # objpos_initial_board = copy.copy(objpos_start)
                    # objrot_initial_board = rm.rodrigues(rotate_axis, theta_sim[0])
                    # objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    # obj_current = copy.deepcopy(obj)
                    # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    # pickle.dump(obj_current, open("obj.pickle", "wb"))
                    ## ---------------------------------

                    ## 引き上げた後の物体を障害物として追加
                    # obj_current = copy.deepcopy(obj)
                    # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    obscmlist.append(obj_current)
                    ## ------------------------------------------

                    if ropepullendflag[0] == 1:
                        print("十分引き上げました。プッシングします")
                        np.save('finalposlist.npy', [])
                        ropepullingflag[0] = 0
                        pushingflag[0] = 1
                    else:
                        objpos_final = pickle.load(open("hold_pos_final.pickle", "rb"))
                        print("hold_pos_final", objpos_final)
                        midgoaljnt_lft, midpathgoal_lft, tostartvec = pickle.load(open("keeplist.pickle", "rb"))

                        ## ---シミュレーション用---
                        # ropelinesorted = []
                        # currentlftpos = robot_s.getee("lft")[0]
                        # dir = rm.unit_vector(ropetoppos - currentlftpos)
                        # i = 0
                        # while True:
                        #     pos = currentlftpos + (i+1)*dir
                        #     if pos[2] <= 1700:
                        #         ropelinesorted.append(pos)
                        #         i += 1
                        #     else:
                        #         break
                        # ropelinesorted = ropelinesorted[::-1]
                        # np.save('RopeVertex_test.npy', ropelinesorted)
                        # # ropelinesorted = np.load('RopeVertex_test.npy')
                        ## ----------------------

                        ## ---実験用(点群を取得)---　TODO シミュレーションの際はコメントアウト
                        ## 上側
                        eeposlft = rbt.getee("lft")[0]
                        pointrange = [50, eeposlft[0] + 100, eeposlft[1] - 100, eeposlft[1] + 100, eeposlft[2] + 50,
                                      1700]
                        newpcd = getpointcloudkinectforrope_up(rbt, "lft", ropetoppos, pointrange)
                        ropeline_up = doRANSAC(newpcd, 5)
                        ropelinesorted = descendingorder(ropeline_up, 2)

                        ## 下側
                        # pointrange_under = [50, eeposlft[0] + 100, eeposlft[1] - 100, eeposlft[1] + 100, eeposlft[2] - 250, eeposlft[2] - 50]
                        # newpcd = getpointcloudkinectforrope_down(robot_s, "lft", pointrange_under)
                        # ropeline_under = doRANSAC(newpcd, 10)
                        # ropelinesorted_under = descendingorder(ropeline_under, axis=2)
                        #
                        # # ropelinesorted = ropelinesorted_up + ropelinesorted_under
                        # ropelinesorted = np.concatenate([ropelinesorted_up, ropelinesorted_under])
                        np.save('RopeVertex_data.npy', ropelinesorted)
                        ## ---------------------

                        endflag = 0
                        startpointid = 0
                        while True:
                            print("startpointid", startpointid)
                            # robot_s.goinitpose()
                            ## 始点について計算
                            IKpossiblelist_start_rgt, objpos_initial_rgt, objrot_initial_rgt, startpointid = decidestartpose(
                                ropelinesorted, "rgt",
                                predefined_grasps_rgt, ctcallback_rgt,
                                currentrgtjnt, startpointid)

                            ## 終点について計算
                            # obstacles = [obj_current]
                            objpos_finallist_rgt = create_candidate_points(arm_name="rgt",
                                                                           initialhandpos=objpos_initial_rgt)
                            np.save('finalposlist.npy', objpos_finallist_rgt)

                            assessment_value_list_rgt, totalIKpossiblelist_startgoal_rgt, costlist_rgt, elements_for_predictlist_rgt, togoalveclist_rgt = getsuitablegoalpos_second(
                                IKpossiblelist_start_rgt, objpos_initial_rgt,
                                objpos_finallist_rgt, "rgt", predefined_grasps_rgt,
                                ctcallback_rgt, predictlist)

                            for i, each_assessment_value_list in enumerate(assessment_value_list_rgt):
                                useid = each_assessment_value_list[1]
                                use_element = elements_for_predictlist_rgt[i]
                                togoalvec_rgt = togoalveclist_rgt[i]
                                print("useid", useid)
                                IKpossiblelist_startgoal_rgt = totalIKpossiblelist_startgoal_rgt[int(useid)]
                                if IKpossiblelist_startgoal_rgt is False:
                                    continue

                                d_before, theta_before, theta_beforebefore = predictlist
                                ## 予測値
                                d_next, theta_next, use_objpos_final_rgt, flag = use_element
                                print("d_next = ", d_next)
                                print("theta_next = ", theta_next)
                                print("objpos final = ", use_objpos_final_rgt)
                                ropepullendflag[0] = flag

                                IKpossiblelist_rgt = decidemidpose(IKpossiblelist_startgoal_rgt, handdirlist_rgt, "rgt",
                                                                   ctcallback_rgt, objpos_final=use_objpos_final_rgt)
                                ropepulling_rgt, usingposelist_rgt, usingposeid = ropepullingmotion(IKpossiblelist_rgt,
                                                                                                    togoalvec_rgt,
                                                                                                    ctcallback_rgt)
                                if ropepulling_rgt is False:
                                    continue

                                if pullingcounter[0] == 1:
                                    gotoinitialrgtpoint = RRTmotion(rbt.initjnts[3:9], usingposelist_rgt[2][0],
                                                                    ctcallback_rgt, obscmlist, 30, 10)

                                    if len(ropepulling_rgt) > 0 and gotoinitialrgtpoint is not False:
                                        print("gotoinitialrgtpath", gotoinitialrgtpoint)
                                        experimentlist.append(
                                            [gotoinitialrgtpoint + usingposelist_rgt[3][0], 85, 0, "rgt",
                                             "gotoinitialrgtpoint", 2.0])

                                        endflag = 1
                                        predictlist[0] = d_next
                                        predictlist[2] = theta_before
                                        break

                                else:
                                    rgtmidpath = RRTmotion(currentrgtjnt, usingposelist_rgt[2][0], ctcallback_rgt,
                                                           obscmlist, 30, 10)

                                    if len(ropepulling_rgt) > 0 and rgtmidpath is not False:
                                        print("rgtmidpath", rgtmidpath)
                                        experimentlist.append(
                                            [rgtmidpath + usingposelist_rgt[3][0], 85, 0, "rgt", "rgtmidpath", 2.0])
                                        endflag = 1
                                        predictlist[0] = d_next
                                        predictlist[2] = theta_before
                                        break

                            if endflag == 1:
                                break

                            startpointid += 1

                        keeplist = [usingposelist_rgt[2][1], usingposelist_rgt[3][1], tostartvec]
                        pickle.dump(keeplist, open("keeplist.pickle", "wb"))

                        obj_next_predict = copy.deepcopy(obj)
                        pos = copy.copy(objpos_start)
                        rot = rm.rodrigues(rotate_axis, theta_next)
                        mat = rm.homobuild(pos, rot)
                        obj_next_predict.setMat(base.pg.np4ToMat4(mat))

                        experimentlist.append([midpathgoal_lft, 85, 85, "lft", "moveto_midgoal", 1.0])
                        ## 引いた後の姿勢が物体と衝突するとき
                        if ctcallback_lft.iscollided(currentlftjnt, [obj_next_predict]) is True:
                            print("next collided")
                            startpointid = 0
                            endflag = 0
                            while True:
                                rbt.goinitpose()
                                print("regrasppath search : ", str(startpointid + 1) + "回目")
                                IKpossiblelist_start, objpos_initial, objrot_initial, startpointid = decidestartpose(
                                    ropelinesorted, "lft",
                                    predefined_grasps_lft, ctcallback_lft,
                                    currentlftjnt, startpointid)
                                # print("IKpossiblelist_start test", IKpossiblelist_start)
                                # print("len test", len(IKpossiblelist_start))
                                for i in IKpossiblelist_start:
                                    direction = rm.unit_vector(handdirlist_lft[i[2]]) * (-1)
                                    distance = 80
                                    flag = 0
                                    while True:
                                        preregrasppath = ctcallback_lft.getLinearPrimitivenothold(i[0], direction,
                                                                                                  distance, obscmlist,
                                                                                                  type="source")
                                        if len(preregrasppath) > 0:
                                            flag = 1
                                            break
                                        else:
                                            distance -= 1
                                            if distance < 20:
                                                break

                                    if flag == 1:
                                        preregrasppath = preregrasppath[::-1]
                                        if ctcallback_lft.iscollided(currentlftjnt, [obj_current]) is True:
                                            moveup = ctcallback_lft.getLinearPrimitivenothold(currentlftjnt, [0, 0, 1],
                                                                                              50, [], type="source")
                                            lftmidpath = RRTmotion(moveup[-1], preregrasppath[0], ctcallback_lft,
                                                                   obscmlist, 30, 10)
                                            if lftmidpath is not False:
                                                lftmidpath = list(moveup) + list(lftmidpath)
                                        else:
                                            lftmidpath = RRTmotion(currentlftjnt, preregrasppath[0], ctcallback_lft,
                                                                   obscmlist, 30, 10)
                                        if lftmidpath is not False:
                                            endflag = 1
                                            break

                                if endflag == 1:
                                    break
                                startpointid += 1
                                if startpointid == len(ropelinesorted):
                                    break
                            experimentlist.append([lftmidpath, 85, 85, "lft", "lftmidpath", 1.0])
                        experimentlist.append([ropepulling_rgt, 0, 0, "rgt", "ropepulling_rgt", 1.0])

                        rbt.movearmfk(currentrgtjnt, "rgt")
                        rbt.movearmfk(currentlftjnt, "lft")
                        obscmlist.pop(-1)

                elif activelist_real[4] == "ropepulling_rgt" and onearmflag[0] == 0:
                    pullingcounter[0] += 1
                    pullingcounter_rgt[0] += 1
                    print("右手で" + str(pullingcounter_rgt[0]) + "回引きました")
                    currentrgtjnt = rbt.getarmjnts("rgt")
                    currentlftjnt = rbt.getarmjnts("lft")

                    ## 物体の点群を検出(実際の点群)　TODO シミュレーションの際はコメントアウト
                    newpcd = getpointcloudkinect(objpointrange)

                    ## ransac and icp(実機) --------------------------------------
                    refpoint_fitting = np.array([[-w / 2, -l / 2, 0, 1],
                                                 [-w / 2, l / 2, 0, 1],
                                                 [w / 2, l / 2, 0, 1],
                                                 [w / 2, l / 2, 0, 1]])
                    targetpointnew, refpoint_fitting = objectfitting(newpcd, board, refpoint_fitting)
                    pickle.dump(targetpointnew, open("targetpointnew.pickle", "wb"))
                    xaxis_obj, yaxis_obj, zaxis_obj, theta, cog = getobjaxis(targetpointnew, refpoint_fitting)
                    print("theta = ", theta)
                    if theta > thetathreshold:
                        ropepullendflag[0] = 1
                    ## thetaの結果を代入
                    predictlist[1] = theta
                    # objpos_initial_board, objrot_initial_board = getobjposandrot(cog, xaxis_obj, yaxis_obj, zaxis_obj)
                    objpos_initial_board, objrot_initial_board = getobjposandrot(zaxis_obj)
                    objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    obj_current = copy.deepcopy(obj)
                    obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    obj_current.setColor(1, 0, 0, .5)
                    # objmnp[0] = obj_current
                    pickle.dump(obj_current, open("obj.pickle", "wb"))
                    ## -------------------------------

                    ## 物体の情報を更新(シミュレーション用) TODO 実機の際はコメントアウト
                    # theta_sim[0] += 5
                    # theta = theta_sim[0]
                    # print("theta = ", theta_sim[0])
                    # predictlist[1] = theta_sim[0]
                    # print("predictlist = ", predictlist)
                    # objpos_initial_board = copy.copy(objpos_start)
                    # objrot_initial_board = rm.rodrigues(rotate_axis, theta_sim[0])
                    # objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    # obj_current = copy.deepcopy(obj)
                    # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    # pickle.dump(obj_current, open("obj.pickle", "wb"))
                    ## ---------------------------------

                    if ropepullendflag[0] == 1:
                        print("十分引き上げました プッシングします")
                        np.save('finalposlist.npy', [])
                        ropepullingflag[0] = 0
                        pushingflag[0] = 1
                    else:
                        objpos_final = pickle.load(open("hold_pos_final.pickle", "rb"))
                        midgoaljnt_rgt, midpathgoal_rgt, tostartvec = pickle.load(open("keeplist.pickle", "rb"))

                        ## ---シミュレーション用
                        # ropelinesorted = []
                        # currentrgtpos = robot_s.getee("rgt")[0]
                        # dir = rm.unit_vector(ropetoppos - currentrgtpos)
                        # i = 0
                        # while True:
                        #     pos = currentrgtpos + (i+1)*dir
                        #     if pos[2] <= 1700:
                        #         ropelinesorted.append(pos)
                        #         i += 1
                        #     else:
                        #         break
                        # ropelinesorted = ropelinesorted[::-1]
                        # np.save('RopeVertex_test.npy', ropelinesorted)
                        ## ----------------------

                        ## ---実験用(点群を取得)-----　TODO シミュレーションの際はコメントアウト---
                        eeposrgt = rbt.getee("rgt")[0]
                        pointrange = [50, eeposrgt[0] + 100, eeposrgt[1] - 100, eeposrgt[1] + 100, eeposrgt[2] + 50,
                                      1700]
                        newpcd = getpointcloudkinectforrope_up(rbt, "rgt", ropetoppos, pointrange)
                        ropeline_up = doRANSAC(newpcd, 5)
                        ropelinesorted = descendingorder(ropeline_up, 2)

                        # pointrange = [50, eeposrgt[0] + 100, eeposrgt[1] - 100, eeposrgt[1] + 100, eeposrgt[2] - 250, eeposrgt[2] - 50]
                        # newpcd = getpointcloudkinectforrope_down(robot_s, "rgt", pointrange)
                        # ropeline_under = doRANSAC(newpcd, 5)
                        # ropelinesorted_under = descendingorder(ropeline_under, 2)

                        # ropelinesorted = ropelinesorted_up + ropelinesorted_under
                        # ropelinesorted = np.concatenate([ropelinesorted_up, ropelinesorted_under])
                        np.save('RopeVertex_data.npy', ropelinesorted)
                        ## ------------------------

                        ## 引き上げた後の物体を障害物として追加
                        # obj_current = copy.deepcopy(obj)
                        # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                        obscmlist.append(obj_current)
                        ## -------------------------

                        startpointid = 0
                        endflag = 0

                        while True:
                            print("startpointid", startpointid)
                            # robot_s.goinitpose()
                            IKpossiblelist_start_lft, objpos_initial_lft, objrot_initial_lft, startpointid = decidestartpose(
                                ropelinesorted, "lft",
                                predefined_grasps_lft, ctcallback_lft,
                                currentlftjnt, startpointid)

                            obstacles = [obj_current]
                            # obstacles = [obs]
                            objpos_finallist_lft = create_candidate_points(arm_name="lft",
                                                                           initialhandpos=objpos_initial_lft,
                                                                           obstacles=obstacles)
                            np.save('finalposlist.npy', objpos_finallist_lft)

                            assessment_value_list_lft, totalIKpossiblelist_startgoal_lft, costlist_lft, elements_for_predictlist_lft, togoalveclist_lft = getsuitablegoalpos_second(
                                IKpossiblelist_start_lft, objpos_initial_lft,
                                objpos_finallist_lft, "lft", predefined_grasps_lft,
                                ctcallback_lft, predictlist)

                            ## 通常どおりのとき
                            for i, each_assessment_value_list in enumerate(assessment_value_list_lft):
                                useid = each_assessment_value_list[1]
                                assessment_value = each_assessment_value_list[0]
                                objposfinal_height = elements_for_predictlist_lft[i][2][2]
                                diff_height = objpos_initial_lft[2] - objposfinal_height
                                print("diff_height = ", diff_height)
                                ## 評価値が負のとき、引くことができない → 左手でロープを掴み、右手のみで引けるようにする
                                if assessment_value <= 0 or diff_height < 150 or i == len(
                                        assessment_value_list_lft) - 1:
                                    onearmflag[0] = 1
                                    for uselist in IKpossiblelist_start_lft:
                                        use_startjnt = uselist[0]
                                        id_using = uselist[2]
                                        use_handdir = handdirlist_lft[id_using]
                                        direction = rm.unit_vector(use_handdir) * (-1)

                                        straightpath = ctcallback_lft.getLinearPrimitivenothold(use_startjnt, direction,
                                                                                                50, obscmlist,
                                                                                                type="source")[::-1]
                                        if len(straightpath) > 0:
                                            if ctcallback_lft.iscollided(currentlftjnt, [obj_current]) is True:
                                                print("current arm is in collision")
                                                while True:
                                                    i = 50
                                                    moveup = ctcallback_lft.getLinearPrimitivenothold(currentlftjnt,
                                                                                                      [0, 0, 1], i, [],
                                                                                                      type="source")
                                                    if ctcallback_lft.iscollided(moveup[-1], [obj_current]) is False:
                                                        break
                                                    i += 2
                                                movetostartjnt = RRTmotion(moveup[-1], straightpath[0], ctcallback_lft,
                                                                           obscmlist, 20, 10)
                                                if movetostartjnt is not False:
                                                    movetostartjnt = list(moveup) + list(movetostartjnt)
                                            else:
                                                movetostartjnt = RRTmotion(currentlftjnt, straightpath[0],
                                                                           ctcallback_lft, obscmlist, 20, 10)

                                            if movetostartjnt is not False:
                                                # experimentlist.append([movetostartjnt + straightpath, 85, 0, "lft", "onearmgrasp_lft", 1.0])
                                                experimentlist.append(
                                                    [movetostartjnt, 85, 85, "lft", "movotostartjnt", 1.0])
                                                experimentlist.append(
                                                    [straightpath, 85, 0, "lft", "onearmgrasp_lft", 1, 0])
                                                break
                                    break

                                print("useid", useid)
                                use_element = elements_for_predictlist_lft[i]
                                togoalvec_lft = togoalveclist_lft[i]
                                IKpossiblelist_startgoal_lft = totalIKpossiblelist_startgoal_lft[int(useid)]

                                ## 予測値の計算
                                d_before, theta_before, theta_beforebefore = predictlist
                                ## 予測値
                                d_next, theta_next, use_objpos_final_lft, flag = use_element
                                print("d_next = ", d_next)
                                print("theta_next = ", theta_next)
                                print("objpos final = ", use_objpos_final_lft)
                                ropepullendflag[0] = flag

                                IKpossiblelist_lft = decidemidpose(IKpossiblelist_startgoal_lft, handdirlist_lft, "lft",
                                                                   ctcallback_lft, objpos_final=use_objpos_final_lft)
                                ropepulling_lft, usingposelist_lft, usingposeid_lft = ropepullingmotion(
                                    IKpossiblelist_lft, togoalvec_lft, ctcallback_lft, theta=theta,
                                    theta_next=theta_next)
                                if ropepulling_lft is False:
                                    continue
                                print("ropepulling_lft", ropepulling_lft)
                                if ctcallback_lft.iscollided(currentlftjnt, [obj_current]) is True:
                                    moveup = ctcallback_lft.getLinearPrimitivenothold(currentlftjnt, [0, 0, 1], 50, [],
                                                                                      type="source")
                                    print("moveup", moveup)
                                    lftmidpath = RRTmotion(moveup[-1], usingposelist_lft[2][0], ctcallback_lft,
                                                           obscmlist, 30, 10)
                                    lftmidpath = list(moveup) + list(lftmidpath)
                                else:
                                    lftmidpath = RRTmotion(currentlftjnt, usingposelist_lft[2][0], ctcallback_lft,
                                                           obscmlist, 30, 10)

                                if len(ropepulling_lft) > 0 and lftmidpath is not False:
                                    print("lftmidpath", lftmidpath)
                                    endflag = 1
                                    predictlist[0] = d_next
                                    predictlist[2] = theta_before
                                    break

                            if onearmflag[0] == 1:
                                break

                            if endflag == 1:
                                keeplist = [usingposelist_lft[2][1], usingposelist_lft[3][1], tostartvec]
                                pickle.dump(keeplist, open("keeplist.pickle", "wb"))

                                experimentlist.append(
                                    [list(lftmidpath) + list(usingposelist_lft[3][0]), 85, 0, "lft", "lftmidpath", 2.0])
                                experimentlist.append([midpathgoal_rgt, 85, 85, "rgt", "moveto_goalrgt", 1.0])
                                experimentlist.append([ropepulling_lft, 0, 0, "lft", "ropepulling_lft", 1.0])
                                break

                            startpointid += 1

                        rbt.movearmfk(currentrgtjnt, "rgt")
                        rbt.movearmfk(currentlftjnt, "lft")
                        obscmlist.pop(-1)

                elif activelist_real[4] == "ropepulling_rgt" and onearmflag[0] == 1:
                    currentrgtjnt = rbt.getarmjnts("rgt")
                    currentlftjnt = rbt.getarmjnts("lft")
                    currentrgtpos = rbt.getee("rgt")[0]
                    currentlftpos = rbt.getee("lft")[0]

                    pullingcounter[0] += 1
                    pullingcounter_rgt[0] += 1
                    print("右手で" + str(pullingcounter_rgt[0]) + "回引きました")

                    ## 物体の点群を検出(実際の点群)　TODO シミュレーションの際はコメントアウト---
                    newpcd = getpointcloudkinect(objpointrange)

                    ## ransac and icp(実機) --------------------------------------
                    refpoint_fitting = np.array([[-w / 2, -l / 2, 0, 1],
                                                 [-w / 2, l / 2, 0, 1],
                                                 [w / 2, l / 2, 0, 1],
                                                 [w / 2, -l / 2, 0, 1]])
                    targetpointnew, refpoint_fitting = objectfitting(newpcd, board, refpoint_fitting)
                    pickle.dump(targetpointnew, open("targetpointnew.pickle", "wb"))
                    xaxis_obj, yaxis_obj, zaxis_obj, theta, cog = getobjaxis(targetpointnew, refpoint_fitting)
                    print("theta = ", theta)
                    if theta > thetathreshold:
                        ropepullendflag[0] = 1
                    ## thetaの結果を代入
                    predictlist[1] = theta
                    # objpos_initial_board, objrot_initial_board = getobjposandrot(cog, xaxis_obj, yaxis_obj, zaxis_obj)
                    objpos_initial_board, objrot_initial_board = getobjposandrot(zaxis_obj)
                    objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    obj_current = copy.deepcopy(obj)
                    obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    # obj_current.setColor(1, 0, 0, .5)
                    # objmnp[0] = obj_current
                    pickle.dump(obj_current, open("obj.pickle", "wb"))
                    ## -------------------------------

                    ## 物体の情報を更新(シミュレーション用)
                    # theta_sim[0] += 5
                    # print("theta = ", theta_sim[0])
                    # predictlist[1] = theta_sim[0]
                    # print("predictlist = ", predictlist)
                    # objpos_initial_board = copy.copy(objpos_start)
                    # objrot_initial_board = rm.rodrigues(rotate_axis, theta_sim[0])
                    # objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    # obj_current = copy.deepcopy(obj)
                    # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    # pickle.dump(obj_current, open("obj.pickle", "wb"))
                    ## ---------------------------------

                    if ropepullendflag[0] == 1:
                        print("十分引き上げました プッシングします")
                        np.save('finalposlist.npy', [])
                        ropepullingflag[0] = 0
                        pushingflag[0] = 1
                    else:
                        ## ---シミュレーション用
                        # ropelinesorted = []
                        # dir = rm.unit_vector(ropetoppos - currentrgtpos)
                        # i = 1
                        # while True:
                        #     lft_append_pos = currentrgtpos + dir * i
                        #     ropelinesorted.append(lft_append_pos)
                        #     if lft_append_pos[2] > 1750:
                        #         break
                        #     i += 1
                        # ropelinesorted = ropelinesorted[::-1]
                        # np.save('RopeVertex_test.npy', ropelinesorted)
                        ## -----------------------

                        ## ---実験用(点群を取得)-----　TODO シミュレーションの際はコメントアウト---
                        # pointrange = [50, currentrgtpos[0] + 100, currentrgtpos[1]-50, currentlftpos[1]-50, currentrgtpos[2] + 50, 1750]
                        # newpcd = getpointcloudkinectforrope_up(robot_s, "rgt", ropetoppos, pointrange)
                        # ropeline_up = doRANSAC(newpcd, 5)
                        # ropelinesorted = descendingorder(ropeline_up, 2)
                        ropelinesorted = []
                        direction = rm.unit_vector(ropetoppos - currentrgtpos)
                        i = 200
                        while True:
                            appendpos = currentrgtpos + i * direction
                            ropelinesorted.append(appendpos)
                            i += 1
                            if appendpos[2] > 1750:
                                break
                        ropelinesorted = ropelinesorted[::-1]
                        np.save('RopeVertex_data.npy', ropelinesorted)
                        ## ------------------------

                        ## 引き上げた後の物体を障害物として追加
                        # obj_current = copy.deepcopy(obj)
                        # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                        obscmlist.append(obj_current)
                        ## -------------------------

                        pregraspforlft = np.array(
                            [-10.728168111426358, -193.9447141062168, -45.034213897162765, 149.37995277108735,
                             -126.43569808570105, 76.11025958193447])
                        startpointid = 0
                        endflag = 0
                        while True:
                            print("startpointid", startpointid)
                            # robot_s.goinitpose()

                            IKpossiblelist_start_lft, objpos_initial_lft, objrot_initial_lft, startpointid = decidestartpose(
                                ropelinesorted, "lft", predefined_grasps_lft,
                                ctcallback_lft, pregraspforlft, startpointid)
                            print("decidestartpose complete")

                            for i in IKpossiblelist_start_lft:
                                direction = rm.unit_vector(handdirlist_lft[i[2]]) * (-1)
                                distance = 50
                                flag = 0
                                while True:
                                    preregrasppath = ctcallback_lft.getLinearPrimitivenothold(i[0], direction, distance,
                                                                                              obscmlist, type="source")
                                    if len(preregrasppath) > 0:
                                        flag = 1
                                        break
                                    else:
                                        distance -= 1
                                        if distance < 20:
                                            break

                                if flag == 1:
                                    preregrasppath = preregrasppath[::-1]
                                    possible = 1
                                    if ctcallback_lft.iscollided(currentlftjnt, [obj_current]):
                                        length = 30
                                        while True:
                                            path = ctcallback_lft.getLinearPrimitivenothold(currentlftjnt, [0, 0, 1],
                                                                                            length, [], type="source")
                                            if len(path) > 0:
                                                break
                                            length -= 1
                                            if length < 0:
                                                possible = 0
                                                break
                                        if possible == 1:
                                            lftmidpath = list(path) + RRTmotion(path[-1], ctcallback_lft,
                                                                                preregrasppath[0], obscmlist, 5, 10)
                                        else:
                                            lftmidpath = []
                                    else:
                                        lftmidpath = RRTmotion(currentlftjnt, preregrasppath[0], ctcallback_lft,
                                                               obscmlist, 5, 10)
                                    if lftmidpath is not False:
                                        endflag = 1
                                        break

                            if endflag == 1:
                                break
                            startpointid += 1
                            if startpointid == len(ropelinesorted):
                                break

                        experimentlist.append([lftmidpath, 85, 85, "lft", "lftmidpath", 1.0])
                        experimentlist.append([preregrasppath, 85, 0, "lft", "onearmgrasp_lft", 1.0])

                        rbt.movearmfk(currentrgtjnt, "rgt")
                        rbt.movearmfk(currentlftjnt, "lft")
                        obscmlist.pop(-1)

                elif activelist_real[4] == "onearmgrasp_lft":
                    currentrgtjnt = rbt.getarmjnts("rgt")
                    currentlftjnt = rbt.getarmjnts("lft")
                    currentlftpos = rbt.getee("lft")[0]

                    obj_current = pickle.load(open("obj.pickle", "rb"))

                    ## 引き上げた後の物体を障害物として追加
                    # obj_current = copy.deepcopy(obj)
                    # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    obscmlist.append(obj_current)
                    pickle.dump(obscmlist, open("obscmlist.pickle", "wb"))

                    ## ロープの下の部分を検出
                    ## --シミュレーション用
                    # ropelinesorted = []
                    # for i in range(100):
                    #     ropelinesorted.append([currentlftpos[0], currentlftpos[1], currentlftpos[2] - 100 -i])
                    # ropelinesorted = np.asarray(ropelinesorted)
                    # np.save('RopeVertex_test.npy', ropelinesorted)
                    ## --------------------------------------------------

                    ## 下側　Todo シミュレーションの際はコメントアウト --
                    time.sleep(3)
                    uc.opengripper('rgt')
                    time.sleep(5)
                    eeposlft = rbt.getee("lft")[0]
                    # if 1600 < eeposlft[2] < 1750:
                    #     pointrange = [50, eeposlft[0] + 100, eeposlft[1] - 100, eeposlft[1] + 100, 1550, 1650]
                    # else:
                    #     pointrange = [50, eeposlft[0] + 100, eeposlft[1] - 100, eeposlft[1] + 100, 1600, 1700]
                    pointrange = [eeposlft[0] - 50, eeposlft[0] + 50, eeposlft[1] - 70, eeposlft[1] + 30, 1600,
                                  eeposlft[2] - 50]
                    newpcd = getpointcloudkinectforrope_down(rbt, "lft", pointrange)
                    ropeline_under = doRANSAC(newpcd, 5)
                    ropelinesorted = descendingorder(ropeline_under, 2)
                    np.save('RopeVertex_data.npy', ropelinesorted)
                    ## --------------------------------------------------

                    startpointid_second = 0
                    endflag_second = 0
                    while True:
                        # robot_s.goinitpose()
                        IKpossiblelist_start_rgt, objpos_initial_rgt, objrot_initial_rgt, startpointid_second = decidestartpose(
                            ropelinesorted, "rgt", predefined_grasps_rgt,
                            ctcallback_rgt, currentrgtjnt, startpointid_second)
                        print("startpointid_second = ", startpointid_second)

                        # obstacles = [obj_current]
                        objpos_finallist_rgt = create_candidate_points(arm_name="rgt",
                                                                       initialhandpos=objpos_initial_rgt,
                                                                       limitation=currentlftpos[1])
                        np.save('finalposlist.npy', objpos_finallist_rgt)

                        # elements_for_predictlist = predictlistforpullinglength(objpos_initial_rgt, objpos_finallist_rgt)

                        assessment_value_list_rgt, totalIKpossiblelist_startgoal_rgt, costlist_rgt, elements_for_predictlist_rgt, togoalveclist_rgt = getsuitablegoalpos_second(
                            IKpossiblelist_start_rgt, objpos_initial_rgt,
                            objpos_finallist_rgt, "rgt", predefined_grasps_rgt,
                            ctcallback_rgt, predictlist)

                        # print("total", totalIKpossiblelist_startgoal_rgt)
                        for i, each_assessment_value_list in enumerate(assessment_value_list_rgt):
                            useid = each_assessment_value_list[1]
                            use_element = elements_for_predictlist_rgt[i]
                            togoalvec_rgt = togoalveclist_rgt[i]
                            print("useid", useid)
                            IKpossiblelist_startgoal_rgt = totalIKpossiblelist_startgoal_rgt[int(useid)]
                            if IKpossiblelist_startgoal_rgt is False:
                                continue

                            ## 予測値の計算
                            d_before, theta_before, theta_beforebefore = predictlist
                            ## 予測値
                            d_next, theta_next, use_objpos_final_rgt, flag = use_element
                            print("d_next = ", d_next)
                            print("theta_next = ", theta_next)
                            print("objpos final = ", use_objpos_final_rgt)
                            ropepullendflag[0] = flag

                            IKpossiblelist_rgt = decidemidpose(IKpossiblelist_startgoal_rgt, handdirlist_rgt, "rgt",
                                                               ctcallback_rgt, objpos_final=use_objpos_final_rgt)
                            ropepulling_rgt, usingposelist_rgt, usingposeid = ropepullingmotion(IKpossiblelist_rgt,
                                                                                                togoalvec_rgt,
                                                                                                ctcallback_rgt)
                            print("ropepulling_rgt", ropepulling_rgt)
                            if ropepulling_rgt is not False:
                                rgtmidpath = RRTmotion(currentrgtjnt, usingposelist_rgt[2][0], ctcallback_rgt,
                                                       obscmlist, 30, 10)
                            else:
                                continue

                            if len(ropepulling_rgt) > 0 and rgtmidpath is not False:
                                print("rgtmidpath", rgtmidpath)
                                endflag_second = 1
                                predictlist[0] = d_next
                                predictlist[2] = theta_before
                                break

                        if endflag_second == 1:
                            experimentlist.append(
                                [rgtmidpath + usingposelist_rgt[3][0], 85, 0, "rgt", "rgtmidpath", 2.0])

                            obj_next_predict = copy.deepcopy(obj)
                            pos = copy.copy(objpos_start)
                            rot = rm.rodrigues(rotate_axis, theta_next)
                            mat = rm.homobuild(pos, rot)
                            obj_next_predict.setMat(base.pg.np4ToMat4(mat))

                            pregraspforlft = np.array(
                                [-10.728168111426358, -193.9447141062168, -45.034213897162765, 149.37995277108735,
                                 -126.43569808570105, 76.11025958193447])
                            ## 引き上げた後、左手と衝突する場合
                            if ctcallback_lft.iscollided(currentlftjnt, [obj_next_predict]) is True:
                                movetopregrasplft = RRTmotion(currentlftjnt, pregraspforlft, ctcallback_lft, obscmlist,
                                                              30, 10)
                                experimentlist.append([movetopregrasplft, 85, 85, "lft", "movetopregrasplft", 2.0])

                            # experimentlist.append([[], 85, 85, "lft", "lftopengripper", 2.0])
                            lftgoaway = activelist_real[0][::-1]
                            experimentlist.append([lftgoaway, 85, 85, "lft", "lftgoaway", 1.0])
                            experimentlist.append([ropepulling_rgt, 0, 0, "rgt", "ropepulling_rgt", 2.0])
                            break

                        startpointid_second += 1

                    rbt.movearmfk(currentrgtjnt, "rgt")
                    rbt.movearmfk(currentlftjnt, "lft")
                    obscmlist.pop(-1)

                if pushingflag[0] == 1:
                    currentrgtjnt = rbt.getarmjnts("rgt")
                    currentlftjnt = rbt.getarmjnts("lft")

                    ## 実機 todo シミュレーションの際はコメントアウト
                    ## 物体のモデルを配置
                    objpos_initial, objrot_initial = getobjposandrot(zaxis_obj)

                    ## シミュレーション
                    # theta = thetathreshold
                    # hold_pos_init = copy.copy(objpos_start)
                    # hold_rot_init = rm.rodrigues(rotate_axis, theta)

                    obj_current = copy.deepcopy(obj)
                    mat = rm.homobuild(objpos_initial, objrot_initial)
                    obj_current.setMat(base.pg.np4ToMat4(mat))

                    ## --------------------- pushingの動作計画(実機) todo シミュレーションの際はコメントアウト -------------------------
                    pushpath_all, pushpos_all, obj_all, hangedpos_all = getpushingpath(theta, rotate_axis)

                    pickle.dump(obj_all[pushcounter[0]], open("obj.pickle", "wb"))
                    pickle.dump([pushpath_all, pushpos_all, obj_all, hangedpos_all], open("pushinglist.pickle", "wb"))
                    ## ---------------------------------------------------------------------------

                    ## --------------------- pushingの動作計画(シミュレーション) -------------------------
                    # limitdegree = 100
                    # pushpath_all, pushpos_all, obj_all, hangedpos_all = pickle.load(open("pushinglist_sim.pickle", "rb"))
                    ## ---------------------------------------------------------------------------------
                    print("------ pushing planning end_type -------")

                    pre_pushpose_pre = np.array(
                        [-33.35622966184176, -138.5040412924666, -74.8527980774441, 128.17060347355363,
                         -133.2811458983196, 104.94565852176])

                    ## 左腕を現在の位置から、初期位置へ移動
                    # returninitialpath = RRTmotion(currentlftjnt, pre_pushpose_pre, ctcallback_lft, obscmlist, 20, 30)
                    # ## 初期位置から準備位置へ移動
                    # prestartlftpath = RRTmotion(pre_pushpose_pre, pushpose_pre, ctcallback_lft, obscmlist, 20, 30)
                    # pickle.dump(prestartlftpath, open("prestartlftpath.pickle", "wb"))

                    ## 現在の物体を障害物として追加
                    obstacles = copy.deepcopy(obscmlist)
                    obstacles.append(obj_current)
                    hangedpos_current = gethangedpos(objpos_initial, objrot_initial)
                    hangetopulley = rm.unit_vector(pulleypos - hangedpos_current)
                    i = 1
                    while True:
                        hangropepos = copy.copy(hangedpos_current) + i * hangetopulley
                        hangpointobj = copy.deepcopy(ropeobj)
                        mat = rm.homobuild(hangropepos, np.eye(3))
                        hangpointobj.setMat(base.pg.np4ToMat4(mat))
                        obstacles.append(hangpointobj)
                        i += 1
                        if i > 300:
                            break

                    joint = copy.copy(currentlftjnt)
                    flag = 0
                    if ctcallback_rgt.iscollided(joint, [obj_current]):
                        length = 30
                        while True:
                            path = ctcallback_lft.getLinearPrimitivenothold(currentlftjnt, [0, 0, 1], length, [],
                                                                            type="source")
                            if len(path) > 0:
                                joint = path[-1]
                                break
                            length -= 1
                            if length < 0:
                                flag = 1
                                break

                    if flag == 1:
                        prestartlftpath = list(
                            RRTmotion(currentlftjnt, pre_pushpose_pre, ctcallback_lft, obstacles, 5, 20)) + list(
                            RRTmotion(pre_pushpose_pre, pushpose_pre, ctcallback_lft, obstacles, 5, 20))
                    else:
                        prestartlftpath = RRTmotion(joint, pushpose_pre, ctcallback_lft, obstacles, 5, 30)

                    ## 準備位置からプッシング位置へ移動
                    pushpos = pushpos_all[0]
                    pushjnt = rbt.numikmsc(pushpos, pushrot, pushpose_pre, "lft")
                    pickle.dump(pushjnt, open("pushjnt.pickle", "wb"))
                    rbt.movearmfk(pushjnt, "lft")
                    print("pushjnt", pushjnt)
                    startlftpath = RRTmotion(pushpose_pre, pushjnt, ctcallback_lft, obscmlist, 5, 10)
                    print("startlftpath = ", startlftpath)

                    ## 直接左手を初期位置に移動
                    if activelist_real[4] == "ropepulling_rgt":
                        # experimentlist.append([returninitialpath, 85, 0, "lft", "gotoinitiallft", 1.0])
                        pushstartpoint[0] = pathcounter[0] + 1

                    ## 右手で再把持
                    elif activelist_real[4] == "ropepulling_lft":
                        midgoaljnt_rgt, midpathgoal_rgt, tostartvec = pickle.load(open("keeplist.pickle", "rb"))
                        rbt.movearmfk(currentlftjnt, "lft")
                        eeposlft = rbt.getee("lft")[0]
                        pointrange = [eeposlft[0] - 100, eeposlft[0] + 100,
                                      eeposlft[1] - 100, eeposlft[1] + 100,
                                      eeposlft[2] + 50, eeposlft[2] + 150]
                        newpcd = getpointcloudkinect(pointrange)
                        ropeline = doRANSAC(newpcd, 5)
                        ropelinesorted = descendingorder(ropeline, 2)
                        np.save('RopeVertex_data.npy', ropelinesorted)

                        startpointid = 0
                        endflag = 0
                        while True:
                            rbt.goinitpose()
                            print("regrasppath search : ", str(startpointid + 1) + "回目")
                            IKpossiblelist_start, objpos_initial, objrot_initial, startpointid = decidestartpose(
                                ropelinesorted, "rgt",
                                predefined_grasps_rgt, ctcallback_rgt,
                                currentrgtjnt, startpointid)
                            print("IKpossiblelist_start test", IKpossiblelist_start)
                            print("len test", len(IKpossiblelist_start))

                            for i in IKpossiblelist_start:
                                direction = rm.unit_vector(handdirlist_rgt[i[2]]) * (-1)
                                distance = 80
                                flag = 0
                                while True:
                                    preregrasppath = ctcallback_rgt.getLinearPrimitivenothold(i[0], direction, distance,
                                                                                              obscmlist, type="source")
                                    if len(preregrasppath) > 0:
                                        flag = 1
                                        break
                                    else:
                                        distance -= 1
                                        if distance < 20:
                                            break

                                if flag == 1:
                                    preregrasppath = preregrasppath[::-1]
                                    regrasppath = RRTmotion(currentrgtjnt, preregrasppath[0], ctcallback_rgt, obscmlist,
                                                            5, 20)
                                    if regrasppath is not False:
                                        endflag = 1
                                        break

                            if endflag == 1:
                                break
                            startpointid += 1
                            if startpointid == len(ropelinesorted):
                                break

                        experimentlist.append([regrasppath + preregrasppath, 85, 0, "rgt", "regrasprgt", 1.0])
                        # experimentlist.append([returninitialpath, 85, 0, "lft", "gotoinitiallft", 1.0])
                        pushstartpoint[0] = pathcounter[0] + 2  ## プッシング手前位置にくるときのカウンタ

                    # experimentlist.append([prestartlftpath + startlftpath, 0, 0, "lft", "gotopushposlft", 5.0])
                    experimentlist.append([list(prestartlftpath), 0, 0, "lft", "gotopushposlft", 2.0])
                    experimentlist.append([list(startlftpath), 0, 0, "lft", "gotopushposlft", 2.0])

                    rbt.movearmfk(currentrgtjnt, "rgt")
                    rbt.movearmfk(currentlftjnt, "lft")

            ### -----------------------------

            ## プッシング
            elif pushingflag[0] == 1:
                currentrgtjnt = rbt.getarmjnts("rgt")
                currentlftjnt = rbt.getarmjnts("lft")
                currentrgtpos = rbt.getee("rgt")[0]
                ## 2020,0425
                ## 押し動作の軌道などを取り出す
                pushpath_all, pushpos_all, obj_all, hangedpos_all = pickle.load(open("pushinglist.pickle", "rb"))

                # if pathcounter[0] > pushstartpoint[0] and (activelist_real[3] == "lft" or activelist_real[4] == "move"):
                if (activelist_real[3] == "lft" and pathcounter[0] > pushstartpoint[0]) or activelist_real[4] == "move":
                    print("activelist[4]", activelist_real[4])
                    ## 1つ前のプッシングによる左手の高さの変化を求める
                    # use_circlepath = pickle.load(open("use_circlepath.pickle", "rb"))

                    if pushcounter[0] + 1 < len(pushpath_all):
                        # print("押した回数：", pushcounter[0])
                        # pushcounter[0] += 1

                        ## todo looselength_all と統合
                        # diff_ropelen = 100
                        print("pushcounter = ", pushcounter[0])
                        hangedpos = hangedpos_all[pushcounter[0]]
                        len1 = np.linalg.norm(pulleypos - hangedpos)
                        print("len1 = ", len1)
                        hangedpos_after = hangedpos_all[pushcounter[0] + 1]
                        len2 = np.linalg.norm(pulleypos - hangedpos_after)
                        print("len2 = ", len2)

                        diff_ropelen = abs(len1 - len2) * 4
                        # diff_ropelen = 20
                        print("diff_ropelen = ", diff_ropelen)

                        if rgtregraspflag[0] == 0:
                            rgtgoalpos = np.array([238.55902527, 15.77300387, 1799.74181438])
                            direction = rm.unit_vector(rgtgoalpos - currentrgtpos)
                            while True:
                                if currentrgtpos[2] + diff_ropelen >= 1750:
                                    diff_ropelen = 1750 - currentrgtpos[2]
                                rgtpath = ctcallback_rgt.getLinearPrimitivenothold(currentrgtjnt, direction,
                                                                                   diff_ropelen, obscmlist,
                                                                                   type="source")
                                if len(rgtpath) > 0:
                                    break
                                else:
                                    diff_ropelen -= 1
                                    if diff_ropelen == 0:
                                        rgtpath = []
                                        break

                            experimentlist.append([rgtpath, 0, 0, "rgt", "rgtrope", 2.0])
                            rbt.movearmfk(currentrgtjnt, "rgt")
                            rbt.movearmfk(currentlftjnt, "lft")

                        ## 0301
                        ## 右手でロープを再び掴む
                        else:
                            eepos_rgt, eerot_rgt = rbt.getee("rgt")
                            currentrgtjnt = rbt.getarmjnts("rgt")
                            currentlftjnt = rbt.getarmjnts("lft")

                            length = 80
                            direction = eerot_rgt[:, 2] * (-1)
                            while True:
                                movetomidpath2 = ctcallback_rgt.getLinearPrimitivenothold(currentrgtjnt, direction,
                                                                                          length, obscmlist,
                                                                                          type="source")
                                if len(movetomidpath2) > 0:
                                    break
                                length -= 1
                                if length < 0:
                                    print("path2が計算できません")
                                    return False

                            midjnt_rgt = movetomidpath2[-1]
                            # uc.movejntssgl(midjnt_rgt, "rgt")

                            time.sleep(5)
                            ## 実際の点群
                            # pointrange = [100, eepos_rgt[0] + 50, eepos_rgt[1] - 100, eepos_rgt[1] + 100, eepos_rgt[2] - 400, eepos_rgt[2] - 100]
                            # criteriapos = np.array([250, 0, 1150])
                            # pointrange = [criteriapos[0]-50, criteriapos[0]+50,
                            #               criteriapos[1]-100, criteriapos[1]+100,
                            #               criteriapos[2]+300, criteriapos[2]+500]
                            pointrange = [100, eepos_rgt[0] + 50, -100, 100, eepos_rgt[2] - 200, eepos_rgt[2] - 100]
                            newpcd = getpointcloudkinect(pointrange)
                            ropelinesorted = doRANSAC(newpcd, 5)
                            ropelinesorted = ascendingorder(ropelinesorted, axis=2)
                            np.save('RopeVertex_data.npy', ropelinesorted)

                            ## シミュレーション用
                            # ropelinesorted = np.load("RopeVertex_test.npy")[::-1]

                            startpointid = 0
                            endflag = 0
                            while True:
                                print("startpointid", startpointid)
                                rbt.goinitpose()
                                IKpossiblelist_start_rgt, objpos_initial_rgt, objrot_initial_rgt, startpointid = decidestartpose(
                                    ropelinesorted, "rgt",
                                    predefined_grasps_rgt, ctcallback_rgt,
                                    midjnt_rgt, startpointid)

                                for i in IKpossiblelist_start_rgt:
                                    graspjoint = i[0]
                                    rgtmovepath = RRTmotion(midjnt_rgt, graspjoint, ctcallback_rgt, obscmlist, 30, 3)
                                    if rgtmovepath is not False:
                                        endflag = 1
                                        break

                                if endflag == 1:
                                    break
                                startpointid += 1
                                if startpointid == len(ropelinesorted):
                                    break

                            experimentlist.append([movetomidpath2 + list(rgtmovepath), 85, 0, "rgt", "move", 2.0])

                            rgtregraspflag[0] = 0
                            rbt.movearmfk(currentrgtjnt, "rgt")
                            rbt.movearmfk(currentlftjnt, "lft")

                    ## 最後までpushしたとき
                    else:
                        goalpos = np.array([238.55902527, 15.77300387, 1750.74181438])
                        direction = rm.unit_vector(goalpos - currentrgtpos)
                        length = np.linalg.norm(goalpos - currentrgtpos)
                        while True:
                            rgtpath = ctcallback_rgt.getLinearPrimitivenothold(currentrgtjnt, direction, length,
                                                                               obscmlist, type="source")
                            if len(rgtpath) > 0:
                                break
                            length -= 1
                            if length < 0:
                                rgtpath = None
                                break
                        experimentlist.append([rgtpath, 0, 0, "rgt", "rgtrope", 2.0])
                        pushendflag[0] = 1
                        rbt.movearmfk(currentrgtjnt, "rgt")
                        rbt.movearmfk(currentlftjnt, "lft")

                ## プッシュ位置にきたときand右手でロープを上げ下げするとき(pushstartpoint[0]より後)
                # elif pathcounter[0] == pushstartpoint[0] or (pathcounter[0] > pushstartpoint[0] and activelist_real[3] == "rgt" and activelist_real[4] != "move"):
                elif (activelist_real[3] == "rgt" and pathcounter[0] > pushstartpoint[0]) and activelist_real[
                    4] != "move":
                    print("activelist[4]", activelist_real[4])

                    if pushendflag[0] == 0:
                        # pickle.dump(obj_all[pushcounter[0]], open("obj.pickle", "wb"))
                        # objmnp[0] = obj_all[pushcounter[0]]
                        nextpushpath = pushpath_all[pushcounter[0]]
                        used_pushpathlist.append(nextpushpath)
                        experimentlist.append([nextpushpath, 0, 0, "lft", "lftpush", 5.0])

                        pushcounter[0] += 1
                        pickle.dump(obj_all[pushcounter[0]], open("obj.pickle", "wb"))
                    ## pushendflag[0] = 1のとき
                    else:
                        pushingflag[0] = 0
                        ropeloosningflag[0] = 1
                        print("左手でロープを掴みます")

                        currentrgtpos, currentrgtrot = rbt.getee("rgt")
                        currentrgtjnt = rbt.getarmjnts("rgt")
                        currentlftjnt = rbt.getarmjnts("lft")
                        objpos_final = copy.copy(currentrgtpos)
                        pickle.dump(objpos_final, open("hold_pos_final.pickle", "wb"))

                        # ## beniya boardの場合
                        # returntopushjnt = list(pickle.load(open("prestartlftpath.pickle", "rb")))
                        # inipath = list(RRTmotion(returntopushjnt[0], robot_s.initjnts[9:15], ctcallback_lft, obscmlist, 10, 30))
                        #
                        # for eachpath in used_pushpathlist:
                        #     returntopushjnt = returntopushjnt + eachpath
                        # returntopushjnt = returntopushjnt[::-1]
                        # returntopushjnt += inipath

                        lastpushpath = pushpath_all[pushcounter[0] - 1][::-1]

                        ## ---シミュレーション用---
                        # ropeline = []
                        # for i in range(100):
                        #     ropeline.append([250, 0, 1700 - i])
                        # ropelinesorted = ropeline[::-1]
                        # np.save("RopeVertex_test.npy", ropelinesorted)
                        ## ----------------------

                        ## ---実験用(点群を取得)　Todo シミュレーションの際はコメントアウト -------
                        eeposrgt = rbt.getee("rgt")[0]
                        pointrange = [eeposrgt[0] - 50, eeposrgt[0] + 50,
                                      eeposrgt[1] - 100, eeposrgt[1] + 100,
                                      eeposrgt[2] - 200, eeposrgt[2] - 50]
                        newpcd = getpointcloudkinect(pointrange)
                        ropeline = doRANSAC(newpcd, 5)
                        ropelinesorted = ascendingorder(ropeline, 2)
                        np.save('RopeVertex_data.npy', ropelinesorted)
                        ## ------------------------

                        # rgtdir = rm.unit_vector(currentrgtrot[:, 2]) * (-1)
                        rgtdir = np.array([0, -1, 0])
                        length = 80
                        while True:
                            rgtmovepath = ctcallback_rgt.getLinearPrimitivenothold(currentrgtjnt, rgtdir, length,
                                                                                   obscmlist, type="source")
                            if len(rgtmovepath) > 0:
                                break
                            else:
                                length -= 1
                                if length < 5:
                                    print("右手を移動させることができません")
                                    return False

                        startpointid = 0
                        while True:
                            print("startpointid", startpointid)
                            rbt.goinitpose()
                            IKpossiblelist_start_lft, objpos_initial_lft, objrot_initial_lft, startpointid = decidestartpose(
                                ropelinesorted, "lft", predefined_grasps_lft, ctcallback_lft,
                                currentlftjnt, startpointid)
                            IKpossiblelist_startgoal_lft, objpos_final, tostartvec, togoalvec = decidegoalpose(
                                IKpossiblelist_start_lft, objpos_initial_lft, "lft", predefined_grasps_lft,
                                ctcallback_lft, objpos_final=objpos_final, label="up")
                            # IKpossiblelist_startgoal_lft = decidegoalpose_onepoint(IKpossiblelist_start_lft, objpos_initial_lft, hold_pos_final, "lft", predefined_grasps_lft, ctcallback_lft)

                            IKpossiblelist_lft = decidemidpose(IKpossiblelist_startgoal_lft, handdirlist_lft, "lft",
                                                               ctcallback_lft)

                            togoalvec = copy.copy(objpos_final - objpos_initial_lft)
                            tostartvec = copy.copy(togoalvec)[::-1]
                            ropepulling_lft, usingposelist_lft, usingposeid_lft = ropepullingmotion(IKpossiblelist_lft,
                                                                                                    togoalvec,
                                                                                                    ctcallback_lft)
                            print("obscmlist", len(obscmlist))
                            gotoinitiallftpoint = RRTmotion(lastpushpath[-1], usingposelist_lft[2][0], ctcallback_lft,
                                                            obscmlist, 30, 3)

                            if len(ropepulling_lft) > 0 and gotoinitiallftpoint is not False:
                                print("緩める：ropepulling_lft", ropepulling_lft)
                                print("gotoinitiallftpoint", gotoinitiallftpoint)
                                break

                            startpointid += 1

                        keeplist = [usingposelist_lft[2][1], usingposelist_lft[3][1], tostartvec]
                        pickle.dump(keeplist, open("keeplist.pickle", "wb"))

                        experimentlist.append([lastpushpath, 0, 0, "lft", "lastpushpath", 2.0])
                        experimentlist.append(
                            [gotoinitiallftpoint + usingposelist_lft[3][0], 60, 0, "lft", "gotoinitiallftpoint", 2.0])
                        experimentlist.append([rgtmovepath, 60, 60, "rgt", "rgtmove", 1.0])
                        experimentlist.append([ropepulling_lft, 0, 0, "lft", "ropepulling_lft", 1.0])

                        rbt.movearmfk(currentrgtjnt, "rgt")
                        rbt.movearmfk(currentlftjnt, "lft")

                        # experimentlist.append([returntopushjnt, 0, 0, "lft", "goinitpose", 2.0])
                        # robot_s.movearmfk(currentrgtjnt, "rgt")
                        # robot_s.movearmfk(currentlftjnt, "lft")

            ## ロープを緩める
            elif ropeloosningflag[0] == 1:

                if activelist_real[4] == "ropepulling_lft":
                    currentrgtjnt = rbt.getarmjnts("rgt")
                    currentlftjnt = rbt.getarmjnts("lft")
                    objpos_final = pickle.load(open("hold_pos_final.pickle", "rb"))
                    midgoaljnt_lft, midpathgoal_lft, tostartvec = pickle.load(open("keeplist.pickle", "rb"))

                    ## 物体の点群を検出(実機)　todo シミュレーションの際はコメントアウト
                    newpcd = getpointcloudkinect(objpointrange)
                    refpoint_fitting = np.array([[-w / 2, -l / 2, 0, 1],
                                                 [-w / 2, l / 2, 0, 1],
                                                 [w / 2, l / 2, 0, 1],
                                                 [w / 2, -l / 2, 0, 1]])
                    targetpointnew, refpoint_fitting = objectfitting(newpcd, board, refpoint_fitting)
                    pickle.dump(targetpointnew, open("targetpointnew.pickle", "wb"))
                    xaxis_obj, yaxis_obj, zaxis_obj, theta, cog = getobjaxis(targetpointnew, refpoint_fitting)
                    objpos_initial_board, objrot_initial_board = getobjposandrot_after(cog, xaxis_obj, yaxis_obj,
                                                                                       zaxis_obj)
                    pickle.dump([objpos_initial_board, objrot_initial_board, xaxis_obj],
                                open("objinitialandxaxis_obj.pickle", "wb"))
                    objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    obj_current = copy.deepcopy(obj)
                    obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    pickle.dump(obj_current, open("obj.pickle", "wb"))

                    obscmlist.append(obj_current)
                    print("プッシング後の物体の傾き：", theta)

                    ## 物体の点群を検出(シミュレーション)
                    # theta_sim[0] -= 10
                    # theta = theta_sim[0]
                    # print("theta = ", theta_sim[0])
                    # objpos_initial_board = copy.copy(objpos_start)
                    # objrot_initial_board = rm.rodrigues(rotate_axis, theta)
                    # refpoint = getrefpoint(objpos_initial_board, objrot_initial_board)
                    # objrot_ver = rm.rodrigues(rotate_axis, 90)
                    # criteriapos = getrefpoint(objpos_initial_board, objrot_ver)
                    # objpos_initial_board += criteriapos - refpoint
                    #
                    # objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    # obj_current = copy.deepcopy(obj)
                    # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    # pickle.dump(obj_current, open("obj.pickle", "wb"))
                    ## -------------------------------------

                    if theta < endthreshold:
                        print("物体を十分降ろしました")
                        geterror(refpoint_fitting)
                        endpoint[0] = pathcounter[0] + 1
                        experimentlist.append([[], 85, 85, "lft", "end_opengripper", 1.0])
                    else:
                        ## ---シミュレーション用---
                        # ropeline = []
                        # for i in range(100):
                        #     ropeline.append([250, 0, 1700 - i])
                        # ropelinesorted = ropeline[::-1]
                        # np.save("RopeVertex_test.npy", ropelinesorted)
                        ## ----------------------

                        ## ---実験用(点群を取得)　todo シミュレーションの際はコメントアウト　---
                        eeposlft = rbt.getee("lft")[0]
                        pointrange = [eeposlft[0] - 100, eeposlft[0] + 50, eeposlft[1] - 100, eeposlft[1] + 100,
                                      eeposlft[2] - 150, eeposlft[2] - 50]
                        newpcd = getpointcloudkinect(pointrange)
                        ropeline = doRANSAC(newpcd, 5)
                        ropelinesorted = ascendingorder(ropeline, 2)
                        # print("rope after lftpulling = ", ropelinesorted)
                        np.save('RopeVertex_data.npy', ropelinesorted)
                        ## ---------------------

                        startpointid = 0
                        while True:
                            rbt.goinitpose()
                            IKpossiblelist_start_rgt, objpos_initial_rgt, objrot_initial_rgt, startpointid = decidestartpose(
                                ropelinesorted,
                                "rgt", predefined_grasps_rgt,
                                ctcallback_rgt, currentrgtjnt, startpointid)
                            IKpossiblelist_startgoal_rgt, objpos_final, tostartvec, togoalvec = decidegoalpose(
                                IKpossiblelist_start_rgt,
                                objpos_initial_rgt, "rgt",
                                predefined_grasps_rgt,
                                ctcallback_rgt, objpos_final=objpos_final, label="up")
                            IKpossiblelist_rgt = decidemidpose(IKpossiblelist_startgoal_rgt, handdirlist_rgt, "rgt",
                                                               ctcallback_rgt)
                            ropepulling_rgt, usingposelist_rgt, usingposeid = ropepullingmotion(IKpossiblelist_rgt,
                                                                                                togoalvec,
                                                                                                ctcallback_rgt)

                            rgtmidpath = RRTmotion(currentrgtjnt, usingposelist_rgt[2][0], ctcallback_rgt, obscmlist,
                                                   30, 3)

                            if len(ropepulling_rgt) > 0 and rgtmidpath is not False:
                                print("緩める：ropepulling_rgt", ropepulling_rgt)
                                print("rgtmidpath", rgtmidpath)
                                break

                            startpointid += 1

                        keeplist = [usingposelist_rgt[2][1], usingposelist_rgt[3][1], tostartvec]
                        pickle.dump(keeplist, open("keeplist.pickle", "wb"))

                        experimentlist.append([rgtmidpath + usingposelist_rgt[3][0], 85, 0, "rgt", "rgtmidpath", 1.0])
                        experimentlist.append([midpathgoal_lft, 85, 85, "lft", "moveto_midgoal", 1.0])
                        experimentlist.append([ropepulling_rgt, 0, 0, "rgt", "ropepulling_rgt", 1, 0])

                        obscmlist.pop(-1)
                        rbt.movearmfk(currentrgtjnt, "rgt")
                        rbt.movearmfk(currentlftjnt, "lft")

                elif activelist_real[4] == "ropepulling_rgt":
                    currentrgtjnt = rbt.getarmjnts("rgt")
                    currentlftjnt = rbt.getarmjnts("lft")
                    objpos_final = pickle.load(open("hold_pos_final.pickle", "rb"))
                    midgoaljnt_rgt, midpathgoal_rgt, tostartvec = pickle.load(open("keeplist.pickle", "rb"))

                    ## 物体の点群を検出(実機) todo シミュレーションの際はコメントアウト
                    newpcd = getpointcloudkinect(objpointrange)
                    refpoint_fitting = np.array([[-w / 2, -l / 2, 0, 1],
                                                 [-w / 2, l / 2, 0, 1],
                                                 [w / 2, l / 2, 0, 1],
                                                 [w / 2, -l / 2, 0, 1]])
                    targetpointnew, refpoint_fitting = objectfitting(newpcd, board, refpoint_fitting)
                    pickle.dump(targetpointnew, open("targetpointnew.pickle", "wb"))
                    xaxis_obj, yaxis_obj, zaxis_obj, theta, cog = getobjaxis(targetpointnew, refpoint_fitting)
                    objpos_initial, objrot_initial = getobjposandrot_after(cog, xaxis_obj, yaxis_obj, zaxis_obj)
                    pickle.dump([objpos_initial, objrot_initial, xaxis_obj],
                                open("objinitialandxaxis_obj.pickle", "wb"))
                    objmat4 = rm.homobuild(objpos_initial, objrot_initial)
                    obj_current = copy.deepcopy(obj)
                    obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    pickle.dump(obj_current, open("obj.pickle", "wb"))
                    print("プッシング後の物体の傾き：", theta)

                    ## 物体の点群を検出(シミュレーション)
                    # theta_sim[0] -= 10
                    # theta = theta_sim[0]
                    # print("theta = ", theta_sim[0])
                    # objpos_initial_board = copy.copy(objpos_start)
                    # objrot_initial_board = rm.rodrigues(rotate_axis, theta)
                    # refpoint = getrefpoint(objpos_initial_board, objrot_initial_board)
                    # objrot_ver = rm.rodrigues(rotate_axis, 90)
                    # criteriapos = getrefpoint(objpos_initial_board, objrot_ver)
                    # objpos_initial_board += criteriapos - refpoint
                    #
                    # objmat4 = rm.homobuild(objpos_initial_board, objrot_initial_board)
                    # obj_current = copy.deepcopy(obj)
                    # obj_current.setMat(base.pg.np4ToMat4(objmat4))
                    # pickle.dump(obj_current, open("obj.pickle", "wb"))
                    ## -------------------------------------

                    if theta < endthreshold:
                        print("物体を十分降ろしました")
                        geterror(refpoint_fitting)
                        endpoint[0] = pathcounter[0] + 1
                        experimentlist.append([[], 85, 85, "rgt", "end_opengripper", 1.0])
                    else:
                        ## ---シミュレーション用---
                        # ropeline = []
                        # for i in range(100):
                        #     ropeline.append([250, 0, 1700 - i])
                        # ropelinesorted = ropeline[::-1]
                        # np.save("RopeVertex_test.npy", ropelinesorted)
                        ## ----------------------

                        ## ----------------------
                        ## ---実験用(点群を取得) todo シミュレーションの際はコメントアウト-----
                        eeposrgt = rbt.getee("rgt")[0]
                        pointrange = [eeposrgt[0] - 100, eeposrgt[0] + 50, eeposrgt[1] - 100, eeposrgt[1] + 100,
                                      eeposrgt[2] - 200, eeposrgt[2] - 50]
                        newpcd = getpointcloudkinect(pointrange)
                        ropeline = doRANSAC(newpcd, 5)
                        ropelinesorted = ascendingorder(ropeline, 2)
                        np.save('RopeVertex_data.npy', ropelinesorted)
                        ## ------------------------

                        startpointid = 0
                        while True:
                            rbt.goinitpose()
                            IKpossiblelist_start_lft, objpos_initial_lft, objrot_initial_lft, startpointid = decidestartpose(
                                ropelinesorted, "lft",
                                predefined_grasps_lft, ctcallback_lft,
                                currentlftjnt, startpointid)
                            IKpossiblelist_startgoal_lft, objpos_final, tostartvec, togoalvec = decidegoalpose(
                                IKpossiblelist_start_lft,
                                objpos_initial_lft, "lft",
                                predefined_grasps_lft,
                                ctcallback_lft, objpos_final=objpos_final, label="up")
                            IKpossiblelist_lft = decidemidpose(IKpossiblelist_startgoal_lft, handdirlist_lft, "lft",
                                                               ctcallback_lft)
                            ropepulling_lft, usingposelist_lft, usingposeid_lft = ropepullingmotion(IKpossiblelist_lft,
                                                                                                    togoalvec,
                                                                                                    ctcallback_lft)
                            lftmidpath = RRTmotion(currentlftjnt, usingposelist_lft[2][0], ctcallback_lft, obscmlist,
                                                   30, 3)

                            if len(ropepulling_lft) > 0 and lftmidpath is not False:
                                print("緩める：ropepulling_lft", ropepulling_lft)
                                print("lftmidpath", lftmidpath)
                                break

                            startpointid += 1

                        keeplist = [usingposelist_lft[2][1], usingposelist_lft[3][1], tostartvec]
                        pickle.dump(keeplist, open("keeplist.pickle", "wb"))

                        experimentlist.append([lftmidpath + usingposelist_lft[3][0], 85, 0, "lft", "lftmidpath", 1.0])
                        experimentlist.append([midpathgoal_rgt, 85, 85, "rgt", "moveto_goalrgt", 1.0])
                        experimentlist.append([ropepulling_lft, 0, 0, "lft", "ropepulling_lft", 1.0])

                        rbt.movearmfk(currentrgtjnt, "rgt")
                        rbt.movearmfk(currentlftjnt, "lft")
            ###--------------------------

            pathcounter[0] += 1
            activelist_sim = experimentlist[pathcounter[0]]
            base.inputmgr.keyMap['space'] = False
            taskMgr.remove('updatemotionsec')
            motioncounter[0] = 0
            taskMgr.doMethodLater(0.1, updatemotionsec, "updatemotionsec",
                                  extraArgs=[activelist_sim, rbtmnp, objmnp,
                                             motioncounter, rbt, pnt, finalpos], appendTask=True)
        else:
            # pathcounter[0] = 0
            taskMgr.remove('updateshow')
            return task.again
        return task.again


    taskMgr.doMethodLater(0.04, updatesection, "updatesection",
                          extraArgs=[rbtmnpani, objmnpani, motioncounter,
                                     rbt, pntani, finalposani],
                          appendTask=True)

    base.run()
