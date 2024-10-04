import wrs.robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
import wrs.modeling.dynamics.bullet.bdmodel as bdm
import wrs.visualization.panda.world as wd
from wrs import basis as rm, modeling as gm, modeling as cm
import numpy as np


def get_lnk_bdmodel(robot_s, component_name, lnk_id):
    lnk = robot_s.manipulator_dict[component_name].lnks[lnk_id]
    bd_lnk = bdm.BDModel(lnk["collisionmodel"], mass=0, type="box", dynamic=False)
    bd_lnk.set_homomat(rm.homomat_from_posrot(lnk["gl_pos"], lnk["gl_rotmat"]))
    return bd_lnk

def update_robot_bdmodel(robot_s, bd_lnk_list):
    cnter = 0
    for arm_name in ["lft_arm", "rgt_arm"]:
        for lnk_id in [1, 2, 3, 4, 5, 6]:
            lnk = robot_s.manipulator_dict[arm_name].lnks[lnk_id]
            bd_lnk_list[cnter].set_homomat(rm.homomat_from_posrot(lnk["gl_pos"], lnk["gl_rotmat"]))
            cnter+=1


def get_robot_bdmoel(robot_s):
    bd_lnk_list = []
    for arm_name in ["lft_arm", "rgt_arm"]:
        for lnk_id in [1, 2, 3, 4, 5, 6]:
            bd_lnk_list.append(get_lnk_bdmodel(robot_s, arm_name, lnk_id))
    return bd_lnk_list


if __name__ == '__main__':

    base = wd.World(cam_pos=[10, 0, 5], lookat_pos=[0, 0, 1])
    base.setFrameRateMeter(True)
    gm.gen_frame().attach_to(base)
    # obj_box = mcm.gen_box(xyz_lengths=[.2, 1, .3], rgba=[.3, 0, 0, 1])
    obj_box = cm.gen_sphere(radius=.5, rgba=[.3, 0, 0, 1])
    obj_bd_box = bdm.BDModel(obj_box, mass=.3, type="convex")
    obj_bd_box.set_pos(np.array([.7, 0, 2.7]))
    obj_bd_box.start_physics()
    base.attach_internal_update_obj(obj_bd_box)

    robot_s = ur3ed.UR3EDual()
    robot_s.fk("both_arm", np.radians(np.array([-90,-60,-60,180,0,0,90,-120,60,0,0,0])))
    robot_s.gen_stickmodel().attach_to(base)
    robot_s.show_cdprimit()
    bd_lnk_list = get_robot_bdmoel(robot_s)
    for bdl in bd_lnk_list:
        bdl.start_physics()
        base.attach_internal_update_obj(bdl)

    def update(robot_s, bd_lnk_list, task):
        if base.inputmgr.keymap['space'] is True:
            la_jnt_values = robot_s.get_jnt_values("lft_arm")
            ra_jnt_values = robot_s.get_jnt_values("rgt_arm")
            rand_la = np.random.rand(6)*.01
            rand_ra = np.random.rand(6)*.01
            la_jnt_values=la_jnt_values+rand_la
            ra_jnt_values=ra_jnt_values+rand_ra
            robot_s.fk(component_name="both_arm", joint_values=np.hstack((la_jnt_values, ra_jnt_values)))
            update_robot_bdmodel(robot_s, bd_lnk_list)
            base.inputmgr.keymap['space'] = False
        return task.cont
    taskMgr.add(update, "update", extraArgs=[robot_s, bd_lnk_list], appendTask=True)
    base.run()
