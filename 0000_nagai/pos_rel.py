from wrs import wd, rm, mcm, mgm
from wrs.robot_sim.robots.khi.khi_main import KHI_DUAL
from wrs.robot_sim.robots.khi import meshes as mesh
import os

if __name__ == '__main__':
    class Data(object):
        def __init__(self, mot_data):
            self.counter = 0
            self.mot_data = mot_data

    base = wd.World(cam_pos=[.5,.5,.5], lookat_pos=[0, 0, 0])
    mgm.gen_frame(ax_length=.15).attach_to(base)

    khibt = KHI_DUAL()
    # khibt.gen_meshmodel().attach_to(base)

    # load objects
    # workbench
    workbench_file = os.path.join(mesh.__path__[0], "workbench.stl")
    workbench_cm = mcm.CollisionModel(initor=workbench_file,rgb=rm.const.orange_red)
    workbench_cm.pos = rm.np.array([0,0,0])
    workbench_cm.rotmat = rm.rotmat_from_euler(ai=rm.np.pi/2,aj=0,ak=0,order='rxyz')
    work = workbench_cm

    # bracketR1
    bracketR1_file = os.path.join(mesh.__path__[0], "bracketR1.stl")
    bracketR1_cm = mcm.CollisionModel(initor=bracketR1_file,rgb=rm.const.gray)
    bracketR1_cm.pos = rm.np.array([-0.0899,0.022,0])
    bracketR1_cm.rotmat = rm.rotmat_from_euler(ai=0,aj=0,ak=0,order='rxyz')
    bracketR1_cm.attach_to(work)

    # capacitor
    capacitor_file = os.path.join(mesh.__path__[0], "capacitor.stl")
    capacitor_cm = mcm.CollisionModel(initor=capacitor_file,rgb=rm.const.blue)
    capacitor_cm.pos = rm.np.array([0.02,0.046,0])
    capacitor_cm.rotmat = rm.rotmat_from_euler(ai=0,aj=-rm.np.pi/2,ak=0,order='rxyz')
    capacitor_cm.attach_to(work)

    # relay_205B
    relay_205B_file = os.path.join(mesh.__path__[0], "relay_205B.stl")
    relay_205B_cm = mcm.CollisionModel(initor=relay_205B_file,rgb=rm.const.black)
    relay_205B_cm.pos = rm.np.array([0.091,-0.003,0.0225])
    relay_205B_cm.rotmat = rm.rotmat_from_euler(ai=0,aj=rm.np.pi/2,ak=rm.np.pi/2,order='rxyz')
    relay_205B_cm.attach_to(work)

    # belt
    belt_file = os.path.join(mesh.__path__[0], "belt.stl")
    belt_cm = mcm.CollisionModel(initor=belt_file,rgb=rm.const.deep_sky_blue)
    belt_cm.pos = rm.np.array([-0.044,0.05,0.027])
    belt_cm.rotmat = rm.rotmat_from_euler(ai=rm.np.pi/2,aj=0,ak=0,order='rxyz')
    belt_cm.attach_to(work)

    # terminal_block
    terminal_block_file = os.path.join(mesh.__path__[0], "terminal_block.stl")
    terminal_block_cm = mcm.CollisionModel(initor=terminal_block_file,rgb=rm.const.yellow)
    terminal_block_cm.pos = rm.np.array([0.065,0.023,0])
    terminal_block_cm.rotmat = rm.rotmat_from_euler(ai=0,aj=rm.np.pi/2,ak=0,order='rxyz')
    terminal_block_cm.attach_to(work)
    
    #numpy print options
    rm.np.set_printoptions(precision=3,suppress=True)

    #print rotmats and pos
    print("workbench rotmat is:\t pos is:")
    print(workbench_cm.rotmat,workbench_cm.pos)
    print("bracketR1 rotmat is:\t pos is:")
    print(bracketR1_cm.rotmat,bracketR1_cm.pos)
    print("capacitor rotmat is:\t pos is:")
    print(capacitor_cm.rotmat,capacitor_cm.pos)
    print("relay____ rotmat is:\t pos is:")
    print(relay_205B_cm.rotmat,relay_205B_cm.pos)
    print("belt_____ rotmat is:\t pos is:")
    print(belt_cm.rotmat,belt_cm.pos)
    print("terminal_ rotmat is:\t pos is:")
    print(terminal_block_cm.rotmat,terminal_block_cm.pos)
    
    # workbench collision check
    print("workbench collision check")
    print("bracket collided?:", workbench_cm.is_mcdwith(bracketR1_cm))
    print("capacitor collided?:", workbench_cm.is_mcdwith(capacitor_cm))
    print("relay collided?:", workbench_cm.is_mcdwith(relay_205B_cm))
    print("belt collided?:", workbench_cm.is_mcdwith(belt_cm))
    print("terminal collided?:", workbench_cm.is_mcdwith(terminal_block_cm))

    # bracketR1 collision check
    print("\nbracketR1 collision check")
    print("workbench collided?:", bracketR1_cm.is_mcdwith(workbench_cm))
    mcd_result, cdpoint = bracketR1_cm.is_mcdwith(workbench_cm, toggle_contacts=True)
    for pnt in cdpoint:
        mgm.gen_sphere(pos=pnt,  rgb=rm.np.array([1, 0, 0]), alpha=1, radius =.0002).attach_to(base)
    
    print("capacitor collided?:", bracketR1_cm.is_mcdwith(capacitor_cm))
    print("relay collided?:", bracketR1_cm.is_mcdwith(relay_205B_cm))
    print("belt collided?:", bracketR1_cm.is_mcdwith(belt_cm))
    print("terminal collided?:", bracketR1_cm.is_mcdwith(terminal_block_cm))
    
    work.alpha = .3
    work.attach_to(base)
    base.run()