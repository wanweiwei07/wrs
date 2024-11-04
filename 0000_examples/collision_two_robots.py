from wrs import rm, wd, mgm, cbt

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

cbt1 = cbt.Cobotta(pos=rm.vec(0, -.15, 0), rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.radians(90)), name="cbt1")
cbt2 = cbt.Cobotta(pos=rm.vec(0, .15, 0), rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.radians(-90)), name="cbt2")

cbt1.gen_meshmodel().attach_to(base)
cbt2.gen_meshmodel().attach_to(base)

for i in range(1000):
    print(i, " 1000")
    cbt1.goto_given_conf(cbt1.rand_conf())
    cbt2.goto_given_conf(cbt2.rand_conf())
    if cbt1.is_collided(other_robot_list=[cbt2], toggle_dbg=True):
        print('Collided!')
        cbt1.gen_meshmodel(rgb=rm.const.red).attach_to(base)
        cbt2.gen_meshmodel(rgb=rm.const.blue).attach_to(base)
        break
base.run()
