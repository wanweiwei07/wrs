from wrs import wd, rm, mgm, mcm, cbt

base = wd.World(cam_pos=rm.vec(1.2, .7, 1), lookat_pos=rm.vec(.0, 0, .15))
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), rgb=rm.vec(.7, .7, .7), alpha=.7)
ground.pos = rm.np.array([0, 0, -.51])
ground.attach_to(base)

robot = cbt.Cobotta()
robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
seed_jnt_values = None
for z in rm.np.linspace(.1, .6, 5):
    goal_pos = rm.np.array([.2, -.1, z])
    goal_rot = rm.rotmat_from_axangle(rm.const.y_ax, rm.pi * 1 / 2)
    mgm.gen_frame(goal_pos, goal_rot).attach_to(base)
    jnt_values = robot.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rot, seed_jnt_values=seed_jnt_values)
    print(jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        seed_jnt_values = jnt_values
    robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
base.run()
