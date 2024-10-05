from wrs import wd, rm, mgm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
frame_o = mgm.gen_frame(ax_length=.2)
frame_o.attach_to(base)
# rotmat = rm.rotmat_from_axangle(rm.np.array([1,1,1]), rm.pi/4)
rotmat = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
# frame_a = mgm.gen_mycframe(ax_length=.2, rotmat=rotmat)
frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat)
frame_a.attach_to(base)

print(rotmat)
base.run()
