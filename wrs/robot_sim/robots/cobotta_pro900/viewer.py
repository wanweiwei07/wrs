from wrs import wd, rm, mgm

base = wd.World(cam_pos=rm.vec(2, 1, 2), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame().attach_to(base)
obj = mgm.GeometricModel(initor='./meshes/bone_v2.stl')
obj.rgb = rm.const.ivory
print(rm.np.max(obj.trm_mesh.vertices[:,0]))
print(rm.np.min(obj.trm_mesh.vertices[:,0]))
print(rm.np.max(obj.trm_mesh.vertices[:,1]))
print(rm.np.min(obj.trm_mesh.vertices[:,1]))
print(rm.np.max(obj.trm_mesh.vertices[:,2]))
print(rm.np.min(obj.trm_mesh.vertices[:,2]))
obj.attach_to(base)
base.run()
