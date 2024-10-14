from wrs import wd, mgm

base = wd.World()
obj = mgm.GeometricModel(initor='./meshes/bone_v2.stl')
obj.attach_to(base)
base.run()