import visualization.panda.world as wd
import modeling.geometric_model as gm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
gm.gen_frame(length=.2).attach_to(base)

base.run()