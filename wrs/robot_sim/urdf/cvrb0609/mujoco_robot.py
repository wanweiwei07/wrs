import mujoco
import wrs.modeling.dynamics.mj_xml as mjx
import networkx as nx
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path('cvrb0609.urdf')

if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    from wrs import modeling as gm

    base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, .7])
    gm.gen_frame().attach_to(base)

    mj_model = mjx.MJModel("cvrb0609.urdf")
    # print(mj_model.chains)
    base.run_mj_physics(mj_model, 0)
    # mujoco.mj_forward(mj_model.model, mj_model.data)
    # for geom_dict in mj_model.body_geom_dict.values():
    #     for key, geom in geom_dict.items():
    #         pos = mj_model.data.geom_xpos[key]
    #         rotmat = mj_model.data.geom_xmat[key].reshape(3, 3)
    #         geom.pose = [pos, rotmat]
    #         geom.attach_to(base)plt.figure(figsize=(12, 8))
    joint_link_graph = mj_model.jlg
    pos = nx.spring_layout(joint_link_graph)
    nx.draw(joint_link_graph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
    plt.title("Joint-Link Tree")
    plt.show()

    base.run()
