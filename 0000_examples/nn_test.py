import pyvista as pv

pv.global_theme.silhouette.color = 'grey'
pv.global_theme.outline_color = 'white'
# pv.global_theme.show_edges = True
p = pv.Plotter(border_color="white", window_size=[1024, 768])
mesh = pv.read("objects/bunnysim.stl")
edges = mesh.extract_feature_edges(boundary_edges=True, manifold_edges = True, non_manifold_edges=True)
p.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True)
p.add_mesh(edges, color="red", line_width=5)
p.show()
