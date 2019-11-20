from Mesh import make_mesh_1d, plot_mesh


mesh = make_mesh_1d('test_mesh1', 0.0, 1.0,
                           10, 5, 'lg', 'uniform', 'random', transform= lambda x: x**2)


plot_mesh(mesh)
