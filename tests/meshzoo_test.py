import meshzoo
import meshio


def make_square_mesh_uniform(nx, ny, save_mesh=False):

    points, elements = meshzoo.rectangle(
            xmin=0.0, xmax=10.0,
            ymin=0.0, ymax=10.0,
            nx=nx, ny=ny
            )

    print('Points:\n', points)
    print('Elements:\n', elements)

    if save_mesh:       # write to GMSH file format 2.2
                        # meshio can read 4.1 but not write

        filename = '../mesh/square_mesh_L10_' + 'x_' + str(nx) + 'y_' + str(ny) + ".msh"
        mesh = meshio.Mesh(points, {"triangle": elements})
        meshio.write(filename, mesh)

    return points[:, 0:2], elements


make_square_mesh_uniform(11, 11, save_mesh=True)

