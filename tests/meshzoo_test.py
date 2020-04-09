import meshzoo
import meshio


def make_square_mesh_uniform(nx, ny, save_mesh=False):

    points, elements = meshzoo.rectangle(
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            nx=nx, ny=ny,
            zigzag=True
            )

    print('Points:\n', points)
    print('Elements:\n', elements)

    if save_mesh:       # write to GMSH file format 2.2
                        # meshio can read 4.1 but not write

        filename = './square_mesh_' + 'x_' + str(nx) + 'y_' + str(ny) + ".msh"
        mesh = meshio.Mesh(points, {"triangle": elements})
        meshio.write(filename, mesh, file_format='gmsh2-ascii')

    return points[:, 0:2], elements


make_square_mesh_uniform(11, 11, save_mesh=True)

