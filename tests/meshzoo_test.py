import meshzoo
import meshio


def make_square_mesh_uniform(nx, ny, 
            xmin=0.0, xmax=10.0,
            ymin=0.0, ymax=10.0,
            save_mesh=False):

    points, elements = meshzoo.rectangle(
            xmin=0.0, xmax=xmax,
            ymin=0.0, ymax=ymax,
            nx=nx, ny=ny
            )

    print('Points:\n', points)
    print('Elements:\n', elements)

    if save_mesh:       # write to GMSH file format 2.2
                        # meshio can read 4.1 but not write

        filename = '../mesh/square_mesh_L20_' + 'x_' + str(nx) + 'y_' + str(ny) + ".msh"
        mesh = meshio.Mesh(points, {"triangle": elements})
        meshio.write(filename, mesh)

    return points[:, 0:2], elements


L = 20.
make_square_mesh_uniform(21, 21, 
            xmin=0.0, xmax=L,
            ymin=0.0, ymax=L,
            save_mesh=True)