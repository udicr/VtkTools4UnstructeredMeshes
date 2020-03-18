from vtk import *
from vtk.util.numpy_support import vtk_to_numpy
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def check_same_grid(coords1, cells1, coords2, cells2):
    check = True
    for i in range(len(coords1)):
        if not np.all(coords1[i] == coords2[i]):
            check = False
    for i in range(len(cells1)):
        if not np.all(cells1[i] == cells2[i]):
            check = False
    return check


def get_cell_coords(cells, coords):
    cell_length = cells[0] + 1
    cells = np.array_split(cells, len(cells) / cell_length)
    cell_coords = [[coords[cell[i]] for i in range(1, len(cell))] for cell in cells]
    return cell_coords


def get_midpoint(cell_coord):
    minimal_x = min([coord[0] for coord in cell_coord])
    minimal_y = min([coord[1] for coord in cell_coord])
    maximal_x = max([coord[0] for coord in cell_coord])
    maximal_y = max([coord[1] for coord in cell_coord])
    return np.array([(maximal_x - minimal_x) / 2.0, (maximal_y - minimal_y) / 2.0, 0.0])


def same_point(x, y):  # 2D Reduction
    nearmode = True
    if nearmode:
        return abs(x[0] - y[0]) < 0.000001 and abs(x[1] - y[1]) < 0.000001
    else:
        return (x[0] == y[0]) and (x[1] == y[1])


def get_share_point(coarse_cell, fine_cell):
    shared_point = None
    for x in coarse_cell:
        for y in fine_cell:
            if same_point(x, y):
                shared_point = y
    return shared_point


def is_inside(coarse_cell, fine_cell):
    shared_point = get_share_point(coarse_cell, fine_cell)
    if shared_point is None:
        return False
    on_midpoint = 0
    for point in coarse_cell:
        if not same_point(point, shared_point):
            n = point.size
            midpoint = np.array([shared_point[i] + 0.5 * (point[i] - shared_point[i]) for i in range(n)])
            for y in fine_cell:
                if same_point(midpoint, y):
                    on_midpoint += 1
                    if on_midpoint > 1:
                        return True
    return False


def get_value_from_coarse(coarse_cell_coords, coarse_value,
                          fine_cell_coord):
    for i in range(len(coarse_cell_coords)):
        coarse_cell_coord = coarse_cell_coords[i]
        if is_inside(coarse_cell_coord, fine_cell_coord):
            return coarse_value[i]
    raise IndexError("Fine Cell seems not to be covered properly by Coarse Mesh")


def get_indices_from_coarse(coarse_cell_coords,
                            fine_cell_coords):
    indices = []
    for fine_cell_coord in fine_cell_coords:
        for i in range(len(coarse_cell_coords)):
            if is_inside(coarse_cell_coords[i], fine_cell_coord):
                indices.append(i)
    return indices



class VtkReader:
    def __init__(self, cwd):
        self.wd = cwd
        self.reader = vtk.vtkGenericDataObjectReader()
        self.coordinates = None
        self.cells = None
        self.values = None

    def to_numpy(self):
        return self.coordinates, self.cells, self.values


class VtkSingleReader(VtkReader):
    def __init__(self, filename, cwd):
        super().__init__(cwd)
        self.filename = self.wd + filename
        self.reader.SetFileName(self.filename)
        self.reader.Update()
        self.grid = self.reader.GetOutput()
        self.coordinates = vtk_to_numpy(self.grid.GetPoints().GetData())
        self.cells = vtk_to_numpy(self.grid.GetCells().GetData())


class VtkScalarReader(VtkSingleReader):
    def __init__(self, filename, cwd=os.getcwd()):
        super().__init__(filename, cwd)
        cell_data = self.grid.GetCellData()
        cell_data_scalars = cell_data.GetScalars()
        self.values = vtk_to_numpy(cell_data_scalars)


class VtkVectorReader(VtkSingleReader):
    def __init__(self, filename, cwd=os.getcwd()):
        super().__init__(filename, cwd)
        cell_data = self.grid.GetCellData()
        cell_data_vectors = cell_data.GetVectors()
        self.values = vtk_to_numpy(cell_data_vectors)


class VtkGroupScalarReader(VtkReader):
    def __init__(self, group, cwd=os.getcwd()):
        super().__init__(cwd)
        vtk_dir = self.wd
        for d in group.split("/")[:-1]:
            vtk_dir += d + "/"
        n_sol = len([name for name in os.listdir(vtk_dir) if os.path.isfile(os.path.join(vtk_dir, name)) if
                     group.split("/")[-1] in name])
        self.values = []
        for i in range(n_sol):
            self.reader.SetFileName(self.wd + group + "%04d.vtk" % i)
            self.reader.Update()
            grid = self.reader.GetOutput()
            points = grid.GetPoints()
            vtkcells = grid.GetCells()
            cell_data = grid.GetCellData()
            cell_data_scalars = cell_data.GetScalars()
            _coordinates = vtk_to_numpy(points.GetData())
            _cells = vtk_to_numpy(vtkcells.GetData())
            _values = vtk_to_numpy(cell_data_scalars)
            if i == 0:
                self.grid = self.reader.GetOutput()
                self.coordinates = _coordinates
                self.cells = _cells
                self.values.append(_values)
            # bypassed check same grid maybe want to check here
            else:
                self.values.append(_values)


class VtkPlot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.wd = os.getcwd()

    def set_wd(self, nwd):
        self.wd = nwd

    def add_pcolormesh(self, file, cb=True, **kwargs):
        solution = VtkScalarReader(file, self.wd)
        coords, cells, values = solution.to_numpy()
        cell_coords = get_cell_coords(cells, coords)

        x = sorted(set([coord[0] for coord in coords]))
        y = sorted(set([coord[1] for coord in coords]))
        n = len(x) - 1
        v = np.zeros((n, n))
        for i in range(len(cell_coords)):
            cell = cell_coords[i]
            bottom_left_x = sorted(set([coord[0] for coord in cell]))[0]
            bottom_left_y = sorted(set([coord[1] for coord in cell]))[0]
            cell_index_x = x.index(bottom_left_x)
            cell_index_y = y.index(bottom_left_y)
            cell_value_index = i
            v[cell_index_y, cell_index_x] = values[cell_value_index]
        im = self.ax.pcolormesh(x, y, v, cmap='coolwarm', vmin=min(values), vmax=max(values), **kwargs)
        if cb:
            self.fig.colorbar(im, ax=self.ax)

    def add_quivers(self, file, quiver_filter=2, quiver_scale=0.15, **kwargs):
        flux = VtkVectorReader(file, self.wd)
        coords, cells, values = flux.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        x = sorted(set([coord[0] for coord in coords]))
        y = sorted(set([coord[1] for coord in coords]))
        n = len(x) - 1
        v1 = np.zeros((n, n))
        v2 = np.zeros((n, n))
        for i in range(len(cell_coords)):
            cell = cell_coords[i]
            bottom_left_x = sorted(set([coord[0] for coord in cell]))[0]
            bottom_left_y = sorted(set([coord[1] for coord in cell]))[0]
            cell_index_x = x.index(bottom_left_x)
            cell_index_y = y.index(bottom_left_y)
            cell_value_index = i
            v1[cell_index_y, cell_index_x] = values[cell_value_index][0]
            v2[cell_index_y, cell_index_x] = values[cell_value_index][1]
        X, Y = np.meshgrid(x, y)
        self.ax.quiver(X[::quiver_filter, ::quiver_filter], Y[::quiver_filter, ::quiver_filter],
                       quiver_scale * v1[::quiver_filter, ::quiver_filter],
                       quiver_scale * v2[::quiver_filter, ::quiver_filter], scale=1.0, scale_units='inches', **kwargs)

    def save(self, filename):
        plt.savefig(filename)


class AnimatedVtkPlot(VtkPlot):  # Todo Future ( see deprecated code )
    def __init__(self, _frames):
        super().__init__()
        self.step = 0
        self.frames = _frames
        self.interval = 100
        self.flux = None

    def add_solution(self, group, **kwargs):
        pass

    def animate(self):
        pass

    def add_flux(self, file, **kwargs):
        pass

    def save(self, filename):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=256, interval=100)
        ani.save("animation1.mp4")  # , bitrate=-1


def get_max_fluxnorm(working_dir="../build/data/vtk/", filename="sample_4_1/flux.vtk"):
    """
    :param filename: path to flux file
    :return: maximum Norm (euclidean) of the given flux file
    """
    flux = VtkVectorReader(filename, working_dir)
    coords, cells, vecs = flux.to_numpy()
    norms = [sum([i * i for i in vec]) for vec in vecs]
    return max(norms)


def solution_quiver_plot():
    s = VtkPlot()
    s.set_wd("/home/user/Workspace/mlmcExperiment200220/0.01/vtk/")
    s.add_pcolormesh("sample_5_1/U.0128.vtk")
    s.add_quivers("sample_5_1/flux.vtk", quiver_filter=2)
    plt.show()


if __name__ == "__main__":
    solution_quiver_plot()
