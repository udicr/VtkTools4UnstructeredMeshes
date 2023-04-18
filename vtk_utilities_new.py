import os

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np

try:
    from vtk import vtkXMLGenericDataObjectReader
    from vtk.util.numpy_support import vtk_to_numpy
except ModuleNotFoundError:
    def vtkXMLGenericDataObjectReader():
        print("vtk module was not found")

    def vtk_to_numpy():
        print("vtk module was not found")


def check_same_grid(coords1, cells1, coords2, cells2):
    check = True
    for i in range(len(coords1)):
        if not np.all(coords1[i] == coords2[i]):
            check = False
    for i in range(len(cells1)):
        if not np.all(cells1[i] == cells2[i]):
            check = False
    return check


def get_cell_list(cells):
    cell_length = cells[0] + 1
    cells = np.array_split(cells, len(cells) / cell_length)
    cell_list = [[cell[i] for i in range(1, len(cell))] for cell in cells]
    return cell_list


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
    return np.array([(maximal_x + minimal_x) / 2.0, (maximal_y + minimal_y) / 2.0, 0.0])


def same_point(x, y, nearmode=True):  # 2D Reduction
    if nearmode:
        return abs(x[0] - y[0]) < 0.000001 and abs(x[1] - y[1]) < 0.000001
    else:
        return (x[0] == y[0]) and (x[1] == y[1])


'''
Upscaling Stuff for Tests and Usage see 
https://github.com/udicr/VtkTools4UnstructeredMeshes.git
'''


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
            midpoint = np.array(
                [shared_point[i] + 0.5 * (point[i] - shared_point[i]) for i in range(n)])
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


'''
Reading methods
'''


class VtkReader:
    def __init__(self, cwd):
        self.wd = cwd
        self.reader = vtkXMLGenericDataObjectReader()
        self.coordinates = None
        self.cells = None
        self.values = None

    def to_numpy(self):
        return self.coordinates, self.cells, self.values


class VtkSingleReader(VtkReader):
    def __init__(self, filename, cwd):
        super().__init__(cwd)
        self.filename = os.path.abspath(os.path.join(self.wd, filename))
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename)
        self.reader.SetFileName(self.filename)
        self.reader.Update()
        self.grid = self.reader.GetOutput()
        self.coordinates = vtk_to_numpy(self.grid.GetPoints().GetData())
        self.cells = vtk_to_numpy(self.grid.GetCells().GetData())


class VtkScalarReader(VtkSingleReader):
    def __init__(self, filename, cwd=os.getcwd(), convert_to_cell_data=False):
        super().__init__(filename, cwd)
        cell_data_scalars = self.grid.GetCellData().GetScalars()
        point_data_scalars = self.grid.GetPointData().GetScalars()
        self.has_point_data = False
        self.has_cell_data = False

        if point_data_scalars is not None:
            if convert_to_cell_data:
                cell_data_scalars = self.grid.GetPointData(). \
                    PointDatatoCellData().GetScalars()
            else:
                self.has_point_data = True
                self.values = vtk_to_numpy(point_data_scalars)

        if cell_data_scalars is not None:
            self.has_cell_data = True
            self.values = vtk_to_numpy(cell_data_scalars)


class VtkVectorReader(VtkSingleReader):
    def __init__(self, filename, cwd=os.getcwd()):
        super().__init__(filename, cwd)
        cell_data_vectors = self.grid.GetCellData().GetVectors()
        self.has_cell_vectors = False
        if cell_data_vectors is not None:
            self.has_cell_vectors = True
            self.values = vtk_to_numpy(cell_data_vectors)


class VtkGroupScalarReader(VtkReader):
    def __init__(self, group, cwd=os.getcwd()):
        super().__init__(cwd)
        vtu_dir = self.wd
        for d in group.split("/")[:-1]:
            vtu_dir += d + "/"
        n_sol = len([name for name in os.listdir(vtu_dir) if
                     os.path.isfile(os.path.join(vtu_dir, name)) if
                     group.split("/")[-1] in name])
        self.values = []
        for i in range(n_sol):
            self.reader.SetFileName(self.wd + group + "%04d.vtu" % i)
            self.reader.Update()
            grid = self.reader.GetOutput()
            points = grid.GetPoints()
            vtucells = grid.GetCells()
            cell_data = grid.GetCellData()
            cell_data_scalars = cell_data.GetScalars()
            _coordinates = vtk_to_numpy(points.GetData())
            _cells = vtk_to_numpy(vtucells.GetData())
            _values = vtk_to_numpy(cell_data_scalars)
            if i == 0:
                self.grid = self.reader.GetOutput()
                self.coordinates = _coordinates
                self.cells = _cells
                self.values.append(_values)
            else:
                self.values.append(_values)


'''
Plotting methods
'''


class VtkPlot:
    def __init__(self, **kwargs):
        self.fig = plt.figure(**kwargs)
        self.ax = self.fig.add_subplot(111)
        self.wd = os.getcwd()
        self.set_wd('../build/data/vtu/')

    def set_wd(self, nwd):
        self.wd = nwd

    '''
    2D non default routines
    '''

    def add_pcolormesh(self, file, cb=True, ax=None, **kwargs):
        if ax is None:
            ax = self.ax
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
        im = ax.pcolormesh(x, y, v, cmap='coolwarm', vmin=min(values), vmax=max(values),
                           **kwargs)
        if cb:
            self.fig.colorbar(im, ax=ax)

    def add_contourf(self, file, cb=True, **kwargs):
        solution = VtkScalarReader(file, self.wd)
        coords, cells, values = solution.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        x = sorted(set([coord[0] for coord in coords]))
        y = sorted(set([coord[1] for coord in coords]))
        n = len(x) - 1
        h = 1.0 / n
        v = np.zeros((n, n))
        for i in range(len(cell_coords)):
            cell = cell_coords[i]
            bottom_left_x = sorted(set([coord[0] for coord in cell]))[0]
            bottom_left_y = sorted(set([coord[1] for coord in cell]))[0]
            cell_index_x = x.index(bottom_left_x)
            cell_index_y = y.index(bottom_left_y)
            cell_value_index = i
            v[cell_index_y, cell_index_x] = values[cell_value_index]
        px = [i + h / 2. for i in x[:-1]]
        py = [i + h / 2. for i in y[:-1]]
        im = self.ax.contourf(px, py, v, cmap='coolwarm', vmin=min(values),
                              vmax=max(values), **kwargs)
        if cb:
            self.fig.colorbar(im, ax=self.ax)

    def add_imshow(self, file, cb=True, ax=None, **kwargs):
        if ax is None:
            ax = self.ax
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
        im = ax.imshow(v, cmap='coolwarm', origin='lower', vmin=min(values),
                       vmax=max(values), extent=[x[0], x[-1], y[0], y[-1]], **kwargs)
        if cb:
            self.fig.colorbar(im, ax=ax, use_gridspec=True)

    def add_point_data_interpolation(self, file, cb=True, interpolation=True,
                                     fading_stairs=10, ax=None, **kwargs):
        if ax is None:
            ax = self.ax
        solution = VtkScalarReader(file, self.wd)
        coords, cells, values = solution.to_numpy()
        x = list(set([coord[0] for coord in coords]))
        y = list(set([coord[1] for coord in coords]))
        plt.axis('scaled')
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        xl = [coords[i, 0] for i in range(len(coords))]
        yl = [coords[i, 1] for i in range(len(coords))]
        vl = [values[i] for i in range(len(coords))]
        if interpolation:
            for i in range(fading_stairs, 1, -1):
                im = ax.scatter(x=xl, y=yl, c=vl, cmap='coolwarm', marker='s',
                                linewidths=0, s=(i * 8) ** 2 + 8 ** 2,
                                alpha=1.0 / i,
                                **kwargs)
        im = ax.scatter(x=xl, y=yl, c=vl, cmap='coolwarm', marker='s', linewidths=0,
                        s=8 ** 2, **kwargs)

        if cb:
            self.fig.colorbar(im, ax=ax, use_gridspec=True)

    def add_quivers(self, file, quiver_filter=2, quiver_scale=0.15, ax=None, **kwargs):
        vec_reader = VtkVectorReader(file, self.wd)
        coords, cells, values = vec_reader.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        x = list(sorted(set([coord[0] for coord in coords])))
        y = sorted(set([coord[1] for coord in coords]))
        plt.xlim(x[0], x[-1])
        plt.ylim(y[0], y[-1])
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
        x_m = [(x[i + 1] + x[i]) / 2.0 for i in range(n)]
        y_m = [(y[i + 1] + y[i]) / 2.0 for i in range(n)]
        X, Y = np.meshgrid(x_m, y_m)
        ax.quiver(X[::quiver_filter, ::quiver_filter],
                  Y[::quiver_filter, ::quiver_filter],
                  quiver_scale * v1[::quiver_filter, ::quiver_filter],
                  quiver_scale * v2[::quiver_filter, ::quiver_filter], scale=1.0,
                  scale_units='inches', **kwargs)

    '''
    2D default routines
    '''

    def add_patch_collection(self, solution, cb=True, ax=None, vmin=None, vmax=None,
                             **kwargs):
        coords, cells, values = solution.to_numpy()
        cell_list = get_cell_list(cells)
        x = list(set([coord[0] for coord in coords]))
        y = list(set([coord[1] for coord in coords]))
        plt.axis('scaled')
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        patches = [
            Polygon(np.array([[coords[vertex][0], coords[vertex][1]] for vertex in cell]),
                    joinstyle='miter', linewidth=None) for cell in cell_list]
        plt.set_cmap('coolwarm')
        p = PatchCollection(patches, **kwargs)
        p.set_array(np.array(values))
        if not (vmin is None and vmax is None):
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            p.set_norm(norm)
        ax.add_collection(p)
        if cb:
            self.fig.colorbar(p, ax=ax)

    def add_point_data_patch_collection(self, solution, cb=True, ax=None, vmin=None,
                                        vmax=None, **kwargs):
        coords, cells, values = solution.to_numpy()
        cell_list = get_cell_list(cells)
        x = list(set([coord[0] for coord in coords]))
        y = list(set([coord[1] for coord in coords]))
        plt.axis('scaled')
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        patches = [
            Polygon(np.array([[coords[vertex][0], coords[vertex][1]] for vertex in cell]))
            for cell in cell_list
        ]
        value_list = [sum([values[vertex] for vertex in cell]) / len(cell) for cell in
                      cell_list]
        plt.set_cmap('coolwarm')
        p = PatchCollection(patches)
        p.set_array(np.array(value_list))
        if not (vmin is None and vmax is None):
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            p.set_norm(norm)
        ax.add_collection(p)
        if cb:
            self.fig.colorbar(p, ax=ax)

    def add_mesh(self, file, color='black', linewidth=0.5, scalar=True, point_data=False,
                 ax=None, **kwargs):
        if ax is None:
            ax = self.ax
        if scalar:
            solution = VtkScalarReader(file, self.wd)
        else:
            solution = VtkVectorReader(file)
        coords, cells, values = solution.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        x = sorted(set([coord[0] for coord in coords]))
        y = sorted(set([coord[1] for coord in coords]))
        plt.axis('scaled')
        plt.xlim(x[0], x[-1])
        plt.ylim(y[0], y[-1])
        for cell in cell_coords:
            for i in range(len(cell)):
                if i + 1 < len(cell):
                    x = [cell[i][0], cell[i + 1][0]]
                    y = [cell[i][1], cell[i + 1][1]]
                else:
                    x = [cell[i][0], cell[0][0]]
                    y = [cell[i][1], cell[0][1]]
                ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)

    def add_vtu(self, file, add_mesh=False, mesh_linewidth=0.1, cb=True, ax=None,
                vmin=None, vmax=None, **kwargs):
        if ax is None:
            ax = self.ax
        scalar_reader = VtkScalarReader(file, self.wd)
        vec_reader = VtkVectorReader(file, self.wd)

        if scalar_reader.has_cell_data:
            self.add_patch_collection(scalar_reader, cb=cb, ax=ax, vmin=vmin, vmax=vmax,
                                      **kwargs)
            if add_mesh:
                self.add_mesh('u.vtu', point_data=False, linewidth=mesh_linewidth, ax=ax)
        if scalar_reader.has_point_data:
            self.add_point_data_patch_collection(scalar_reader, cb=cb, ax=ax, vmin=vmin,
                                                 vmax=vmax,
                                                 **kwargs)
            if add_mesh:
                self.add_mesh('u.vtu', point_data=True, linewidth=mesh_linewidth, ax=ax)
        if vec_reader.has_cell_vectors:
            self.add_quivers(file, quiver_filter=1, quiver_scale=0.15, ax=ax, **kwargs)

    '''
    1D Vtk Data
    '''

    def add_cell_data_line(self, file, ax=None, color='black', **kwargs):
        if ax is None:
            ax = self.ax
        solution = VtkScalarReader(file, self.wd)
        coords, cells, values = solution.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        vertices = np.array([[point[0] for point in cell] for cell in cell_coords])
        for i in range(len(vertices) - 1):
            ax.plot(vertices[i], [values[i], values[i]], color=color, **kwargs)
        ax.plot(vertices[len(vertices) - 1],
                [values[len(vertices) - 1], values[len(vertices) - 1]], label=file,
                color=color, **kwargs)

    def add_point_data_line(self, file, ax=None, color='black', **kwargs):
        if ax is None:
            ax = self.ax
        solution = VtkScalarReader(file, self.wd)
        coords, cells, values = solution.to_numpy()
        vertices = np.array([point[0] for point in coords])
        sorted_vertices = sorted(vertices)
        sorted_values = [values[np.where(vertices == vertex)[0][0]] for vertex in
                         sorted_vertices]
        ax.plot(sorted_vertices, sorted_values, label=file, color=color, **kwargs)

    def add_1d_flux(self, file, ax=None, color='black', **kwargs):
        pass  # Todo

    def add_1d_vtu(self, file, cell_data=True, ax=None, color='black', **kwargs):
        if ax is None:
            ax = self.ax
        if cell_data:
            self.add_cell_data_line(file, ax, color, **kwargs)
        else:
            self.add_point_data_line(file, ax, color, **kwargs)

    @staticmethod
    def save(filename):
        plt.savefig(filename, dpi=400)
