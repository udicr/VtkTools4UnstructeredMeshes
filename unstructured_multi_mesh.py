from vtk import *
from vtk.util.numpy_support import vtk_to_numpy
from vtk import (vtkUnstructuredGridReader, vtkDataSetMapper, vtkActor,
                 vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor)
import os
import numpy as np
from matplotlib.path import Path


def check_same_grid(coords1, cells1, coords2, cells2):
    check = True
    for i in range(len(coords1)):
        if not np.all(coords1[i] == coords2[i]):
            check = False
    for i in range(len(cells1)):
        if not np.all(cells1[i] == cells2[i]):
            check = False
    return check


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


class UnstructuredMultiMesh:
    def __init__(self, _type, working_dir=None, filename=None):
        self._type = _type
        self.filename = filename
        self.working_dir = working_dir
        self.coordinates = []
        self.cells = []
        self.cell_type = 8
        self.values = None
        self.cell_coords = None

    def read_data(self):
        reader = vtk.vtkGenericDataObjectReader()
        if self._type == "permability":
            if self.filename is None:
                filename = self.working_dir + "permeability.vtk"
            else:
                filename = self.working_dir + self.filename
            reader.SetFileName(filename)
            reader.Update()
            ug = reader.GetOutput()
            points = ug.GetPoints()
            vtkcells = ug.GetCells()
            cell_data = ug.GetCellData()
            cell_data_scalars = cell_data.GetScalars()
            coordinates = vtk_to_numpy(points.GetData())
            cells = vtk_to_numpy(vtkcells.GetData())
            values = vtk_to_numpy(cell_data_scalars)
            self.coordinates, self.cells, self.values = coordinates, cells, values
        elif self._type == "flux":
            if self.filename is None:
                filename = self.working_dir + "sample_5_0/flux.vtk"
            else:
                filename = self.working_dir + self.filename
            reader.SetFileName(filename)
            reader.Update()
            ug = reader.GetOutput()
            points = ug.GetPoints()
            vtkcells = ug.GetCells()
            cell_data = ug.GetCellData()
            coordinates = vtk_to_numpy(points.GetData())
            cells = vtk_to_numpy(vtkcells.GetData())
            vectors = cell_data.GetVectors()
            self.coordinates, self.cells, self.values = coordinates, cells, vtk_to_numpy(vectors)
        elif self._type == "solution":
            if self.filename is None:
                filename = "sample_5_0/U."
            else:
                filename = self.filename
            vtk_dir = self.working_dir
            for d in filename.split("/")[:-1]:
                vtk_dir += d + "/"
            n_sol = len([name for name in os.listdir(vtk_dir) if os.path.isfile(os.path.join(vtk_dir, name)) if
                         "U." in name])
            coordinates = cells = values = []
            for i in range(n_sol):
                _filename = self.working_dir + filename + "%04d.vtk" % i
                reader.SetFileName(_filename)
                reader.Update()
                ug = reader.GetOutput()
                points = ug.GetPoints()
                vtkcells = ug.GetCells()
                cell_data = ug.GetCellData()
                cell_data_scalars = cell_data.GetScalars()
                _coordinates = vtk_to_numpy(points.GetData())
                _cells = vtk_to_numpy(vtkcells.GetData())
                _values = vtk_to_numpy(cell_data_scalars)
                if i == 0:
                    coordinates = _coordinates
                    cells = _cells
                    values.append(_values)
                else:
                    values.append(_values)
                '''
                elif check_same_grid(coordinates, cells, _coordinates, _cells):
                    values.append(_values)
                else:
                    raise IndexError("Meshes of the Solutions do not fit")
                '''
            self.coordinates, self.cells, self.values = coordinates, cells, values
        else:
            raise NotImplementedError("Currently only supporting Types: 'solution','flux','permeability' ")

    def get_cell_coords(self):
        cell_length = self.cells[0] + 1
        cells = np.array_split(self.cells, len(self.cells) / cell_length)
        self.cell_coords = [[self.coordinates[cell[i]] for i in range(1, len(cell))] for cell in cells]

    def Upscale_into(self, new_coords, new_cells):
        if self.cell_coords is None:
            self.get_cell_coords()
        new_umm = UnstructuredMultiMesh(self._type, self.working_dir)
        new_umm.coordinates = new_coords
        new_umm.cells = new_cells
        new_umm.get_cell_coords()
        upscale_indices = get_indices_from_coarse(self.cell_coords, new_umm.cell_coords)
        if self._type == "solution":
            new_umm.values = []
            for mesh in self.values:  # mesh i
                new_values = []
                for j in range(len(new_umm.cell_coords)):  # cell j
                    if upscale_indices == []:
                        raise IndexError("Upscale Indices seem to be empty")
                    else:
                        new_values.append(mesh[upscale_indices[j]])
                nv = np.array(new_values)
                new_umm.values.append(nv)
                new_umm.values.append(
                    nv)  # pure Upscale no Interpolation here   warum hat coarse sol zu wenige timesols
            return new_umm
        elif self._type == "flux" or self._type == "permeability":
            new_values = []
            for new_cell in new_umm.cell_coords:
                new_values.append(
                    get_value_from_coarse(self.cell_coords, self.values, new_cell))  # faster for single meshes

            new_umm.values = np.array(new_values)

            return new_umm
        else:
            raise NotImplementedError("Currently only supporting Types: 'solution','flux','permeability' ")

    def scaling(self, scale_factor):
        if self._type == "solution":
            self.values = [np.array([scale_factor * el for el in val]) for val in self.values]
        elif self._type == "flux":
            self.values = np.array([[scale_factor * el for el in val] for val in self.values])
        elif self._type == "permeability":
            self.values = np.array([scale_factor * el for el in self.values])


if __name__ == "__main__":
    # print(get_max_fluxnorm())
    # co, ce, va = read_data("solution")

    '''
    sample = UnstructuredMultiMesh("solution", working_dir="../../mlmcExperiment200220/0.01/vtk/",
                                   filename="sample_6_0/U.")
    sample.read_data()
    sample.get_cell_coords()
    sample2 = UnstructuredMultiMesh("solution", working_dir="../../mlmcExperiment200220/0.01/vtk/",
                                    filename="sample_5_0/U.")
    sample2.read_data()
    sample2.get_cell_coords()
    upscale_ind2 = get_indices_from_coarse(sample2.cell_coords, sample.cell_coords)
    # print(len(sample.values[0]))
    print(upscale_ind2)
    '''