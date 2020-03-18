from vtk import *
import os
import numpy as np
from matplotlib.path import Path
from vtk_utilities import *


class UnstructuredMesh:
    def __init__(self):
        self.cells = None
        self.coordinates = None
        self.cell_type = 8
        self.cell_coords = None
        self.values = None
        self.wd = os.getcwd() + "/"

    def set_wd(self, nwd):
        self.wd = nwd

    def get_cell_coords(self):
        cell_length = self.cells[0] + 1
        cells = np.array_split(self.cells, len(self.cells) / cell_length)
        self.cell_coords = [[self.coordinates[cell[i]] for i in range(1, len(cell))] for cell in cells]

    def scaling(self, scale):
        pass

    def read_data(self, filename):
        pass

    def Upscale_into(self, new_coords, new_cells):
        new_umm = UnstructuredMultiScalarMesh()
        new_umm.wd = self.wd
        new_umm.cell_type = self.cell_type
        new_umm.coordinates = new_coords
        new_umm.cells = new_cells
        new_umm.get_cell_coords()
        return new_umm

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __radd__(self, other): # for usage of sum (sum begins with 0 + ...)
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def save(self, filename):
        header = ["# vtk DataFile Version 2.0\n", "Unstructured Grid by Python Frontend of M++\n", "ASCII\n",
                  "DATASET UNSTRUCTURED_GRID\n"]
        n_points = len(self.coordinates)
        _cells = np.array_split(self.cells, len(self.cells) / (self.cells[0] + 1))
        n_cells = len(_cells)
        with open(filename, 'w') as file:
            file.writelines(header)
            file.write("POINTS " + str(n_points) + " float\n")
            for coord in self.coordinates:
                file.write(str(coord[0]) + " " + str(coord[1]) + " " + str(coord[2]) + "\n")
            file.write("CELLS " + str(n_cells) + " " + str(n_cells * 5) + "\n")
            for cell in _cells:
                file.write(
                    str(cell[0]) + " " + str(cell[1]) + " " + str(cell[2]) + " " + str(cell[3]) + " " + str(
                        cell[4]) + "\n")
            file.write("CELL_TYPES " + str(n_cells) + "\n")
            for cell in _cells:
                file.write(str(self.cell_type) + "\n")
        return n_cells


class UnstructuredScalarMesh(UnstructuredMesh):
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.read_data(filename)

    def read_data(self, filename):
        sm = VtkScalarReader(filename, self.wd)
        self.coordinates, self.cells, self.values = sm.to_numpy()
        self.get_cell_coords()

    def scaling(self, scale):
        self.values = np.array([scale * el for el in self.values])

    def __add__(self, other):
        result = UnstructuredScalarMesh()
        result.set_wd(self.wd)
        result.cells = self.cells
        result.coordinates = self.coordinates
        result.cell_type = self.cell_type
        result.cell_coords = self.cell_coords
        result.values = np.array([self.values[j] + other.values[j] for j in range(len(self.values))])
        return result

    def __sub__(self, other):
        result = UnstructuredScalarMesh()
        result.set_wd(self.wd)
        result.cells = self.cells
        result.coordinates = self.coordinates
        result.cell_type = self.cell_type
        result.cell_coords = self.cell_coords
        result.values = np.array([self.values[j] - other.values[j] for j in range(len(self.values))])
        return result

    def save(self, filename):
        n_cells = super().save(filename)
        value_header = ["CELL_DATA " + str(n_cells) + "\n", "SCALARS scalar_value float 1\n",
                        "LOOKUP_TABLE default\n"]
        with open(filename, 'a') as file:
            file.writelines(value_header)
            for val in self.values:
                file.write(str(val) + "\n")

    def Upscale_into(self, new_coords, new_cells):
        if self.cell_coords is None:
            raise ValueError("You have to read data before upscaling it!")
        new_umm = super().Upscale_into(new_coords, new_cells)
        upscale_indices = get_indices_from_coarse(self.cell_coords, new_umm.cell_coords)
        new_values = []
        for j in range(len(new_umm.cell_coords)):  # cell j
            if upscale_indices == []:
                raise IndexError("Upscale Indices seem to be empty")
            else:
                new_values.append(self.values[upscale_indices[j]])
        new_umm.values = np.array(new_values)
        return new_umm

    def save_plot(self, vtkfile, pngfile):
        """
        :param vtkfile: file where the mesh is saved to, typically .vtk ending
        :param pngfile: file where the plot is saved to, typically .png/.jpg ending
        :return:
        """
        self.save(vtkfile)
        s = VtkPlot()
        s.set_wd(self.wd)
        s.add_pcolormesh(vtkfile)
        s.save(pngfile)


class UnstructuredMultiScalarMesh(UnstructuredMesh):
    def __init__(self, filename=None):
        super().__init__()
        self.values = []
        if filename is not None:
            self.read_data(filename)

    def read_data(self, filename):
        smm = VtkGroupScalarReader(filename, self.wd)
        self.coordinates, self.cells, self.values = smm.to_numpy()
        self.get_cell_coords()

    def Upscale_into(self, new_coords, new_cells):
        if self.cell_coords is None:
            raise ValueError("You have to read data before upscaling it!")
        new_umm = super().Upscale_into(new_coords, new_cells)
        upscale_indices = get_indices_from_coarse(self.cell_coords, new_umm.cell_coords)
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
            new_umm.values.append(nv)  # pure Upscale no Interpolation here
        return new_umm

    def scaling(self, scale):
        self.values = [np.array([scale * el for el in val]) for val in self.values]

    def __add__(self, other):
        result = UnstructuredMultiScalarMesh()
        result.set_wd(self.wd)
        result.cells = self.cells
        result.coordinates = self.coordinates
        result.cell_type = self.cell_type
        result.cell_coords = self.cell_coords
        nt = len(self.values)
        n = self.values[0].shape[0]
        result.values = [np.array([self.values[k][j] + other.values[k][j] for j in range(n)]) for k in range(nt)]
        return result

    def __sub__(self, other):
        result = UnstructuredMultiScalarMesh()
        result.set_wd(self.wd)
        result.cells = self.cells
        result.coordinates = self.coordinates
        result.cell_type = self.cell_type
        result.cell_coords = self.cell_coords
        nt = len(self.values)
        n = self.values[0].shape[0]
        result.values = [np.array([self.values[k][j] - other.values[k][j] for j in range(n)]) for k in range(nt)]
        return result

    def save(self, group_name):

        for i in range(len(self.values)):
            filename = group_name + "%04d.vtk" % i
            n_cells = super().save(filename)
            value_header = ["CELL_DATA " + str(n_cells) + "\n", "SCALARS scalar_value float 1\n",
                            "LOOKUP_TABLE default\n"]
            with open(filename, 'a') as file:
                file.writelines(value_header)
                for val in self.values[i]:
                    file.write(str(val) + "\n")


class UnstructuredVectorMesh(UnstructuredMesh):
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.read_data(filename)

    def read_data(self, filename):
        sm = VtkVectorReader(filename, self.wd)
        self.coordinates, self.cells, self.values = sm.to_numpy()
        self.get_cell_coords()

    def scaling(self, scale):
        self.values = np.array([np.array([scale * el for el in val]) for val in self.values])

    def __add__(self, other):
        result = UnstructuredVectorMesh()
        result.set_wd(self.wd)
        result.cells = self.cells
        result.coordinates = self.coordinates
        result.cell_type = self.cell_type
        result.cell_coords = self.cell_coords
        result.values = np.array([[self.values[i][j] + other.values[i][j] for j in range(len(self.values[i]))] for i in
                                range(len(self.values))])
        return result

    def __sub__(self, other):
        result = UnstructuredVectorMesh()
        result.set_wd(self.wd)
        result.cells = self.cells
        result.coordinates = self.coordinates
        result.cell_type = self.cell_type
        result.cell_coords = self.cell_coords
        result.values = np.array([[self.values[i][j] - other.values[i][j] for j in range(len(self.values[i]))] for i in
                                  range(len(self.values))])
        return result

    def save(self, filename):
        n_cells = super().save(filename)
        value_header = ["CELL_DATA " + str(n_cells) + "\n", "VECTORS vector_value float\n"]
        with open(filename, 'a') as file:
            file.writelines(value_header)
            for val in self.values:
                file.write(str(val[0]) + " " + str(val[1]) + " " + str(val[2]) + "\n")

    def Upscale_into(self, new_coords, new_cells):
        if self.cell_coords is None:
            raise ValueError("You have to read data before upscaling it!")
        new_umm = super().Upscale_into(new_coords, new_cells)
        upscale_indices = get_indices_from_coarse(self.cell_coords, new_umm.cell_coords)
        new_values = []
        if upscale_indices == []:
            raise IndexError("Upscale Indices seem to be empty")
        for j in range(len(new_umm.cell_coords)):  # cell j
            new_values.append(self.values[upscale_indices[j]])
        new_umm.values = np.array(new_values)
        return new_umm

    def save_plot(self, vtkfile, pngfile):
        """
        :param vtkfile: file where the mesh is saved to, typically .vtk ending
        :param pngfile: file where the plot is saved to, typically .png/.jpg ending
        :return:
        """
        self.save(vtkfile)
        s = VtkPlot()
        s.set_wd(self.wd)
        s.add_quivers(vtkfile)
        s.save(pngfile)


if __name__ == "__main__":
    mesh1 = UnstructuredMultiScalarMesh("../build/data/vtk/sample_4_0/U.")
    mesh2 = UnstructuredMultiScalarMesh("../build/data/vtk/sample_4_1/U.")
    mesh4 = mesh1 + mesh2
    print(mesh4.values[0])
    mesh4.save("sample/U.")
