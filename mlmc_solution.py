from unstructured_multi_mesh import *
import os
import numpy as np
from matplotlib.path import Path


def add_meshes(meshlst):
    if len(meshlst) == 1:
        return meshlst[0]
    if meshlst[0]._type == "permeability" or meshlst[0]._type == "flux":
        n = len(meshlst)
        values = [sum([meshlst[i].values[j] for i in range(n)]) for j in range(len(meshlst[0].values[0]))]
        resulting_mesh = UnstructuredMultiMesh(meshlst[0]._type)
        resulting_mesh.coordinates = meshlst[0].coordinates
        resulting_mesh.cells = meshlst[0].cells
        resulting_mesh.values = values
        return resulting_mesh
    elif meshlst[0]._type == "solution":
        n = len(meshlst)
        nt = len(meshlst[0].values)
        values = []
        for k in range(nt):
            _values = []
            for j in range(meshlst[0].values[0].shape[0]):
                _values.append(sum([meshlst[i].values[k][j] for i in range(n)]))
            values.append(np.array(_values))

        resulting_mesh = UnstructuredMultiMesh(meshlst[0]._type)
        resulting_mesh.coordinates = meshlst[0].coordinates
        resulting_mesh.cells = meshlst[0].cells
        resulting_mesh.values = values
        return resulting_mesh
    else:
        raise NotImplementedError


def subtract_meshes(mesh1, mesh2):
    mesh2.scaling(-1.0)
    resulting_mesh = add_meshes([mesh1, mesh2])
    return resulting_mesh


class McUmm:
    def __init__(self, working_dir, level, sample_amount, _type, only_fine=True, filename="mc/U."):
        self.working_dir = working_dir
        self.level = level
        self.sample_amount = sample_amount
        self._type = _type
        self.umm = UnstructuredMultiMesh(_type, working_dir=working_dir)
        self.only_fine = only_fine
        self.calculate()
        self.filename = filename

    def calculate(self):
        if self._type == "solution":
            _name = "U."
        else:
            _name = self._type + ".vtk"
        if self.only_fine:
            if self.sample_amount > 0:
                mesh_0 = UnstructuredMultiMesh(self._type, working_dir=self.working_dir,
                                               filename="sample_" + str(self.level) + "_0/" + _name)
                mesh_0.read_data()
                meshlst = [mesh_0]
            for i in range(1, self.sample_amount):
                mesh_i = UnstructuredMultiMesh(self._type, working_dir=self.working_dir,
                                               filename="sample_" + str(self.level) + "_" + str(
                                                   i) + "/" + _name)
                mesh_i.read_data()
                # if check_same_grid(mesh_0.coordinates, mesh_0.cells, mesh_i.coordinates, mesh_i.cells):
                meshlst.append(mesh_i)
                # else:
                #    raise IndexError("Meshes do not fit together")
            self.umm = add_meshes(meshlst)
            self.umm.scaling(1.0 / self.sample_amount)
        else:
            if self.sample_amount > 0:
                fine_mesh_0 = UnstructuredMultiMesh(self._type, working_dir=self.working_dir,
                                                    filename="sample_" + str(self.level) + "_0/" + _name)
                fine_mesh_0.read_data()
                fine_meshlst = [fine_mesh_0]
                coarse_mesh_0 = UnstructuredMultiMesh(self._type, working_dir=self.working_dir,
                                                      filename="sample_coarse_" + str(self.level) + "_0/" + _name)
                coarse_mesh_0.read_data()
                coarse_meshlst = [coarse_mesh_0]
            for i in range(1, self.sample_amount):
                fine_mesh_i = UnstructuredMultiMesh(self._type, working_dir=self.working_dir,
                                                    filename="sample_" + str(self.level) + "_" + str(
                                                        i) + "/" + _name)
                fine_mesh_i.read_data()
                coarse_mesh_i = UnstructuredMultiMesh(self._type, working_dir=self.working_dir,
                                                      filename="sample_coarse_" + str(self.level) + "_" + str(
                                                          i) + "/" + _name)
                coarse_mesh_i.read_data()
                # if check_same_grid(fine_mesh_0.coordinates, fine_mesh_0.cells, fine_mesh_i.coordinates,
                #                  fine_mesh_i.cells):
                fine_meshlst.append(fine_mesh_i)
                # else:
                #    raise IndexError("Meshes do not fit together")
                # if check_same_grid(coarse_mesh_0.coordinates, coarse_mesh_0.cells, coarse_mesh_i.coordinates,
                #                  coarse_mesh_i.cells):
                coarse_meshlst.append(coarse_mesh_i)
                # else:
                #   raise IndexError("Meshes do not fit together")
            fine_sum_mesh = add_meshes(fine_meshlst)
            fine_sum_mesh.scaling(1.0 / self.sample_amount)
            coarse_sum_mesh = add_meshes(coarse_meshlst)
            coarse_sum_mesh.scaling(1.0 / self.sample_amount)
            us_coarse_sum_mesh = coarse_sum_mesh.Upscale_into(fine_sum_mesh.coordinates, fine_sum_mesh.cells)
            self.umm = subtract_meshes(fine_sum_mesh, us_coarse_sum_mesh)

    def write_to_vtk(self):
        if self._type == "solution":
            header = ["# vtk DataFile Version 2.0\n", "Unstructured Grid by Python Frontend of M++\n", "ASCII\n",
                      "DATASET UNSTRUCTURED_GRID\n"]
            try:
                os.mkdir(self.working_dir + "mc_solutions")
                os.mkdir(self.working_dir + "mc_solutions/level" + str(self.level))
                print("Created Folder")
            except OSError as e:
                pass
            d = self.working_dir + "mc_solutions/level" + str(self.level)
            n_points = len(self.umm.coordinates)
            cell_length = self.umm.cells[0] + 1
            _cells = np.array_split(self.umm.cells, len(self.umm.cells) / cell_length)
            n_cells = len(_cells)
            value_header = ["CELL_DATA " + str(n_cells) + "\n", "SCALARS scalar_value float 1\n",
                            "LOOKUP_TABLE default\n"]
            for i in range(len(self.umm.values)):
                filename = d + "/" + "U.%04d.vtk" % i
                with open(filename, 'w') as file:
                    file.writelines(header)
                    file.write("POINTS " + str(n_points) + " float\n")
                    for coord in self.umm.coordinates:
                        file.write(str(coord[0]) + " " + str(coord[1]) + " " + str(coord[2]) + "\n")
                    file.write("CELLS " + str(n_cells) + " " + str(n_cells * 5) + "\n")
                    for cell in _cells:
                        file.write(
                            str(cell[0]) + " " + str(cell[1]) + " " + str(cell[2]) + " " + str(cell[3]) + " " + str(
                                cell[4]) + "\n")
                    file.write("CELL_TYPES " + str(n_cells) + "\n")
                    for cell in _cells:
                        file.write(str(self.umm.cell_type) + "\n")
                    file.writelines(value_header)
                    for val in self.umm.values[i]:
                        file.write(str(val) + "\n")
        else:
            raise NotImplementedError


class MlmcUmm:
    def __init__(self, working_dir, levels, sample_amount, _type, filename="mlmc/U."):
        self.working_dir = working_dir
        self.levels = levels
        self.sample_amount = sample_amount
        self._type = _type
        self.umm = UnstructuredMultiMesh(_type, working_dir=working_dir)
        self.calculate()
        self.filename = filename

    def calculate(self, return_mc_meshes=False):
        return_mc_meshes = []
        for i in range(len(self.levels)):
            mc = McUmm(self.working_dir, self.levels[i], self.sample_amount[i], self._type, only_fine=(i == 0))
            if return_mc_meshes:
                return_mc_meshes.append(mc)
            if i == 0:
                self.umm = mc.umm
            else:
                upscaled_all_before = self.umm.Upscale_into(mc.umm.coordinates, mc.umm.cells)
                self.umm = add_meshes([mc.umm, upscaled_all_before])
        if return_mc_meshes:
            return return_mc_meshes

    def write_to_vtk(self):
        if self._type == "solution":
            try:
                os.mkdir(self.working_dir + "/mlmc_solution")
                print("Created Folder")
            except OSError as e:
                pass

            header = ["# vtk DataFile Version 2.0\n", "Unstructured Grid by Python Frontend of M++\n", "ASCII\n",
                      "DATASET UNSTRUCTURED_GRID\n"]

            d = self.working_dir + "/mlmc_solution/"
            n_points = len(self.umm.coordinates)
            cell_length = self.umm.cells[0] + 1
            _cells = np.array_split(self.umm.cells, len(self.umm.cells) / cell_length)
            n_cells = len(_cells)
            value_header = ["CELL_DATA " + str(n_cells) + "\n", "SCALARS scalar_value float 1\n",
                            "LOOKUP_TABLE default\n"]
            for i in range(len(self.umm.values)):
                filename = d + "/" + "U.%04d.vtk" % i
                with open(filename, 'w') as file:
                    file.writelines(header)
                    file.write("POINTS " + str(n_points) + " float" + "\n")
                    for coord in self.umm.coordinates:
                        file.write(str(coord[0]) + " " + str(coord[1]) + " " + str(coord[2]) + "\n")
                    file.write("CELLS " + str(n_cells) + " " + str(n_cells * 5) + "\n")
                    for cell in _cells:
                        file.write(
                            str(cell[0]) + " " + str(cell[1]) + " " + str(cell[2]) + " " + str(cell[3]) + " " + str(
                                cell[4]) + "\n")
                    file.write("CELL_TYPES " + str(n_cells) + "\n")
                    for cell in _cells:
                        file.write(str(self.umm.cell_type) + "\n")
                    file.writelines(value_header)
                    for val in self.umm.values[i]:
                        file.write(str(val) + "\n")
        else:
            raise NotImplementedError


if __name__ == "__main__":
    eps = [0.01, 0.005, 0.003]
    inits = [[387, 15, 4, 2], [1500, 59, 4, 2], [4287, 163, 11, 2]]
    for i in range(3):
        mlmcsol = MlmcUmm(working_dir="../../mlmcExperiment200220/" + str(eps[i]) + "/vtk/", levels=[4, 5, 6, 7],
                          sample_amount=inits[i], _type="solution")
        mlmcsol.write_to_vtk()
