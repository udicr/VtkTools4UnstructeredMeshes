from unstructured_mesh import *
from vtk_utilities import *
import os
import numpy as np
from matplotlib.path import Path
import abc


class McMesh:
    def __init__(self, level, sample_amount, working_dir, only_fine):
        self.wd = working_dir
        self.lvl = level
        self.amount = sample_amount
        self.only_fine = only_fine
        self.mesh = None

    @abc.abstractmethod
    def calculate(self):
        pass

    def set_wd(self, nwd):
        self.wd = nwd


class McScalarMesh(McMesh):
    def __init__(self, level, sample_amount, working_dir, filename, only_fine=False):
        super().__init__(level, sample_amount, working_dir, only_fine)
        self.filename = filename
        self.calculate()

    def calculate(self):
        if self.only_fine:
            mesh_0 = UnstructuredScalarMesh("sample_" + str(self.lvl) + "_0/" + self.filename)
            meshlst = [mesh_0]
            for j in range(1, self.amount):
                mesh_j = UnstructuredScalarMesh("sample_" + str(self.lvl) + "_" + str(
                    j) + "/" + self.filename)
                meshlst.append(mesh_j)
            self.mesh = sum(meshlst)
            self.mesh.scaling(1.0 / self.amount)
        else:
            fine_mesh_0 = UnstructuredScalarMesh("sample_" + str(self.lvl) + "_0/" + self.filename)
            fine_meshlst = [fine_mesh_0]
            coarse_mesh_0 = UnstructuredScalarMesh("sample_coarse_" + str(self.lvl) + "_0/" + self.filename)
            coarse_meshlst = [coarse_mesh_0]
            for j in range(1, self.amount):
                fine_mesh_j = UnstructuredScalarMesh(
                    "sample_" + str(self.lvl) + "_" + str(j) + "/" + self.filename)
                coarse_mesh_j = UnstructuredScalarMesh(
                    "sample_coarse_" + str(self.lvl) + "_" + str(j) + "/" + self.filename)
                fine_meshlst.append(fine_mesh_j)
                coarse_meshlst.append(coarse_mesh_j)
            fine_sum_mesh = sum(fine_meshlst)
            fine_sum_mesh.scaling(1.0 / self.amount)
            coarse_sum_mesh = sum(coarse_meshlst)
            coarse_sum_mesh.scaling(1.0 / self.amount)
            us_coarse_sum_mesh = coarse_sum_mesh.Upscale_into(fine_sum_mesh.coordinates, fine_sum_mesh.cells)
            self.mesh = fine_sum_mesh - us_coarse_sum_mesh


class McVectorMesh(McMesh):
    def __init__(self, level, sample_amount, working_dir, filename, only_fine=False):
        super().__init__(level, sample_amount, working_dir, only_fine)
        self.filename = filename
        self.calculate()

    def calculate(self):
        if self.only_fine:
            mesh_0 = UnstructuredVectorMesh(self.wd +"sample_" + str(self.lvl) + "_0/" + self.filename)
            meshlst = [mesh_0]
            for j in range(1, self.amount):
                mesh_j = UnstructuredVectorMesh(self.wd + "sample_" + str(self.lvl) + "_" + str(
                    j) + "/flux.vtk")
                meshlst.append(mesh_j)
            self.mesh = sum(meshlst)
            self.mesh.scaling(1.0 / self.amount)
        else:
            fine_mesh_0 = UnstructuredVectorMesh(self.wd + "sample_" + str(self.lvl) + "_0/" + self.filename)
            fine_meshlst = [fine_mesh_0]
            coarse_mesh_0 = UnstructuredVectorMesh(self.wd + "sample_coarse_" + str(self.lvl) + "_0/" + self.filename)
            coarse_meshlst = [coarse_mesh_0]
            for j in range(1, self.amount):
                fine_mesh_j = UnstructuredVectorMesh(self.wd + "sample_" + str(self.lvl) + "_" + str(
                    j) + "/" + self.filename)
                coarse_mesh_j = UnstructuredVectorMesh(self.wd + "sample_coarse_" + str(self.lvl) + "_" + str(
                    j) + "/" + self.filename)
                fine_meshlst.append(fine_mesh_j)
                coarse_meshlst.append(coarse_mesh_j)
            fine_sum_mesh = sum(fine_meshlst)
            fine_sum_mesh.scaling(1.0 / self.amount)
            coarse_sum_mesh = sum(coarse_meshlst)
            coarse_sum_mesh.scaling(1.0 / self.amount)
            us_coarse_sum_mesh = coarse_sum_mesh.Upscale_into(fine_sum_mesh.coordinates, fine_sum_mesh.cells)
            self.mesh = fine_sum_mesh - us_coarse_sum_mesh


class McMultiScalarMesh(McMesh):
    def __init__(self, level, sample_amount, working_dir, filename, only_fine=False):
        super().__init__(level, sample_amount, working_dir, only_fine)
        self.filename = filename
        self.calculate()

    def calculate(self):
        if self.only_fine:
            mesh_0 = UnstructuredMultiScalarMesh(self.wd + "sample_" + str(self.lvl) + "_0/U.")
            meshlst = [mesh_0]
            for j in range(1, self.amount):
                mesh_j = UnstructuredMultiScalarMesh(self.wd + "sample_" + str(self.lvl) + "_" + str(
                    j) + "/U.")
                meshlst.append(mesh_j)
            self.mesh = sum(meshlst)
            self.mesh.scaling(1.0 / self.amount)
        else:
            fine_mesh_0 = UnstructuredMultiScalarMesh(self.wd + "sample_" + str(self.lvl) + "_0/U.")
            fine_meshlst = [fine_mesh_0]
            coarse_mesh_0 = UnstructuredMultiScalarMesh(self.wd + "sample_coarse_" + str(self.lvl) + "_0/U.")
            coarse_meshlst = [coarse_mesh_0]
            for j in range(1, self.amount):
                fine_mesh_j = UnstructuredMultiScalarMesh(self.wd + "sample_" + str(self.lvl) + "_" + str(
                    j) + "/U.")
                coarse_mesh_j = UnstructuredMultiScalarMesh(self.wd + "sample_coarse_" + str(self.lvl) + "_" + str(
                    j) + "/U.")
                fine_meshlst.append(fine_mesh_j)
                coarse_meshlst.append(coarse_mesh_j)
            fine_sum_mesh = sum(fine_meshlst)
            fine_sum_mesh.scaling(1.0 / self.amount)
            coarse_sum_mesh = sum(coarse_meshlst)
            coarse_sum_mesh.scaling(1.0 / self.amount)
            us_coarse_sum_mesh = coarse_sum_mesh.Upscale_into(fine_sum_mesh.coordinates, fine_sum_mesh.cells)
            self.mesh = fine_sum_mesh - us_coarse_sum_mesh


class MlmcMesh:
    def __init__(self, levels, sample_amount, working_dir, filename):
        self.wd = working_dir
        self.lvls = levels
        self.filename = filename
        self.amounts = sample_amount
        self.mesh = None
        self.calculate()

    @abc.abstractmethod
    def get_mcmeshes(self, lvl, amount, only_fine=False):
        pass

    def calculate(self, return_mc_meshes=False):
        mc_meshes = []
        for k in range(len(self.lvls)):
            mc = self.get_mcmeshes(self.lvls[k], self.amounts[k], only_fine=(k == 0))
            if return_mc_meshes:
                mc_meshes.append(mc)
            if k == 0:
                self.mesh = mc.mesh
            else:
                upscaled_all_before = self.mesh.Upscale_into(mc.mesh.coordinates, mc.mesh.cells)
                self.mesh = mc.mesh + upscaled_all_before
        if return_mc_meshes:
            return return_mc_meshes


class MlmcScalarMesh(MlmcMesh):
    def get_mcmeshes(self, lvl, amount, only_fine=False):
        mc = McScalarMesh(lvl, amount, self.wd, self.filename, only_fine)
        return mc


class MlmcVectorMesh(MlmcMesh):
    def get_mcmeshes(self, lvl, amount, only_fine=False):
        mc = McVectorMesh(lvl, amount, self.wd, self.filename, only_fine)
        return mc


class MlmcMultiScalarMesh(MlmcMesh):
    def get_mcmeshes(self, lvl, amount, only_fine=False):
        mc = McMultiScalarMesh(lvl, amount, self.wd, self.filename, only_fine)
        return mc


if __name__ == "__main__":
    eps = [0.01, 0.005, 0.003]
    inits = [[387, 15, 4, 2], [1500, 59, 4, 2], [4287, 163, 11, 2]]
    for i in range(1):
        # mlmcsol = MlmcMultiScalarMesh(working_dir="../../mlmcExperiment200220/" + str(eps[i]) + "/vtk/",
        #                              levels=[4, 5, 6, 7],
        #                             sample_amount=inits[i], filename="U.")
        mlmcflux = MlmcVectorMesh(working_dir="../../mlmcExperiment200220/" + str(eps[i]) + "/vtk/",
                                  levels=[4, 5, 6, 7],
                                  sample_amount=inits[i], filename="flux.vtk")
        #mlmcperm = MlmcScalarMesh(working_dir="../../mlmcExperiment200220/" + str(eps[i]) + "/vtk/",
         #                         levels=[4, 5, 6, 7],
          #                        sample_amount=inits[i], filename="permeability.vtk")
        try:
            mlmcflux.mesh.save(mlmcflux.wd + "mlmc/" + "flux.vtk")
           # mlmcperm.mesh.save(mlmcflux.wd + "mlmc/" + "permeability.vtk")
        except OSError as e:
            mlmcflux.mesh.save("flux.vtk")
            #mlmcflux.mesh.save("permeability.vtk")
