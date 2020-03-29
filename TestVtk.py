import unittest
import numpy as np
import os
import shutil
from vtk_utilities import *
from unstructured_mesh import *

minimal_scalar = "# vtk DataFile Version 2.0\nUnstructured Grid by M++\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 4 float\n0.000000   0.000000 0\n1.000000   0.000000 0\n1.000000   1.000000 0\n0.000000   1.000000 0\nCELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n8\nCELL_DATA 1\nSCALARS scalar_value float 1\nLOOKUP_TABLE default\n1.0"
minimal_vector = "# vtk DataFile Version 2.0\nUnstructured Grid by M++\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 4 float\n0.000000   0.000000 0\n1.000000   0.000000 0\n1.000000   1.000000 0\n0.000000   1.000000 0\nCELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n8\nCELL_DATA 1\nVECTORS vector_value float\n-1.000000 1.000000 0"
new_coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0],
                       [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.5],
                       [1.0, 1.0, 0.0]])
new_cells = np.array([int(i) for i in "4 0 1 4 5 4 1 2 3 4 4 5 4 7 6 4 4 3 8 7".split(" ")])


class TestVtk(unittest.TestCase):
    @classmethod
    def tearDown(cls):
        try:
            shutil.rmtree("Testsamples")
        except Exception as e:
            print(e)

    @classmethod
    def setUp(cls):
        try:
            os.mkdir("Testsamples")
            with open("Testsamples/vector.vtk", 'w') as file:
                file.write(minimal_vector)
            for i in range(10):
                with open("Testsamples/scalar.%04d.vtk" % i, 'w') as file:
                    file.write(minimal_scalar)
        except:
            pass


class TestReader(TestVtk):
    def test_scalar_reader(self):
        mesh = VtkScalarReader("scalar.0000.vtk", "Testsamples/")
        coords, cells, values = mesh.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        np.testing.assert_array_equal(values, np.array([1.0]))
        np.testing.assert_array_equal(np.array(cells), np.array([4, 0, 1, 2, 3]))
        np.testing.assert_array_equal(np.array(coords),
                                      np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
        for i in range(4):
            np.testing.assert_array_equal(cell_coords[0][i], np.array(coords[i]))

    def test_multi_scalar_reader(self):
        mesh = VtkGroupScalarReader("scalar.", "Testsamples/")
        coords, cells, values = mesh.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        np.testing.assert_array_equal(values, np.array([[1.0] for i in range(10)]))
        np.testing.assert_array_equal(np.array(cells), np.array([4, 0, 1, 2, 3]))
        np.testing.assert_array_equal(np.array(coords),
                                      np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
        for i in range(4):
            np.testing.assert_array_equal(cell_coords[0][i], np.array(coords[i]))

    def test_vector_reader(self):
        mesh = VtkVectorReader("vector.vtk", "Testsamples/")
        coords, cells, values = mesh.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        np.testing.assert_array_equal(values, np.array([[-1.0, 1.0, 0.0]]))
        np.testing.assert_array_equal(np.array(cells), np.array([4, 0, 1, 2, 3]))
        np.testing.assert_array_equal(np.array(coords),
                                      np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
        for i in range(4):
            np.testing.assert_array_equal(cell_coords[0][i], np.array(coords[i]))


class TestPlotting(TestVtk):

    def test_genplot(self):
        self.assertEqual(os.path.isfile("Testsamples/output.png"), False)
        s = VtkPlot()
        s.set_wd("Testsamples/")
        s.save("Testsamples/output.png")
        self.assertEqual(os.path.isfile("Testsamples/output.png"), True)

    def test_pcolormesh(self):
        s = VtkPlot()
        s.set_wd("Testsamples/")
        s.add_pcolormesh("scalar.0000.vtk")
        s.save("Testsamples/output.png")
        self.assertEqual(os.path.isfile("Testsamples/output.png"), True)

    def test_quivers(self):
        s = VtkPlot()
        s.set_wd("Testsamples/")
        s.add_quivers("vector.vtk")
        s.save("Testsamples/output.png")
        self.assertEqual(os.path.isfile("Testsamples/output.png"), True)

    def test_imshow(self):
        s = VtkPlot()
        s.set_wd("Testsamples/")
        s.add_imshow("scalar.0000.vtk")
        s.save("Testsamples/output.png")
        self.assertEqual(os.path.isfile("Testsamples/output.png"), True)


class TestScalarMeshes(TestVtk):
    def test_read(self):
        mesh = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        np.testing.assert_array_equal(mesh.values, np.array([1.0]))
        np.testing.assert_array_equal(np.array(mesh.cells), np.array([4, 0, 1, 2, 3]))
        np.testing.assert_array_equal(np.array(mesh.coordinates),
                                      np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_scaling(self):
        mesh = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        mesh.scaling(2.0)
        np.testing.assert_array_equal(mesh.values, np.array([2.0]))

    def test_cell_coord(self):
        mesh = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        for i in range(4):
            np.testing.assert_array_equal(mesh.cell_coords[0][i], np.array(mesh.coordinates[i]))

    def test_add(self):
        mesh1 = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        mesh2 = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        mesh3 = mesh1 + mesh2
        np.testing.assert_array_equal(mesh3.values, np.array([2.0]))
        mesh4 = sum([mesh1, mesh2, mesh3])
        np.testing.assert_array_equal(mesh4.values, np.array([4.0]))

    def test_sub(self):
        mesh1 = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        mesh2 = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        mesh1.scaling(3.0)
        mesh3 = mesh1 - mesh2
        np.testing.assert_array_equal(mesh3.values, np.array([2.0]))

    def test_upscale(self):
        mesh = UnstructuredScalarMesh("Testsamples/scalar.0000.vtk")
        new_mesh = mesh.Upscale_into(new_coords, new_cells)
        np.testing.assert_array_equal(new_mesh.coordinates, new_coords)
        np.testing.assert_array_equal(new_mesh.cells, new_cells)
        np.testing.assert_array_equal(new_mesh.values, np.array([1.0, 1.0, 1.0, 1.0]))


class TestVectorMeshes(TestVtk):
    def test_read(self):
        mesh = UnstructuredVectorMesh("Testsamples/vector.vtk")
        np.testing.assert_array_equal(mesh.values, np.array([[-1.0, 1.0, 0.0]]))
        np.testing.assert_array_equal(np.array(mesh.cells), np.array([4, 0, 1, 2, 3]))
        np.testing.assert_array_equal(np.array(mesh.coordinates),
                                      np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_scaling(self):
        mesh = UnstructuredVectorMesh("Testsamples/vector.vtk")
        mesh.scaling(2.0)
        np.testing.assert_array_equal(mesh.values, np.array([[-2.0, 2.0, 0.0]]))

    def test_cell_coord(self):
        mesh = UnstructuredVectorMesh("Testsamples/vector.vtk")
        for i in range(4):
            np.testing.assert_array_equal(mesh.cell_coords[0][i], np.array(mesh.coordinates[i]))

    def test_add(self):
        mesh1 = UnstructuredVectorMesh("Testsamples/vector.vtk")
        mesh2 = UnstructuredVectorMesh("Testsamples/vector.vtk")
        mesh3 = mesh1 + mesh2
        np.testing.assert_array_equal(mesh3.values, np.array([[-2.0, 2.0, 0.0]]))
        mesh4 = sum([mesh1, mesh2, mesh3])
        np.testing.assert_array_equal(mesh4.values, np.array([[-4.0, 4.0, 0.0]]))

    def test_sub(self):
        mesh1 = UnstructuredVectorMesh("Testsamples/vector.vtk")
        mesh2 = UnstructuredVectorMesh("Testsamples/vector.vtk")
        mesh1.scaling(3.0)
        mesh3 = mesh1 - mesh2
        np.testing.assert_array_equal(mesh3.values, np.array([[-2.0, 2.0, 0.0]]))

    def test_upscale(self):
        mesh = UnstructuredVectorMesh("Testsamples/vector.vtk")
        new_mesh = mesh.Upscale_into(new_coords, new_cells)
        np.testing.assert_array_equal(new_mesh.coordinates, new_coords)
        np.testing.assert_array_equal(new_mesh.cells, new_cells)
        np.testing.assert_array_equal(new_mesh.values, np.array([[-1.0, 1.0, 0.0] for i in range(4)]))


class TestMultiScalarMeshes(TestVtk):
    def test_read(self):
        mesh = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        np.testing.assert_array_equal(mesh.values, np.array([[1.0] for i in range(10)]))
        np.testing.assert_array_equal(np.array(mesh.cells), np.array([4, 0, 1, 2, 3]))
        np.testing.assert_array_equal(np.array(mesh.coordinates),
                                      np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_scaling(self):
        mesh = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        mesh.scaling(2.0)
        np.testing.assert_array_equal(mesh.values, np.array([[2.0] for i in range(10)]))

    def test_cell_coord(self):
        mesh = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        for i in range(4):
            np.testing.assert_array_equal(mesh.cell_coords[0][i], np.array(mesh.coordinates[i]))

    def test_add(self):
        mesh1 = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        mesh2 = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        mesh1.scaling(1.0)
        mesh3 = mesh1 + mesh2
        np.testing.assert_array_equal(mesh3.values, np.array([[2.0] for i in range(10)]))
        mesh4 = sum([mesh1, mesh2, mesh3])
        np.testing.assert_array_equal(mesh4.values, np.array([[4.0] for i in range(10)]))

    def test_sub(self):
        mesh1 = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        mesh2 = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        mesh1.scaling(3.0)
        mesh3 = mesh1 - mesh2
        np.testing.assert_array_equal(mesh3.values, np.array([[2.0] for i in range(10)]))

    def test_upscale(self):
        mesh = UnstructuredMultiScalarMesh("Testsamples/scalar.")
        new_mesh = mesh.Upscale_into(new_coords, new_cells)
        np.testing.assert_array_equal(new_mesh.coordinates, new_coords)
        np.testing.assert_array_equal(new_mesh.cells, new_cells)
        np.testing.assert_array_equal(new_mesh.values, np.array([[1.0, 1.0, 1.0, 1.0] for i in range(20)]))


if __name__ == '__main__':
    unittest.main()
