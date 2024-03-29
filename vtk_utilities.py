import vtk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
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
    def __init__(self, **kwargs):
        self.fig = plt.figure(**kwargs)
        self.ax = self.fig.add_subplot(111)
        self.wd = os.getcwd()

    def set_wd(self, nwd):
        self.wd = nwd

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
        im = ax.pcolormesh(x, y, v, cmap='coolwarm', vmin=min(values), vmax=max(values), **kwargs)
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
        im = ax.imshow(v, cmap='coolwarm', origin='lower', vmin=min(values),
                       vmax=max(values), extent=[x[0], x[-1], y[0], y[-1]], **kwargs)
        if cb:
            self.fig.colorbar(im, ax=ax,use_gridspec=True)

    def add_quivers(self, file, quiver_filter=2, quiver_scale=0.15,ax=None, **kwargs):
        if ax is None:
            ax = self.ax
        flux = VtkVectorReader(file, self.wd)
        coords, cells, values = flux.to_numpy()
        cell_coords = get_cell_coords(cells, coords)
        x = sorted(set([coord[0] for coord in coords]))
        y = sorted(set([coord[1] for coord in coords]))
        n = len(x) 
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
        ax.quiver(X[::quiver_filter, ::quiver_filter], Y[::quiver_filter, ::quiver_filter],
                       quiver_scale * v1[::quiver_filter, ::quiver_filter],
                       quiver_scale * v2[::quiver_filter, ::quiver_filter], scale=1.0, scale_units='inches', **kwargs)

    def save(self, filename):
        plt.savefig(filename, dpi=400)


class VtkMPlot(VtkPlot):
    def __init__(self, **kwargs):
        super().__init__()
        self.fig = plt.figure(**kwargs)
        self.wd = os.getcwd()


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


def solution_quiver_plot(sol="sample_7_1/U.0128.vtk", flux="sample_7_1/flux.vtk"):
    s = VtkPlot()
    s.set_wd("../results/MLMCExperiment/0.001/vtk/")
    s.add_pcolormesh(sol)
    s.add_quivers(flux, quiver_filter=6, quiver_scale=0.15)


def solution_contour(sol="sample_7_1/U.0000.vtk"):
    s = VtkPlot()
    s.set_wd("../results/MLMCExperiment/0.001/vtk/")
    s.add_contourf(sol)


def smooth_solution():
    s = VtkPlot()
    s.set_wd("../results/MLMCExperiment/0.001/vtk/")
    s.add_imshow("sample_7_1/U.0000.vtk", interpolation='gaussian')



def solution_borders(sol="sample_4_0/U.0000.vtk"):
    my_dpi = 600.0
    s = VtkPlot(figsize=(4, 4), dpi=my_dpi)
    s.set_wd("../results/MLMCExperiment/0.001/vtk/")
    my_lw = 4 * my_dpi / (1024 * 32)
    s.add_pcolormesh(sol, cb=False, edgecolor='k', linewidth=my_lw)


def permeability_smoothplot(perm="sample_7_0/permeability.vtk"):
    s = VtkPlot()
    s.set_wd("../results/MLMCExperiment/0.001/vtk/")
    s.add_imshow(perm, interpolation='gaussian')


def permeability_quiverplot(perm="sample_7_0/permeability.vtk", flux="sample_7_0/flux.vtk"):
    s = VtkPlot()
    s.set_wd("../results/MLMCExperiment/0.01/vtk/")
    s.add_imshow(perm, cb=False, interpolation='gaussian')
    s.add_quivers(flux, quiver_filter=6, quiver_scale=0.14)


def solution_3(wd = "/home/user/Workspace/mlmcExperiment200220/0.01/vtk/",sample="sample_7_1/",quiver_filter=6,quiver_scale=0.10):
    vtk_dir = wd + sample
    nsol = len([name for name in os.listdir(vtk_dir) if os.path.isfile(os.path.join(vtk_dir, name)) if
         "U." in name])
    my_dpi = 400.0
    s = VtkPlot(figsize=(12, 4), dpi=my_dpi)
    plt.axis('off')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.25])
    s.set_wd(wd)
    s.ax1 = s.fig.add_subplot(gs[0])
    s.add_imshow(sample + "U.%04d.vtk" % 0, cb=False, ax=s.ax1)
    s.ax1.set_title('t = 0')
    s.ax2 = s.fig.add_subplot(gs[1])
    s.add_imshow(sample + "U.%04d.vtk" % int((nsol-1)/2), cb=False, ax=s.ax2)
    s.add_quivers(sample + "flux.vtk",quiver_filter=quiver_filter, quiver_scale=quiver_scale, ax=s.ax2)
    s.ax2.set_title('t = 0.5')
    s.ax3 = s.fig.add_subplot(gs[2])
    s.add_imshow(sample + "U.%04d.vtk" % int(nsol-1), cb=True, ax=s.ax3)
    s.ax3.set_title('t = 1.0')

def save_plots():
    solution_3(wd="../results/MLMCExperiment/0.001/vtk/", sample="sample_4_1/", quiver_filter=1, quiver_scale=0.10)
    plt.savefig("plots/sample_4_1.png")
    solution_3(wd="../results/MLMCExperiment/0.001/vtk/", sample="sample_coarse_5_1/", quiver_filter=1,
               quiver_scale=0.10)
    plt.savefig("plots/sample_coarse_5_1.png")
    solution_3(wd="../results/MLMCExperiment/0.001/vtk/", sample="sample_5_1/", quiver_filter=2, quiver_scale=0.10)
    plt.savefig("plots/sample_5_1.png")
    solution_3(wd="../results/MLMCExperiment/0.001/vtk/", sample="sample_coarse_6_0/", quiver_filter=2,
               quiver_scale=0.12)
    plt.savefig("plots/sample_coarse_6_0.png")
    solution_3(wd="../results/MLMCExperiment/0.001/vtk/", sample="sample_6_0/", quiver_filter=4, quiver_scale=0.12)
    plt.savefig("plots/sample_6_0.png")
    solution_3(wd="../results/MLMCExperiment/0.001/vtk/", sample="sample_coarse_7_0/", quiver_filter=4,
               quiver_scale=0.12)
    plt.savefig("plots/sample_coarse_7_0.png")
    solution_3(wd="../results/MLMCExperiment/0.001/vtk/", sample="sample_7_0/", quiver_filter=8, quiver_scale=0.12)
    plt.savefig("plots/sample_7_0.png")
    permeability_quiverplot()
    plt.savefig("plots/permeability_quiver.png")
    permeability_smoothplot()
    plt.savefig("plots/permeability.png")
    for lvl in ["4","5","6","7"]:
        solution_borders(sol="sample_"+lvl+"_0/U.0000.vtk")
        plt.savefig("plots/mesh"+lvl+".png")
    smooth_solution()
    plt.savefig("plots/anfangsbedingung.png")
    solution_contour()
    plt.savefig("solutioncontour.png")
    solution_quiver_plot()
    plt.savefig("solutionquiver.png")


if __name__ == "__main__":
    #s = VtkPlot()
    #s.set_wd("/local/buchholz/VtkTools4UnstructeredMeshes")
    #s.add_imshow("sample_4_0/U.0000.vtk", interpolation='gaussian')
    #plt.savefig("anfangsbedingung.png")

    solution_3(wd="",sample="sample_4_0/",quiver_filter=1,quiver_scale=0.10)
    plt.show()

