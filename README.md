# VtkTools4UnstructeredMeshes
Vtk files are common to save complicated Datasets, such as Mesh Data from Finite Element Approximations. In my Bachelorthesis I used the Multilevel Monte Carlo Algorithm, to calculate the Exspectation value of a given output functional of the solution of the linear advection equation, which is a commonly model problem for parabolic partiel differential equations (see e.g. Mathematical Aspects of Discontinuous Galerkin Methods). 
The MLMC Algorithm and the Discontinuous Galerkin are implemented in the Codeframework, M++, respectively MLMC-M++, which is developed at the Karlsruher Institute of Technology at the Institute for Applied and Numerical Mathematics in the group Sientific Computing, for more see http://www.math.kit.edu/ianm3/page/mplusplus/en.

This Repository contains some work for better usage of vtk files in Python3, including reading to numpy, plotting meshes (scalar- and vector-valued) with matplotlib and combining meshes to an mlmc solution, which is a in mlmc way combined mean value of all solutions.

vtk_utilities.py contains the reading and plotting part and uses the package vtk (https://pypi.org/project/vtk/).
unstructered_mesh.py contains the classes which I used as container for caclulating the mlmc solution.

Plotting Example - vectorfield an concentration :
![Plotting Example](/plot7.png)

Another more complex Example - multiple timestamps and two levels in comparism :
![Another more complex Example](/sample_7_0.png)
![](/sample_coarse_7_0.png)

UML-Diagramm of the project:

![UML Diagramm](/klassenuml3.png)

