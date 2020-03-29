import subprocess
import os
import sys
from tp_utilities import *
from vtk_utilities import *
import matplotlib.pyplot as plt
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy
from mlmc_solution import *

working_dir = "../build"
os.chdir(working_dir)


def buildMpp():
    subprocess.run(["cmake", ".."], cwd=working_dir)
    stdout = subprocess.Popen(["make", "-j"], cwd=working_dir, stdout=subprocess.PIPE)
    for line in iter(stdout.stdout.readline, b''):
        l = line.decode('utf-8')
        sys.stdout.write(l)
    return "Process finished with exit code 0"


def save_statistics():
    stdout = subprocess.Popen(["python3", "../tools/plot_statistics.py", "test"],
                              cwd=working_dir, stdout=subprocess.PIPE)
    for line in iter(stdout.stdout.readline, b''):
        l = line.decode('utf-8')
        sys.stdout.write(l)
    return "Process finished with exit code 0"


def ConvergenceTest():
    def runMpp(kernels, *args, conf=""):
        runParameters = ["mpirun", "-np", str(kernels),
                         "MLMC-M++", "mlmc_tpconvergencetest"]
        for arg in args:
            runParameters.append(arg)

        stdout = subprocess.Popen(runParameters, stdout=subprocess.PIPE, cwd=working_dir)
        for line in iter(stdout.stdout.readline, b''):
            l = line.decode('utf-8')
            sys.stdout.write(l)
        return "Process finished with exit code 0"

    delete_old()
    stdout = runMpp(32)
    save("mlmcConvergenceTest/")


def mlmcExperiment1(eps,initLevels,initSampleAmount):
    def runMpp(kernels, *args, conf=""):
        runParameters = ["mpirun", "-np", str(kernels),
                         "MLMC-M++", "mconly=false",
                         "Experiment=MLMCExperiment",
                         "epsilon=" + str(eps),
			 "initLevels="+initLevels,
			 "initSampleAmount="+initSampleAmount,
                         "enablePython=0"
                         ]
        for arg in args:
            runParameters.append(arg)

        stdout = subprocess.Popen(runParameters, stdout=subprocess.PIPE, cwd=working_dir)
        for line in iter(stdout.stdout.readline, b''):
            l = line.decode('utf-8')
            sys.stdout.write(l)
        return "Process finished with exit code 0"

    delete_old()
    stdout = runMpp(32)
    save_statistics()
    save("mlmcExperimen270320/" + initLevels +"/" + str(eps) + "/")


def mlmcExperiment(epsilon, initLevels, initSampleAmount):
    def runMpp(kernels, *args, conf=""):
        runParameters = ["mpirun", "-np", str(kernels),
                         "MLMC-M++",
                         "epsilon=" + str(epsilon),
                         "initLevels=" + initLevels,
                         "initSampleAmount=" + initSampleAmount
                         ]
        for arg in args:
            runParameters.append(arg)

        stdout = subprocess.Popen(runParameters, stdout=subprocess.PIPE, cwd=working_dir)
        for line in iter(stdout.stdout.readline, b''):
            l = line.decode('utf-8')
            sys.stdout.write(l)
        return "Process finished with exit code 0"

    delete_old()
    stdout = runMpp(32)
    save_statistics()
    save("mlmcExperiment/" + str(epsilon) + "/")


def mlmcmeshes(working_dir, levels, sample_amount):
    try:
        os.mkdir(working_dir+"mlmc/")
    except OSError:
        pass
    mlmcflux = MlmcVectorMesh(working_dir=working_dir,
                              levels=levels, sample_amount=sample_amount, filename="flux.vtk")
    mlmcflux.mesh.save(mlmcflux.wd + "mlmc/" + "flux.vtk")
    mlmcsol = MlmcMultiScalarMesh(working_dir=working_dir,
                                  levels=levels, sample_amount=sample_amount, filename="U.")
    mlmcsol.mesh.save(mlmcsol.wd + "mlmc/" + "U.")


if __name__ == "__main__":
    for eps in [0.01,0.005,0.003,0.001,0.0005]:
        mlmcExperiment1(eps,"4,5,6","8,4,2")

        
