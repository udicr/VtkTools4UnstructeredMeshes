from distutils.dir_util import copy_tree
import os
import sys
import re
import shutil
import matplotlib.pyplot as plt


def save(path="neu", working_dir="..", in_results=True):
    """
    saves content in (results)/path
    :param path:
    :param working_dir:
    :param in results: in results (default True)
    :return:
    """
    try:
        if in_results:
            os.mkdir(working_dir + "/results/" + path)
        else:
            os.mkdir(working_dir + "/" + path)
        print("Created Folder")
    except OSError as e:
        pass
    if in_results:
        toDirectory = working_dir + "/results/" + path
    else:
        toDirectory = working_dir + "/" + path
    fromDirectory1 = working_dir + "/" + "build/data"
    fromDirectory2 = working_dir + "/" + "build/log"
    copy_tree(fromDirectory1, toDirectory)
    copy_tree(fromDirectory2, toDirectory)


def delete_old():
    """
    deletes old vtk and log folders
    :return:
    """
    folder = '../build/data/vtk'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # subdirs
        except Exception as e:
            print(e)
    folder = '../build/log'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path) #subdirs
        except Exception as e:
            print(e)


def parse_output_allg(paramlist, output=None, logfile="../build/log/log"):
    """
    :param paramlist:
    :param output: optional for direct input
    :param logfile: read from logfile if output is None
    :return:list of list of outputvalues
    """
    out = []
    if output is None:
        # logfile = "mpp/build/log/log"
        with open(logfile) as file:
            lines = file.readlines()
    else:
        lines = output
    for param in paramlist:
        out.append([])
        for line in lines:
            if "Step" in line and not "default" in line and not "reading" in line:
                regex = r"[+-]?[0-9]+[.]?[0-9]*[eE]?[+-]?[0-9]*"
                sparam = " " + param
                if param in line:
                    tmp = line.split(sparam)[1]
                    value = float(re.findall(regex, tmp)[0])
                    # print(value)
                else:
                    value = 0
                out[paramlist.index(param)].append(value)
    return out


def mlmc_massplot():
    """
    parse through the Output Data and read all lines with substring "Step(i)"
    combines results MLMC like and plots mass values
    :return:
    """
    out = parse_output_allg(["t(", "Energy", "Mass", "InFlowRate", "OutFlowRate"])
    t = out[0]
    mass = out[2]
    inflow = out[3]
    dt_3 = 64
    ndt = [dt_3, 2 * dt_3, 4 * dt_3]
    l = [3, 4, 5]
    nl = [10, 0, 0]
    n = 0
    colors = ["darkred", "red", "#ff9966"]
    mconly = True

    for i in range(len(l)):
        fig = plt.figure(i)
        for j in range(nl[i]):
            plt.plot(t[n:n + ndt[i] + 1], mass[n:n + ndt[i] + 1], label="Level " + str(l[i]) + " Sample " + str(j))
            if (j == -1):
                plt.plot(t[n:n + ndt[i] + 1], inflow[n:n + ndt[i] + 1], color="black",
                         label="Inflow Level " + str(l[i]) + " Sample " + str(j))
            n += ndt[i] + 1
            if i > 0 and not mconly:
                plt.plot(t[n:n + ndt[i - 1] + 1], mass[n:n + ndt[i - 1] + 1],
                         label="Level " + str(l[i - 1]) + " Sample " + str(j))
                n += ndt[i - 1] + 1
            plt.legend()

    plt.savefig("MLMCMassPlot.jpg")
