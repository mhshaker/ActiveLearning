import sys
import mysql.connector as db
import os
import math
import numpy as np
import matplotlib.pyplot as plt

local = True
Calibarat_total = False

base_dir = os.path.dirname(os.path.realpath(__file__))
pic_dir = f"{base_dir}/pic/"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

# data_list = ["QSAR","parkinsons","bank","blod","breast","climate", "ionosphere", "vertebral"]
data_list = ["f_mnist"]
modes = "eat"
comparison = ["epist"]
algo = "DF"

# prameters ############################################################################################################################################

# query       = f"SELECT results, id, prams, result_type FROM experiments Where dataset='{data}' AND algo='{algo}' AND run_name='levi30_14fix'"
query = f"SELECT results, id, prams, result_type FROM experiments Where id=1355"

########################################################################################################################################################

# get data from database based on the query above
mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
mycursor = mydb.cursor()
mycursor.execute(query)
results = mycursor.fetchall()


def plot_unc(modes, comparison, name, calibrate=False):

    for data in data_list:
        max_run     = 1000
        xlabel      = "Percentage of synthetic data"
        ylabel      = "Uncertainty"
        jobs = []

        for job in results:
            jobs.append(job)

        plot_list = []
        for job in jobs:
            dir = job[0]
            if dir[0] == ".":
                dir = base_dir + dir[1:]
            if local:
                dir = f"/home/mhshaker/projects/Database/DB_files/job_{job[1]}"
                isFile = os.path.isdir(dir)
                if not isFile:
                    print("[Error] file does not exist")
                    exit()

            plot_list.append(job[1])

            legend = ""

            for text in job[3:]:
                legend += " " +str(text) 


            if calibrate:
                plot_data = []
                plot_comp = []
                mode = "t"
                for comp in comparison:
                    m_dir = dir + "/" + comp + "/" + mode
                    # get the list of file names
                    file_list = []
                    for (dirpath, dirnames, filenames) in os.walk(m_dir):
                        file_list.extend(filenames)

                    # read every file and append all to "all_runs"
                    all_runs = []
                    run_count = 0
                    for f in file_list:
                        # print(f)
                        run_count += 1
                        run_result = np.loadtxt(m_dir+"/"+f)
                        all_runs.append(run_result)
                        if run_count > max_run:
                            break
                    all_runs = np.array(all_runs)
                    run_mean  = np.nanmean(all_runs, axis=0)
                    plot_data.append(run_mean)
                    plot_comp.append(comp)
                
                calibrate_diff = plot_data[0] - plot_data[1]
                calibrator = plot_comp[0]
                if plot_data[1].sum() > plot_data[0].sum():
                    calibrate_diff = plot_data[1] - plot_data[0]
                    calibrator = plot_comp[1]

            for mode in modes:
                for comp in comparison:
                    m_dir = dir + "/" + comp + "/" + mode
                    # get the list of file names
                    file_list = []
                    for (dirpath, dirnames, filenames) in os.walk(m_dir):
                        file_list.extend(filenames)

                    # read every file and append all to "all_runs"
                    all_runs = []
                    run_count = 0
                    for f in file_list:
                        # print(f)
                        run_count += 1
                        run_result = np.loadtxt(m_dir+"/"+f)
                        all_runs.append(run_result)
                        if run_count > max_run:
                            break
                    all_runs = np.array(all_runs)
                    run_mean  = np.nanmean(all_runs, axis=0)
                    if calibrate == True and comp == calibrator:
                        run_mean = run_mean - calibrate_diff
                    std_error = np.std(all_runs, axis=0) / math.sqrt(len(all_runs))
                    steps = np.array(range(len(run_mean)))
                    plt.plot(steps, run_mean, label=legend + f" {mode} {comp}_test")
                    # plt.fill_between(steps, run_mean - std_error, run_mean + std_error, alpha='0.2')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel) 
        plt.title(plot_list)
        plt.legend()
        plot_name = f"{name}_{modes}"
        if calibrate:
            plot_name = "cal_" + plot_name
        plt.savefig(f"/home/mhshaker/projects/uncertainty/pic/{job[3]}_{plot_name}.png")
        plt.close()
        print(f"Plot {plot_name} Done")


prefix = ""
plot_unc("eat", ["ale"], f"{prefix}ale")
plot_unc("eat", ["epist"], f"{prefix}epist")
plot_unc("e", ["epist","ale"], f"{prefix}", Calibarat_total)
plot_unc("a", ["epist","ale"], f"{prefix}", Calibarat_total)
plot_unc("t", ["epist","ale"], f"{prefix}", Calibarat_total)
