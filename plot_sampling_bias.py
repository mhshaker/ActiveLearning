import sys
import mysql.connector as db
import os
import math
import numpy as np
import matplotlib.pyplot as plt

local = True

base_dir = os.path.dirname(os.path.realpath(__file__))
pic_dir = f"{base_dir}/pic/sampling"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

data_list = ["parkinsons", "vertebral", "spect","ionosphere", "climate", "blod", "breast", "diabetes","bank", "QSAR", "spambase"]
algo = "Tree"
run_name = "Paper_results_Bias"
# result_type = "random"
for data in data_list:

    # prameters ############################################################################################################################################

    plot_name   = f"{data}_AU_ENT"
    # plot_name   = "PCA_parkinson_epist"
    query1       = f"SELECT results, id, prams, result_type FROM experiments Where dataset='Jdata/{data}' AND algo='{algo}' AND result_type='ent' AND run_name='{run_name}'"
    query2       = f"SELECT results, id, prams, result_type FROM experiments Where dataset='Jdata/{data}' AND algo='{algo}' AND result_type='rl_a' AND run_name='{run_name}'"

    ########################################################################################################################################################


    plot_value1 = []
    legend_value1 = []
    max_run     = 1000
    xlabel      = "Number of queried instances"
    ylabel      = "Ratio"
    jobs = []

    # get data from database based on the query above
    mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
    mycursor = mydb.cursor()
    mycursor.execute(query1)
    results = mycursor.fetchall()
    for job in results:
        jobs.append(job)

    plot_list = []
    for job in jobs:
        dir = job[0]
        if dir[0] == ".":
            dir = base_dir + dir[1:]
        if local:
            dir = f"/home/mhshaker/Projects/Database/DB_files/job_{job[1]}"
            isFile = os.path.isdir(dir)
            if not isFile:
                print("[Error] file does not exist")
                print(dir)
                exit()

        plot_list.append(job[1])

        legend = ""

        prams = str(job[2])
        pram_name = "batch_size"
        search_pram = f"'{pram_name}': "
        v_index_s = prams.index(search_pram)
        v_index_e = prams.index("}", v_index_s)
        batch_size = int(prams[v_index_s+len(search_pram) : v_index_e])
        # legend += "Batch: " + str(batch_size)

        for text in job[3:]:
            legend += " " +str(text) 
        # print(legend)
        # exit()
        dir = dir + "/acc"
        # get the list of file names
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            file_list.extend(filenames)

        # read every file and append all to "all_runs"
        all_runs = []
        run_count = 0
        for f in file_list:
            # print(f)
            run_count += 1
            run_result = np.loadtxt(dir+"/"+f)
            all_runs.append(run_result)
            if run_count > max_run:
                break
        all_runs = np.array(all_runs)

        run_mean  = np.nanmean(all_runs, axis=0)
        std_error = np.std(all_runs, axis=0) / math.sqrt(len(all_runs))
        steps = np.array(range(len(run_mean))) * batch_size
        legend = legend.replace("rl_e", "EU")
        legend = legend.replace("rl_a", "AU")
        legend = legend.replace("ent", "ENT")
        legend = legend.replace("credal", "CU")
        legend = legend.replace("random", "Rand")
        legend = legend.replace("evid_e", "IEU")
        legend = legend.replace("evid_a", "CEU")
        plot_value1.append(run_mean)
        legend_value1.append(legend)
        # plt.plot(steps, run_mean, label=legend)
        # plt.fill_between(steps, run_mean - std_error, run_mean + std_error, alpha='0.2')


    max_run     = 1000
    xlabel      = "Number of queried instances"
    ylabel      = "Ratio"
    jobs = []

    # get data from database based on the query above
    mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
    mycursor = mydb.cursor()
    mycursor.execute(query2)
    results = mycursor.fetchall()
    for job in results:
        jobs.append(job)

    plot_list = []
    for job, plot_value, legend_value in zip(jobs, plot_value1, legend_value1):
        dir = job[0]
        if dir[0] == ".":
            dir = base_dir + dir[1:]
        if local:
            dir = f"/home/mhshaker/Projects/Database/DB_files/job_{job[1]}"
            isFile = os.path.isdir(dir)
            if not isFile:
                print("[Error] file does not exist")
                print(dir)
                exit()

        plot_list.append(job[1])

        legend = ""

        prams = str(job[2])
        pram_name = "batch_size"
        search_pram = f"'{pram_name}': "
        v_index_s = prams.index(search_pram)
        v_index_e = prams.index("}", v_index_s)
        batch_size = int(prams[v_index_s+len(search_pram) : v_index_e])
        # legend += "Batch: " + str(batch_size)

        prams = str(job[2])
        pram_name = "max_depth"
        search_pram = f"'{pram_name}': "
        v_index_s = prams.index(search_pram)
        v_index_e = prams.index(",", v_index_s)
        max_depth = int(prams[v_index_s+len(search_pram) : v_index_e])
        legend += "Depth: " + str(max_depth)


        for text in job[3:]:
            legend += " " +str(text) 
        # print(legend)
        # exit()
        dir = dir + "/acc"
        # get the list of file names
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            file_list.extend(filenames)

        # read every file and append all to "all_runs"
        all_runs = []
        run_count = 0
        for f in file_list:
            # print(f)
            run_count += 1
            run_result = np.loadtxt(dir+"/"+f)
            all_runs.append(run_result)
            if run_count > max_run:
                break
        all_runs = np.array(all_runs)

        run_mean  = np.nanmean(all_runs, axis=0)
        std_error = np.std(all_runs, axis=0) / math.sqrt(len(all_runs))
        steps = np.array(range(len(run_mean))) * batch_size
        legend = legend.replace("rl_e", "EU")
        legend = legend.replace("rl_a", "AU")
        legend = legend.replace("ent", "ENT")
        legend = legend.replace("credal", "CU")
        legend = legend.replace("random", "Rand")
        legend = legend.replace("evid_e", "IEU")
        legend = legend.replace("evid_a", "CEU")
        plt.plot(steps, run_mean/plot_value, label=legend + "/" + legend_value)
        # plt.fill_between(steps, run_mean - std_error, run_mean + std_error, alpha='0.2')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    # plt.title(plot_list)
    plt.legend()
    plt.savefig(f"{pic_dir}/{plot_name}.png")
    plt.close()
    print(f"Plot {plot_name} Done")