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

data_list = ["parkinsons", "vertebral","ionosphere", "climate", "blod", "breast","bank", "QSAR"] #   ,"spambase", "madelon"
# data_list = ["anthracyclineTaxaneChemotherapy","AP_Breast_Uterus","Dorothea","gisette","OVA_Prostate"]
# data_list = ["vertebral"]
algo = "LR"
run_name = "Paper_results_main"
unc = "rl_e"
for data in data_list:

    # prameters ############################################################################################################################################

    plot_name   = f"{data}_{algo}_unc"
    # plot_name   = run_name
    query       = f"SELECT results, id, prams, result_type FROM experiments Where dataset='Jdata/{data}' AND algo='{algo}' AND run_name='{run_name}' AND result_type='{unc}'" # 
    # query       = f"SELECT results, id, prams, result_type FROM experiments Where id=3133"
    # query       = f"SELECT results, id, prams, result_type FROM experiments Where id='3085' or id='3127'"

    ########################################################################################################################################################


    max_run     = 1000
    xlabel      = "Number of queried instances"
    ylabel      = "Uncertainty"
    jobs = []

    # get data from database based on the query above
    mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
    mycursor = mydb.cursor()
    mycursor.execute(query)
    results = mycursor.fetchall()
    for job in results:
        jobs.append(job)

    fig, ax1 = plt.subplots()

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

   

        for text in job[3:]:
            legend += " " +str(text) 
        # print(legend)
        # exit()

        # get the list of file names for unc_mean
        dir_mean = dir + "/unc_mean" 
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(dir_mean):
            file_list.extend(filenames)
        all_runs = []
        run_count = 0
        for f in file_list:
            # print(f)
            run_count += 1
            run_result = np.loadtxt(dir_mean+"/"+f)
            all_runs.append(run_result)
            if run_count > max_run:
                break
        all_runs = np.array(all_runs)
        run_mean  = np.nanmean(all_runs, axis=0)

        # get the list of file names for unc_std
        dir_std = dir + "/unc_std" 
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(dir_std):
            file_list.extend(filenames)
        all_runs = []
        run_count = 0
        for f in file_list:
            # print(f)
            run_count += 1
            run_result = np.loadtxt(dir_std+"/"+f)
            all_runs.append(run_result)
            if run_count > max_run:
                break
        all_runs = np.array(all_runs)
        run_std  = np.nanmean(all_runs, axis=0)

        # get the list of file names for unc_max
        dir_max = dir + "/unc_max" 
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(dir_max):
            file_list.extend(filenames)
        all_runs = []
        run_count = 0
        for f in file_list:
            # print(f)
            run_count += 1
            run_result = np.loadtxt(dir_max+"/"+f)
            all_runs.append(run_result)
            if run_count > max_run:
                break
        all_runs = np.array(all_runs)
        run_max  = np.nanmean(all_runs, axis=0)

        # get the list of file names for acc
        dir_max = dir + "/acc" 
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(dir_max):
            file_list.extend(filenames)
        all_runs = []
        run_count = 0
        for f in file_list:
            # print(f)
            run_count += 1
            run_result = np.loadtxt(dir_max+"/"+f)
            all_runs.append(run_result)
            if run_count > max_run:
                break
        all_runs = np.array(all_runs)
        run_acc  = np.nanmean(all_runs, axis=0)
        run_acc = run_acc * 100 # to have percentates and not decimals

        legend = legend.replace("rl_e", "EU")
        legend = legend.replace("rl_a", "AU")
        legend = legend.replace("ent", "ENT")
        legend = legend.replace("credal", "CU")
        legend = legend.replace("random", "Rand")
        legend = legend.replace("evid_e", "IEU")
        legend = legend.replace("evid_a", "CEU")
        steps = np.array(range(len(run_mean))) * batch_size
        # steps = np.array(range(len(run_mean)))
        low_std = run_mean - run_std
        low_std = low_std.clip(min=0)
        color = 'k'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Accuracy %', color=color)
        ax1.plot(steps, run_acc, label= "Acc", color='k')
        ax1.legend(loc="center left")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # color = 'tab:orange'
        ax2.set_ylabel(ylabel, color=color)  # we already handled the x-label with ax1
        # ax2.plot(t, data2, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)

        ax2.plot(steps, run_mean, label= legend +" Mean")
        ax2.fill_between(steps, run_mean + run_std, low_std, alpha=0.2,label= legend +" std")
        ax2.plot(steps, run_max, label=legend + " Max")
        ax2.legend(loc="center right")


    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel) 
    # plt.title(plot_list)
    plt.legend(loc="center right")
    fig.savefig(f"{pic_dir}/{plot_name}.png")
    print(f"Plot {plot_name} Done")