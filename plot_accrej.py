import sys
import mysql.connector as db
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import UncertaintyM as unc


unc_value_plot = False

# prameters ############################################################################################################################################

run_name   = "likelihood_fix_3.2"
# data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "blod", "bank", "QSAR", "spambase", "diabetes", "spect"] 
data_list = ["spect"]
modes      = "aet"
# query_temp = f"SELECT id FROM experiments Where task='unc' AND run_name='{run_name}'"
query_temp = f"SELECT id FROM experiments Where task='unc' AND id=1181 or id=1186 or id=1247"

########################################################################################################################################################

file_extention = run_name
pic_dir = f"./pic/{file_extention}"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


for data in data_list:
    
    fig, axs = plt.subplots(1,len(modes))
    fig.set_figheight(5)
    fig.set_figwidth(15)

    legend_list = []
    flap = True

    for mode_index, mode in enumerate(modes):

        plot_name   = f"{data}" 
        query = query_temp + f" AND dataset='Jdata/{data}'"

        xlabel      = "Rejection %"
        ylabel      = "Accuracy %"

        # get input from command line
        mydb = db.connect(host="131.234.250.119", user="noctua", passwd="uncertainty", database="uncertainty")
        mycursor = mydb.cursor()
        mycursor.execute(query) # run_name ='uni_exp'
        results = list(mycursor.fetchall())
        plot_list = []
        for id in results:
            plot_list.append(id[0])
        print(plot_list)
        if len(sys.argv) > 1:
            plot_name = sys.argv[1]
        if len(sys.argv) > 2:
            mode = sys.argv[2]
        if len(sys.argv) > 3:
            plot_list = sys.argv[3:]
        jobs = []
        for job_id in plot_list:
            #                                                                                                           ############# customize based on what you need
            mycursor.execute(f"SELECT results, prams, result_type,algo FROM experiments Where id ={job_id}") 
            results = mycursor.fetchone()

            value = ""
            # prams = results[1]
            # pram_name = "n_estimators"
            # search_pram = f"'{pram_name}': "
            # v_index_s = prams.index(search_pram)
            # v_index_e = prams.index(",", v_index_s)
            # value = prams[v_index_s+len(search_pram) : v_index_e]

            # pram_name = "credal_size"
            # search_pram = f"'{pram_name}': "
            # v_index_s = prams.index(search_pram)
            # v_index_e = prams.index("}", v_index_s)
            # value2 = prams[v_index_s+len(search_pram) : v_index_e]
            # value += " " + value2

            job = [results[0], results[2], value]
            jobs.append(job)

        mode = [char for char in mode]
        # plt.figure(figsize=(15,5)) 
        # fig.figsize = [15,5]

        for job in jobs:
            dir = job[0]
            job = job[1:]
            for mode in mode:
                # print(f"mode {mode} dir {dir}")
                dir_p = dir + "/p"
                dir_l = dir + "/l"
                dir_mode = dir + "/" + mode

                legend = ""
                for text in job:
                    legend += text + " "
                # legend += mode   

                # get the list of file names
                file_list = []
                for (dirpath, dirnames, filenames) in os.walk(dir_mode):
                    file_list.extend(filenames)


                # read every file and append all to "all_runs"
                all_runs_unc = []
                for f in file_list:
                    run_result = np.loadtxt(dir_mode+"/"+f)
                    all_runs_unc.append(run_result)

                all_runs_p = []
                for f in file_list:
                    run_result = np.loadtxt(dir_p+"/"+f)
                    all_runs_p.append(run_result)

                all_runs_l = []
                for f in file_list:
                    run_result = np.loadtxt(dir_l+"/"+f)
                    all_runs_l.append(run_result)

                avg_acc, avg_min, avg_max, avg_random , steps = unc.accuracy_rejection(all_runs_p,all_runs_l,all_runs_unc, unc_value_plot)

                # print(">>>>>>>>", avg_acc)
                linestyle = '-'
                color = "black"
                # if "set" in legend:
                #     linestyle = '--'
                if "convex" in legend:
                    linestyle = ':'
                if "14" in legend:
                    color = "blue"
                if "15" in legend:
                    color = "red"
                if "gs" in legend:
                    color = "green"
                if "rl" in legend:
                    color = "orange"
                if "10" in legend or "levi" in legend:
                    color = "blue"
                if "100" in legend or "levi" in legend:
                    color = "orange"
                
                legend = legend.replace("set14.convex", "Levi-GH-conv")
                legend = legend.replace("set15.convex", "Levi-Ent-conv")
                legend = legend.replace("set14", "Levi-GH")
                legend = legend.replace("set15", "Levi-Ent")
                legend = legend.replace("ent", "Bayes")
                legend = legend.replace("gs", "GS")

                # if "convex" not in legend:
                # axs[mode_index].set_yscale('log')
                axs[mode_index].plot(steps, avg_acc, linestyle=linestyle, color=color)
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", legend)
                if flap:
                    legend_list.append(legend)
                    
                if mode == "a":
                    mode_title = "AU"
                if mode == "e":
                    mode_title = "EU"
                if mode == "t":
                    mode_title = "TU"
                axs[mode_index].set_title(mode_title)
                # axs[mode_index].flat.xlabel(xlabel)
                # axs[mode_index].flat.ylabel(ylabel)
                # print(">>>>>>>>> mode_index ",mode_index)
                # plt.fill_between(steps, avg_min, avg_max, alpha='0.2')
        # print(">>>>>> flap off <<<<<<<")
        flap =False

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    acc_lable_flag = True
    for ax in axs.flat:
        if acc_lable_flag:
            ax.set(xlabel=xlabel, ylabel=ylabel)
            acc_lable_flag = False
        else:
            ax.set(xlabel=xlabel)
    title = plot_name
    # title = title.replace("_a", " AU")
    # title = title.replace("_e", " EU")
    fig.suptitle(title)
    # plt.legend(ncol=5)
    # print(">>>>>>>>>>>>> ",legend_list)
    fig.legend(axs,     # The line objects
           labels=legend_list,   # The labels for each line
           loc="lower center",   # Position of legend
           ncol=6
           )
    fig.savefig(f"./pic/{file_extention}/{plot_name}_{modes}.png",bbox_inches='tight')
    # fig.close()
    print("Plot Done")