import mysql.connector as db
import os

auto_run = False

# data_names     = ["Jdata/vertebral","Jdata/parkinsons","Jdata/ionosphere", "Jdata/climate", "Jdata/blod", "Jdata/breast","Jdata/bank", "Jdata/QSAR"] # , "Jdata/spambase", "Jdata/madelon"
data_names     = ["Jdata/spambase", "Jdata/madelon"] # "Jdata/hillvalley", "Jdata/scene"
algos          = ["PW"] # ,"LR"
modes          = ["rl_e","rl_a", "ent", "credal", "random"] #["ent_e","ent_a","ent_t", "random"]  # ent_e","ent_a","ent_t  "set14", "set15", "set14.convex", "set15.convex", "ent.levi"
# modes          = ["ent","evid_e", "evid_a","random"] 
task           = "sampling"
runs           = 100
prams = {
# 'criterion'        : "entropy", # gini
# 'max_depth'        : 2,
# 'min_samples_leaf' : 3,
# 'n_estimators'     : 10,
'PW_value'         : 20,
'laplace_smoothing': 0,
'split'            : 0.10,
# 'PCA'              : 0,
# 'credal_size'      : 20,
# 'dropconnect_prob' : 0.2,
# 'MC_samples'       : 20,
# 'epochs'           : 2,
# 'init_epochs'      : 10,
'run_start'        : 0,
'batch_size'       : 2,
}


for algo in algos:
    for data_name in data_names:
        for mode in modes:
            run_name       = "PW_k20" #f"{mode}_{algo}" + "noctua_test" # if you want a specific name give it here
            description    = "acc_hist"

            mydb = db.connect(host="131.234.250.119", user="noctua", passwd="uncertainty", database="uncertainty")
            mycursor = mydb.cursor()

            mycursor.execute("SELECT id FROM experiments ORDER BY id DESC LIMIT 1") #get last id in DB
            results = mycursor.fetchall()
            job_id = results[0][0] + 1 # set new id
            result_address = f"/home/mhshaker/projects/Database/DB_files/job_{job_id}" # set results address
            sqlFormula = "INSERT INTO experiments (id, task, run_name, algo, prams, dataset, runs, status, description, result_type, results) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            jobs = [(job_id, task, run_name, algo, str(prams), data_name, runs, "pending", description, mode, result_address)]
            mycursor.executemany(sqlFormula, jobs) # insert new job to DB
            mydb.commit() # save


if auto_run:
    if task == "unc":
        os.system("bash /home/mhshaker/projects/uncertainty/bash/run_unc.sh")
        os.system(f"python3 /home/mhshaker/projects/uncertainty/bash/plot_accrej.py auto_unc tea {job_id}")
    elif task == "sampling":
        os.system("bash /home/mhshaker/projects/uncertainty/bash/run_sampling.sh")
        os.system(f"python3 /home/mhshaker/projects/uncertainty/bash/plot_sampling.py auto_samp {job_id}")
