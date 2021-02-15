import ray
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import data_provider as dp
# import a_DF as df
# import a_eRF as erf
# import a_BN as bn
import a_Tree as tree
import a_LR as lr
import a_PW as pw
from ast import literal_eval
import mysql.connector as db
import math
from sklearn import preprocessing
from random import seed as rand_seed
from random import random
from tqdm import trange
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # export NUMEXPR_NUM_THREADS=6


rand_seed(1)

@ray.remote
def active_learning(seed, features, target, prams, mode, algo, dir, log=True):
    run_number = seed
    seed = seed ** 2
    os.environ['PYTHONHASHSEED']=str(seed)
    rand_seed(seed)
    np.random.seed(seed)

    s_time = time.time()
    # if algo == "LR": # add feature vector of one for the LR to fix the overflow in exp
    #     one_array = np.ones(len(features))
    #     one_array = np.reshape(one_array, (-1,1))
    #     features = np.append(one_array,features, axis=1)
    x_train_all, x_test, y_train_all, y_test = dp.split_data(features, target, split=prams["split"], seed=seed)
    
    # normalizer = preprocessing.MinMaxScaler().fit(x_train_all) # see https://stats.stackexchange.com/questions/19523/need-for-centering-and-standardizing-data-in-regression
    normalizer = preprocessing.StandardScaler().fit(x_train_all)
    x_train_all = normalizer.transform(x_train_all)
    x_test = normalizer.transform(x_test)

    if algo == "LR": # add feature vector of one for the LR to fix the overflow in exp
        one_array = np.ones(len(x_train_all))
        one_array = np.reshape(one_array, (-1,1))
        x_train_all = np.append(one_array,x_train_all, axis=1)
        one_array = np.ones(len(x_test))
        one_array = np.reshape(one_array, (-1,1))
        x_test = np.append(one_array,x_test, axis=1)
    if algo == "BN":
        y_train_all, y_test_BN, model = bn.BN_init(x_train_all, x_test, y_train_all, y_test, prams, mode, seed) # not changing the y_test because I need it to stay as is for acc calculation by hand in the a_BN.py
    initial_train_percent = 0.1
    cut_index = int(len(features) * initial_train_percent)    

    class_balance = random()

    all_class = False
    while all_class == False:
        indexes = np.array(range(len(x_train_all)))
        np.random.shuffle(indexes) # index becomes shuffled index
        x_train_all = x_train_all[indexes]
        y_train_all = y_train_all[indexes]

        x_train = x_train_all[:cut_index].copy()
        y_train = y_train_all[:cut_index].copy()
        x_U     = x_train_all[cut_index:].copy()
        y_U     = y_train_all[cut_index:].copy()

        if len(np.unique(y_train)) > 1:
            all_class = True

    acc_history = []

    unc_max_history = []
    unc_mean_history = []
    unc_std_history = []

    active_learning_steps = int(len(y_U) / prams["batch_size"])
    # print("steps ", active_learning_steps, " real value ", len(y_U) / prams["batch_size"])
    # print(len(y_train) + len(y_U))
    sorted_index = 0

    for active_index in range(active_learning_steps+2):
        # print("[debug] main active learning loop > pool size ", len(x_U))
        if algo == "DF":
            # mode_df = mode.split('_')[0] # spliting the active selection mode (_a _e _t) from the unc method because DF dose not work with that
            _ , t_unc_U, e_unc_U, a_unc_U, model = df.DF_run(x_train, x_U, y_train, y_U, prams, mode, seed, False) # run model
        elif algo == "eRF":
            # mode_df = mode.split('_')[0] # spliting the active selection mode (_a _e _t) from the unc method because DF dose not work with that
            _ , t_unc_U, e_unc_U, a_unc_U, model = erf.eRF_run(x_train, x_U, y_train, y_U, prams, mode, seed, False) # run model
        elif algo == "BN":
            # mode_df = mode.split('_')[0] # spliting the active selection mode (_a _e _t) from the unc method because DF dose not work with that
            acc , t_unc_U, e_unc_U, a_unc_U, model = bn.BN_run(x_train, x_U, y_train, y_U, x_test, y_test, prams, mode, seed, model, active_step=active_index) # run model
        elif algo == "Tree":
            _ , t_unc_U, e_unc_U, a_unc_U, model = tree.Tree_run(x_train, x_U, y_train, y_U, prams, mode, seed) # run model
        elif algo == "PW":
            if(active_index == 0):
                k = int(math.sqrt(len(x_train_all)))
                # if k > 20:
                # k = 20
                prams['PW_value'] = pw.learn_window_width(k, list(x_train_all))
                model = pw.PWC_model(x_train, y_train, prams['PW_value'])
            _ , t_unc_U, e_unc_U, a_unc_U, model = pw.PW_run(x_train, x_U, y_train, y_U, prams, mode, seed, x_test, y_test, model, active_index, sorted_index) # run model
        elif algo == "LR":
            if(active_index == 0):
                prams['LR_value'] = 100
            _ , t_unc_U, e_unc_U, a_unc_U, model = lr.LR_run(x_train, x_U, y_train, y_U, prams, mode, seed, x_test, y_test) # run model
        else:
            print("[ERORR] Undefined Algo")
            exit()

        # if algo == "BN":
        #     # acc_list = [] # multiple runs and average of the acc
        #     # for _ in range(prams["MC_samples"]):
        #     #     acc_list.append(model.evaluate(x_test, y_test_BN, verbose=0)[1])
        #     # acc_list = np.array(acc_list)
        #     # print(">>> debug acc list ",acc_list)
        #     # print(sdlfe)
        #     # acc = np.mean(acc_list)

        #     acc = model.evaluate(x_test, y_test_BN, verbose=0)[1] # single acc 
        #     # print(f" {active_index}/{active_learning_steps+2} >>>>>> {acc}")
        if algo != "BN":
            acc = model.score(x_test, y_test) # get test acc
            
        # print(acc)
        acc_history.append(acc) # append to history

        # if run_number == 7:
        #     print(f">>> acc {acc} | train len {len(x_train)} | pool len {len(x_U)} | batch size ", prams["batch_size"])
        #     print(len(y_train) + len(y_U))
        if len(x_U) <= 0: # break out of the active learning if there is no more data in the pool
            unc_max_history.append(0)
            unc_mean_history.append(0)
            unc_std_history.append(0)
            break

        if   "_e" in mode:
            sorted_index = np.argsort(-e_unc_U, kind='stable') # sort x_U based on epistemic uncertainty
        elif "_a" in mode:
            sorted_index = np.argsort(-a_unc_U, kind='stable') # sort x_U based on aleatoric uncertainty
        else: # total in the case of entropy and credal and also the random method
            sorted_index = np.argsort(-t_unc_U, kind='stable') # sort x_U based on total uncertainty

        x_U     = x_U[sorted_index]
        y_U     = y_U[sorted_index]

        if log:
            if   "_e" in mode:
                unc_log = e_unc_U #[sorted_index]
            elif "_a" in mode:
                unc_log = a_unc_U #[sorted_index]
            else:
                unc_log = t_unc_U #[sorted_index]
            
            unc_max_history.append(unc_log.max())
            unc_mean_history.append(unc_log.mean())
            unc_std_history.append(unc_log.std())
            # print(f"active {mode} step[{active_index}] Selected_index={unc_log.argmax()}", f" | unc: max {unc_log.max()} min {unc_log.min()} mean {unc_log.mean()} std {unc_log.std()}")

        if algo == "BN":
            # x_train = x_U[:prams["batch_size"]] # only batch
            # y_train = y_U[:prams["batch_size"]]
            x_train = np.append(x_train,x_U[:prams["batch_size"]], axis=0) # all training
            y_train = np.append(y_train,y_U[:prams["batch_size"]], axis=0)
        else:
            x_train = np.append(x_train,x_U[:prams["batch_size"]], axis=0) # add new high epistemic data point to the training data
            y_train = np.append(y_train,y_U[:prams["batch_size"]])

        x_U = x_U[prams["batch_size"]:] # remove that data point from x_U and y_U
        y_U = y_U[prams["batch_size"]:]
        if prams["batch_size"] > len(x_U): # fix last batch size
            prams["batch_size"] = len(x_U)
    
    # if log:
    #     unc_mean_history = np.array(unc_mean_history)
    #     unc_std_history = np.array(unc_std_history)
    #     unc_max_history = np.array(unc_max_history)
    #     steps = np.array(range(len(unc_mean_history)))

    #     plt.plot(steps, unc_mean_history,label="mean")
    #     plt.fill_between(steps, unc_mean_history - unc_std_history, unc_mean_history + unc_std_history, alpha=0.2)
    #     plt.plot(steps, unc_max_history,label="Max")
    #     plt.title(mode)
    #     plt.legend()
    #     plt.savefig(f"pic/sampling/run_unc_values/active_unc_value.png")
    #     plt.close()

    e_time = time.time()
    run_time = int(e_time - s_time)
    print(f"{run_number} :{run_time}s")
    return acc_history, unc_mean_history, unc_std_history, unc_max_history



if __name__ == '__main__':
    # prameter init default
    data_name = "Jdata/spambase"
    mode = "ent"
    algo = "PW"
    prams = {
    # 'criterion'        : "entropy",
    'max_depth'        : 5,
    # 'min_samples_leaf' : 0,
    # 'n_estimators'     : 20,

    'dropconnect_prob' : 0.2,
    'epochs'           : 1,
    'init_epochs'      : 10,
    'MC_samples'       : 5,

    'laplace_smoothing': 0,
    'split'            : 0.1,
    'batch_size'       : 1,
    'run_start'        : 0,
    }
    job_id = 0 # for developement
    seed   = 0
    runs   = 1



    base_dir = os.path.dirname(os.path.realpath(__file__))
    dir = f"{base_dir[:-12]}/Database/DB_files/job_{job_id}"

    # get input from command line
    if len(sys.argv) > 1:
        job_id = int(sys.argv[1])
        mydb = db.connect(host="131.234.250.119", user="noctua", passwd="uncertainty", database="uncertainty")
        mycursor = mydb.cursor()
        mycursor.execute(f"SELECT dataset, prams, result_type, results, algo, runs FROM experiments Where id ={job_id}")
        results = mycursor.fetchone()
        data_name = results[0]
        prams = literal_eval(results[1])
        mode = results[2]
        algo = results[4]
        runs = results[5]
        dir = f"{base_dir[:-12]}/Database/DB_files/job_{job_id}"
        mycursor.execute(f"UPDATE experiments SET results='{dir}' Where id={job_id}")
        mydb.commit()
        mycursor.execute(f"UPDATE experiments SET status='running' Where id={job_id}")
        mydb.commit()
    
    # check for directories
    if not os.path.exists(dir):
        os.makedirs(dir+"/acc")
        os.makedirs(dir+"/unc_mean")
        os.makedirs(dir+"/unc_std")
        os.makedirs(dir+"/unc_max")
        

    # get data
    features, target = dp.load_data(data_name)

    # Dimensionality reduction
    # if prams["PCA"] != 0:
    #     pca = PCA(n_components=prams["PCA"])
    #     features = pca.fit(features).transform(features)

    # _, counts = np.unique(target ,return_counts=True)
    # max_class = counts.max()
    # min_class = counts.min()
    # baseline = max_class / (max_class + min_class)
    # print(baseline)
    # exit()

    ray.init()
    ray_array = []
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        file_list.extend(filenames)
    start = 0
    if len(sys.argv) > 1:
        start = len(file_list)
        print("num files = ", start)
        if start == runs*4:
            start = 0
    if start == 0:
        len_u = len(features) * (1- (prams["split"] * 2)) # hard coding
        prams["batch_size"] = int(len_u / (100 / prams["batch_size"]))  # Commented means that batch size is now number and not the percentage
        if prams["batch_size"] < 1:
            prams["batch_size"] = 1
    if prams["run_start"] != 0:
        start = prams["run_start"]
    print(f"job_id {job_id} start")
    print(">>> start runs: ",start)
    # print("batch size ", prams["batch_size"])
    for seed in range(start,runs+start):
        ray_array.append(active_learning.remote(seed, features, target, prams, mode, algo, dir))
    res_array = ray.get(ray_array)
    res_array = np.array(res_array)
    for index, res in enumerate(res_array):
        np.savetxt(f"{dir}/acc/{start+index}.txt", res[0])
        np.savetxt(f"{dir}/unc_mean/{start+index}.txt", res[1])
        np.savetxt(f"{dir}/unc_std/{start+index}.txt", res[2])
        np.savetxt(f"{dir}/unc_max/{start+index}.txt", res[3])

    if len(sys.argv) > 1:
        mycursor.execute(f"UPDATE experiments SET status='done' Where id={job_id}")
        mydb.commit()
        mycursor.execute(f"UPDATE experiments SET prams=\"{str(prams)}\" Where id={job_id}")
        mydb.commit()
        # if prams["run_start"] != 0:
        #     os.system("bash /home/mhshaker/projects/uncertainty/bash/run_sampling.sh")