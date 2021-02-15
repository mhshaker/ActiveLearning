# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:58:11 2020

@author: nguye
"""
#%%
import scipy.optimize as optimize
import numpy as np
from scipy.special import expit
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import UncertaintyM as unc
# import bigfloat
#%%
# def logistic_regression(betas, X, Y):
#     return - np.sum(Y*(X @ betas)) + np.sum(np.log(1.0 + np.exp(X @ betas))) + 0.5*sum([i*j for (i, j) in zip(betas, betas)])

# def logistic_regression_deriv(betas, X, Y):  
#     grad = np.dot(expit(X @ betas) - Y, X)
#     grad = [grad[ind] + betas[ind] for ind in range(len(betas))] 
#     return np.asarray(grad)

# def logistic_regression_Hess(betas, X, Y):
#     pi = expit(X @ betas)
#     dia = [pi_ele * (1- pi_ele) for pi_ele in pi]
#     Hess = (X.T @ np.diag(dia)) @ X
#     diaAdd = [1 for ind in range(len(betas))]
#     return np.asarray(np.add(Hess, np.diag(diaAdd)))
# #%%
# def findMLE(X, Y):
#     i = 0
#     lll_list = []
#     x_list = []
#     while (i<100):
#         result = optimize.minimize(logistic_regression,  np.random.uniform(0, 1, len(X[0])), method='Newton-CG', jac=logistic_regression_deriv,
#                     hess=logistic_regression_Hess, args=(X, Y), options={'disp': False})
#         i +=1
#         if (result.success):
#             lll = - logistic_regression(result.x, X, Y)     
#             return [result.x,lll, result.success]
#         lll_list.append(- logistic_regression(result.x, X, Y))
#         x_list.append(result.x)
#         ind_max = lll_list.index(max(lll_list))
#     print("Error when finding MLL ^_^")
#     print("Randomly repeat 100 times and chose results correspond to time with the highest MLL!")
#     return [x_list[ind_max], lll_list[ind_max], result.success]



########### regularization without intercept

def logistic_regression(betas, X, Y):
    return - np.sum(Y*(X @ betas)) + np.sum(np.log(1.0 + np.exp(X @ betas))) + 0.5*sum([betas[i]**2 for i in range(len(betas)) if i > 0])

def logistic_regression_deriv(betas, X, Y):  
    grad = np.dot(expit(X @ betas) - Y, X)
    grad = [grad[ind] + betas[ind] if ind > 0 else grad[ind] for ind in range(len(betas))] 
    return np.asarray(grad)

def logistic_regression_Hess(betas, X, Y):
    pi = expit(X @ betas)
    dia = [pi_ele * (1- pi_ele) for pi_ele in pi]
    Hess = (X.T @ np.diag(dia)) @ X
    diaAdd = [1 if ind > 0 else 0 for ind in range(len(betas))]
    return np.asarray(np.add(Hess, np.diag(diaAdd)))

def findMLE(X, Y):
    i = 0
    lll_list = []
    x_list = []
    while (i<100):
        result = optimize.minimize(logistic_regression,  np.random.uniform(0, 1, len(X[0])), method='Newton-CG', jac=logistic_regression_deriv,
                    hess=logistic_regression_Hess, args=(X, Y), options={'disp': False})
        i +=1
        if (result.success):
            lll = - logistic_regression(result.x, X, Y)     
            return [result.x,lll, result.success]
        lll_list.append(- logistic_regression(result.x, X, Y))
        x_list.append(result.x)
        ind_max = lll_list.index(max(lll_list))
    print("Error when finding MLL ^_^")
    print("Randomly repeat 100 times and chose results correspond to time with the highest MLL!")
    return [x_list[ind_max], lll_list[ind_max], result.success]



#%%
def logistic_prediction(X, betas):  
    return expit(X @ betas) 
#%%
#'maxiter': 2000,
def optimizePos(X, Y, x_pool, alpha):   
    cons = ({'type': 'eq',
          'fun' : lambda beta: np.dot(x_pool, beta) - np.log(alpha/(1-alpha)),
          'jac' : lambda beta: x_pool})
    result = optimize.minimize(logistic_regression, np.concatenate([[np.log(alpha/(1-alpha))], [0.0]*(len(x_pool)-1)]), 
             args=(X,Y), jac=logistic_regression_deriv, constraints=cons, method='SLSQP', options={'disp': False})
    if (result.success):
        lll = - logistic_regression(result.x, X, Y) 
        return [result.x, lll, result.success]
    print("Error when optimizing the support for positive class ^_^")
    return [math.nan, math.nan, False]
#%%
def optimizeNeg(X, Y, x_pool, alpha):
    cons = ({'type': 'eq',
          'fun' : lambda beta: np.dot(x_pool, beta) - np.log(alpha/(1-alpha)),
          'jac' : lambda beta: x_pool})
    result = optimize.minimize(logistic_regression, np.concatenate([[np.log(alpha/(1-alpha))], [0.0]*(len(x_pool)-1)]), 
             args=(X, Y), jac=logistic_regression_deriv, constraints=cons, method='SLSQP', options={'disp': False})
    if (result.success):
        lll = - logistic_regression(result.x, X, Y) 
        return [result.x, lll, result.success]
    print("Error when optimizing the support for negative class ^_^")
    return [math.nan, math.nan, False]
###########################################################################################################################################
import time

def lr_function2_piP(a, piP, X_train, Y_train, x_pool, Mll):
    if 2*a-1 > piP:
        [betas,lll,success] = optimizePos(X_train, Y_train, x_pool, a)
        if (success):
            piP = max(piP,min(np.exp(lll - Mll),2* a -1))
    return piP

def lr_function2_piN(a, piN, X_train, Y_train, x_pool, Mll):
    if 1-2*a > piN:
        [betas, lll, success] = optimizeNeg(X_train, Y_train, x_pool, a)
        if (success):
            piN = max(piN,min(np.exp(lll - Mll),1 - 2* a))
    return piN

def lr_function(x_pool, X_train, Y_train, betasMll, alphas, Mll):
    piP = max(2*expit(np.dot(betasMll, x_pool))-1,0)
    piN = max(1- 2*expit(np.dot(betasMll, x_pool)),0)
    alphasP = [alpha for alpha in alphas if (alpha >= 0.5) and (2*alpha-1 > piP)]
    alphasN = [alpha for alpha in alphas  if (alpha <= 0.5) and (1-2*alpha > piN)]
    for ind in range(len(alphasP)):
        if 2*alphasP[-(ind+1)]-1 > piP:
            [betas,lll,success] = optimizePos(X_train, Y_train, x_pool, alphasP[-(ind+1)])
            if (success):
                piP = max(piP,min(np.exp(lll - Mll),2* alphasP[-(ind+1)] -1))
    for ind in range(len(alphasN)):
        if 1-2*alphasN[ind] > piN:
            [betas, lll, success] = optimizeNeg(X_train, Y_train, x_pool, alphasN[ind])
            if (success):
                piN = max(piN,min(np.exp(lll - Mll),1 - 2* alphasN[ind]))
    rl_e = min(piP,piN)
    rl_a = 1- max(piP,piN)
    return rl_e, rl_a

def LR_run(X_train, X_pool, Y_train, Y_pool, prams, unc_method, seed, x_test, y_test):

    # adding feauter 1 for LR
    # X_train = np.append(np.ones((len(X_train), 1)), X_train, axis=1)
    # X_pool = np.append(np.ones((len(X_pool), 1)), X_pool, axis=1)
    # x_test = np.append(np.ones((len(x_test), 1)), x_test, axis=1)
    
    rl_e = np.zeros(len(X_pool))
    rl_a = np.zeros(len(X_pool))
    [betasMll, Mll, success] = findMLE(X_train, Y_train)
    Y_Pool_prob =  logistic_prediction(X_pool, betasMll)
    prediction = np.where(Y_Pool_prob > 0.5, 1, 0)

    model = LR_model(betasMll)
    
    pos_prob = Y_Pool_prob
    neg_prob = 1 - pos_prob
    pos_prob = np.reshape(pos_prob, (-1,1))
    neg_prob = np.reshape(neg_prob, (-1,1))
    porb_matrix = np.append(pos_prob, neg_prob, axis=1)



    if   "ent" in unc_method:
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_standard(np.array(porb_matrix))
    elif "rl" in unc_method:

        # print(["accuracy on the pool set is:", accuracy_score(y_test, prediction)])
        alphas_long = np.linspace(0, 1, prams["LR_value"])
        alphas = np.delete(alphas_long, [0, len(alphas_long)-1])

        # stime = time.time()
        # speed up vertion but it is not faster
        # rl_ea = np.apply_along_axis(lr_function, 1, X_pool, X_train, Y_train, betasMll, alphas, Mll)
        # total_uncertainty = rl_ea[:,0] + rl_ea[:,1]
        # epistemic_uncertainty = rl_ea[:,0]
        # aleatoric_uncertainty = rl_ea[:,1]

        # print("apply time ", time.time() - stime)
        # stime = time.time()
        for indp in range(len(X_pool)):
            x_pool = X_pool[indp]
            piP = max(2*expit(np.dot(betasMll, x_pool))-1,0)
            piN = max(1- 2*expit(np.dot(betasMll, x_pool)),0)
            alphasP = [alpha for alpha in alphas if (alpha >= 0.5) and (2*alpha-1 > piP)]
            alphasN = [alpha for alpha in alphas  if (alpha <= 0.5) and (1-2*alpha > piN)]
            for ind in range(len(alphasP)):
                if 2*alphasP[-(ind+1)]-1 > piP:
                    [betas,lll,success] = optimizePos(X_train, Y_train, x_pool, alphasP[-(ind+1)])
                    if (success):
                        piP = max(piP,min(np.exp(lll - Mll),2* alphasP[-(ind+1)] -1))
            for ind in range(len(alphasN)):
                if 1-2*alphasN[ind] > piN:
                    [betas, lll, success] = optimizeNeg(X_train, Y_train, x_pool, alphasN[ind])
                    if (success):
                        piN = max(piN,min(np.exp(lll - Mll),1 - 2* alphasN[ind]))
            rl_e[indp] = min(piP,piN)
            rl_a[indp] = 1- max(piP,piN) 

        # print("for time ", time.time() - stime)
        # print(aaa)
        total_uncertainty = rl_a + rl_e
        epistemic_uncertainty = rl_e
        aleatoric_uncertainty = rl_a

    elif "evid" in unc_method:

        prob_prediction_pool = logistic_prediction(X_pool, betasMll)
        ent_unc, _, _ = unc.uncertainty_ent_standard(np.array(porb_matrix))

        sorted_index = np.argsort(-ent_unc, kind='stable')

        top_uncertain_indices  = sorted_index[:5 * prams['batch_size']]
        evidence_scores = []
        number_features = np.shape(X_pool)[1]

        evid_size_data =  5 * prams['batch_size']
        if len(X_pool) <= evid_size_data:
            evid_size_data = len(X_pool)

        for index in range(evid_size_data):
            current_pool_index = top_uncertain_indices[index]
            current_pool_instance = X_pool[current_pool_index]
            scores = [current_pool_instance[i]*betasMll[i] for i in range(number_features)]
            scores = scores[1:]
            evidence_positive_class = sum([x for x in scores if x > 0])
            evidence_negative_class = sum([-x for x in scores if x < 0])
            evidence_scores.append(evidence_positive_class*evidence_negative_class)
        evid_sorted_index = np.argsort(evidence_scores, kind='stable')
        conf_unc = np.zeros(len(X_pool))
        insof_unc = np.zeros(len(X_pool))

        evid_res_size = prams['batch_size']
        if len(X_pool) <= evid_res_size:
            evid_res_size = len(X_pool)
        conflicting_index_pool = top_uncertain_indices[evid_sorted_index[-evid_res_size:]]
        insufficient_index_pool = top_uncertain_indices[evid_sorted_index[:evid_res_size]]
        conf_unc[conflicting_index_pool] = 1
        insof_unc[insufficient_index_pool] = 1

        aleatoric_uncertainty = conf_unc
        epistemic_uncertainty = insof_unc
        total_uncertainty = conf_unc + insof_unc

    elif "random" in unc_method:
        total_uncertainty = np.random.rand(len(X_pool))
        epistemic_uncertainty = np.random.rand(len(X_pool))
        aleatoric_uncertainty = np.random.rand(len(X_pool))    
    else:
        print(f"[Error] No implementation of {unc_method} for LR")

    # return prediction, t_unc_U, e_unc_U, a_unc_U, betasMll
    return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, model


# for fileName in ["parkinsons.csv"]:
#     seed = 42
#     prams = 100
#     XY = LoadCsvFile(fileName)
#     X = np.array([xy[:-1] for xy in XY])
#     Y = [xy[-1:][0] for xy in XY]
#     Y = np.array([0 if y < 0 else 1 for y in Y])
#     normalizer = preprocessing.Normalizer().fit(X) # see https://stats.stackexchange.com/questions/19523/need-for-centering-and-standardizing-data-in-regression
#     X = normalizer.transform(X)
#     X_train, X_pool, Y_train, Y_pool = train_test_split(X, Y, test_size=0.9, random_state=seed)
#     _ , t_unc_U, e_unc_U, a_unc_U, model = compute_Epistemic_Aleatoric_Entropy(X_train, X_pool, Y_train, Y_pool, prams, seed) # run model
#     print(["Epistemic uncertainty", e_unc_U])
#     print(["Aleatoric uncertainty", a_unc_U])




class LR_model(object):

    def __init__(self, betasMll):
        self.betasMll = betasMll

    def score(self, x_test, y_test):
        prob_prediction =  logistic_prediction(x_test, self.betasMll)
        prediction = np.where(prob_prediction > 0.5, 1, 0)
        return accuracy_score(y_test, prediction)
