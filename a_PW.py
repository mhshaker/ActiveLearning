# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:58:11 2020

@author: nguye
"""
#%%
from scipy.optimize import minimize_scalar
from scipy.spatial import distance
import numpy as np
from scipy.special import expit
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import operator
import UncertaintyM as unc
#%%            
def euclidean_distance(x_1, x_2, number_features):
    return distance.euclidean(x_1, x_2)
    # return math.sqrt(sum([pow((x_1[i] - x_2[i]), 2) for i in range(number_features)]))
#%%
def learn_window_width(k, X):
    number_instances, number_features = np.shape(X)
    distances = []
    for ind1 in range(number_instances):
        distance = []
        for ind2 in range(ind1):
            distance.append(euclidean_distance(X[ind1], X[ind2], number_features))
        for ind2 in range(ind1+1, number_instances):
            distance.append(euclidean_distance(X[ind1], X[ind2], number_features))
        sortedDistance = sorted(distance) 
        distances.append(sortedDistance[k-1])
    return np.mean(distances)
#%%
def find_nearest_neighbors(x_pool, X, Y, prams):
    number_instances, number_features = np.shape(X)
    distances = [(Y[ind], euclidean_distance(x_pool, X[ind], number_features)) for ind in range(number_instances)]   
    distances.sort(key = operator.itemgetter(1))
    neighbor_labels = []
    dist =  distances[0][1]
    x = 0
    # print(f">>> dist {dist}")
    while dist <= prams:   
        neighbor_labels.append(distances[x][0])
        x +=1
        if (x >= len(distances)):
            break 
        else:
            dist =  distances[x][1]
    return neighbor_labels
#%%                
def determine_positive_total_neighbors(X_pool, X_train, Y_train, prams):   
    positive_neighbors = []
    total_neighbors = []
    for ind in range(np.shape(X_pool)[0]):
        neighbor_labels = find_nearest_neighbors(X_pool[ind], X_train, Y_train, prams)
        positive_neighbors.append(neighbor_labels.count(1))
        total_neighbors.append(len(neighbor_labels))
    return [total_neighbors, positive_neighbors]
#%%
def parzen_window_classifier(X_pool, X_train, Y_train, prams):
    total_neighbors, positive_neighbors = determine_positive_total_neighbors(X_pool, X_train, Y_train, prams)
    return np.array([positive_neighbors[ind]/total_neighbors[ind] if total_neighbors[ind] > 0 else 0.5 for ind in range(np.shape(X_pool)[0])])
#%%
def targetFunction(alpha, total_number, positive_number, classId):
    if classId == 1:
       highFunc = max(2*alpha -1,0)
    else:
       highFunc = max(1 -2*alpha,0) 
    negative_number = total_number - positive_number
    proportion = positive_number*(1/float(total_number))
    numerator = (alpha**positive_number)*((1-alpha)**negative_number)
    denominator = (proportion**positive_number)*((1-proportion)**negative_number)
    supportFunc = numerator*(1/float(denominator))
    TargetFunc = - min(supportFunc, highFunc)
    return TargetFunc
#%%
dictionary ={}    
def epistemicAndAleatoric(total_number, positive_number):
    global dictionary    
    key = "%i_%i"%(total_number, positive_number) 
    if (key in dictionary):
        return dictionary.get(key)        
    if total_number == 0:
        return [1,0]
    def Optp(alpha): return targetFunction(alpha, total_number, positive_number, 1)
    posSupPa =  minimize_scalar(Optp, bounds=(0, 1), method='bounded')
    def Optn(alpha): return targetFunction(alpha, total_number, positive_number, -1)
    negSupPa =  minimize_scalar(Optn, bounds=(0, 1), method='bounded')   
    dictionary[key] = [min(-posSupPa.fun, -negSupPa.fun), 1 - max(-posSupPa.fun, -negSupPa.fun)]
    return [min(-posSupPa.fun, -negSupPa.fun), 1 - max(-posSupPa.fun, -negSupPa.fun)]
#%%
def Interval_probability(total_number, positive_number):
    s = 1
    eps = .001
    valLow = (total_number - positive_number + s*eps*.5)/(positive_number+ s*(1-eps*.5))
    valUp = (total_number - positive_number + s*(1-eps*.5))/(positive_number+ s*.5)
    return [1/(1+valUp), 1/(1+valLow)]
#%%
def Credal_Uncertainty(total_number, positive_number):
    lower_probability, upper_probability = Interval_probability(total_number, positive_number) 
    return -max(lower_probability/(1-lower_probability),(1-upper_probability)/upper_probability) 

#%%
def LoadCsvFile(fileName):
# Load data file in csv file
    lines = csv.reader(open(fileName, "r"))
    dataSet = list(lines)
    for i in range(len(dataSet)):
        dataSet[i] = [float(x) for x in dataSet[i]]
    return dataSet

###########################################################################################################################################

def PW_run(X_train, X_pool, Y_train, Y_pool, prams, unc_method, seed, X_test, y_test, model, active_step, sorted_indices = 0):
    rl_e = np.zeros(len(X_pool))
    rl_a = np.zeros(len(X_pool))
    ent_t = np.zeros(len(X_pool))
    credal_t = np.zeros(len(X_pool))
        # compute information in the pool
    if active_step == 0:
        total_neighbors_pool, positive_neighbors_pool, total_neighbors_test, positive_neighbors_test = model.train(X_pool, X_test, X_train, Y_train, prams)
    else:
        total_neighbors_pool, positive_neighbors_pool, total_neighbors_test, positive_neighbors_test = model.update(X_pool, X_test, X_train, Y_train, prams, sorted_indices)        
#    positive_neighbors_pool = np.array(positive_neighbors_pool)
#    total_neighbors_pool = np.array(total_neighbors_pool)
    total_neighbors_pool = np.reshape(total_neighbors_pool, (-1,1))
    positive_neighbors_pool = np.reshape(positive_neighbors_pool, (-1,1))
    # pos_prob = positive_neighbors / total_neighbors
    pos_prob_pool = np.divide(positive_neighbors_pool, total_neighbors_pool, out=np.full(total_neighbors_pool.shape, 0.5), where=total_neighbors_pool!=0)
    neg_prob_pool = 1 - pos_prob_pool
    porb_matrix_pool = np.append(pos_prob_pool, neg_prob_pool, axis=1)
    if   "ent" in unc_method:
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_standard(np.array(porb_matrix_pool))
    elif "rl" in unc_method:
        count_matrix = np.stack((positive_neighbors_pool, total_neighbors_pool-positive_neighbors_pool), axis=1)
        count_matrix = np.reshape(count_matrix, (-1,1,2))
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_avg(count_matrix) 
    elif "credal" in unc_method:
        for indp in range(len(X_pool)):
            credal_t[indp] = Credal_Uncertainty(total_neighbors_pool[indp], positive_neighbors_pool[indp])
        total_uncertainty = credal_t
        epistemic_uncertainty = credal_t
        aleatoric_uncertainty = credal_t
    elif "random" in unc_method:
        total_uncertainty = np.random.rand(len(X_pool))
        epistemic_uncertainty = np.random.rand(len(X_pool))
        aleatoric_uncertainty = np.random.rand(len(X_pool))
    else:
        print(f"[Error] No implementation of {unc_method} for PW")
    prediction = 0
    return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, model # the prediction is empty - you can implement it later

###########################################################################################################################################
class PWC_model(object):
    pram = 0.1
    total_neighbors_pool = 0
    positive_neighbors_pool = 0
    total_neighbors_test = 0
    positive_neighbors_test = 0

    def __init__(self, x_train, y_train, prams):
        self.X_train = x_train
        self.Y_train = y_train
        self.pram = prams

    def find_nearest_neighbors(self, x_pool, X, Y, prams):
        # print(np.shape(X))
        number_instances, number_features = np.shape(X)
        distances = [(Y[ind], euclidean_distance(x_pool, X[ind], number_features)) for ind in range(number_instances)]   
        distances.sort(key = operator.itemgetter(1))
        neighbor_labels = []
        dist =  distances[0][1]
        x = 0
        # print(f">>> dist {dist}")
        while dist <= prams:   
            neighbor_labels.append(distances[x][0])
            x +=1
            if (x >= len(distances)):
                break 
            else:
                dist =  distances[x][1]
        return neighbor_labels

    def train(self, X_pool, X_test, X_train, Y_train, prams):   
        positive_neighbors_pool = []
        total_neighbors_pool = []
        for ind in range(np.shape(X_pool)[0]):
            neighbor_labels_pool = self.find_nearest_neighbors(X_pool[ind], X_train, Y_train, prams['PW_value'])
            positive_neighbors_pool.append(neighbor_labels_pool.count(1))
            total_neighbors_pool.append(len(neighbor_labels_pool))
        self.total_neighbors_pool = np.array(total_neighbors_pool)
        self.positive_neighbors_pool = np.array(positive_neighbors_pool)
        positive_neighbors_test = []
        total_neighbors_test = []
        for ind in range(np.shape(X_test)[0]):
            neighbor_labels_test = self.find_nearest_neighbors(X_test[ind], X_train, Y_train, prams['PW_value'])
            positive_neighbors_test.append(neighbor_labels_test.count(1))
            total_neighbors_test.append(len(neighbor_labels_test))
        self.total_neighbors_test = np.array(total_neighbors_test)
        self.positive_neighbors_test = np.array(positive_neighbors_test)
        return [self.total_neighbors_pool, self.positive_neighbors_pool, self.total_neighbors_test, self.positive_neighbors_test]


    def update(self, X_pool, X_test, X_train, Y_train, prams, sorted_indices):
        total_neighbors_pool = self.total_neighbors_pool[sorted_indices]
        positive_neighbors_pool = self.positive_neighbors_pool[sorted_indices]
        batch_size = len(sorted_indices) - len(X_pool)
        total_neighbors_pool    = total_neighbors_pool[batch_size:]
        positive_neighbors_pool = positive_neighbors_pool[batch_size:]
        number_instances_pool, number_features = np.shape(X_pool) 
    # update information in the pool
        X_batch = X_train[-batch_size:,:]
        Y_batch = Y_train[-batch_size:]
        for ind in range(number_instances_pool): 
            dists_update = [euclidean_distance(X_pool[ind], X_batch[i], number_features) for i in range(batch_size)]  
            index_update = [Y_batch[i] for i in range(batch_size) if dists_update[i] <= prams['PW_value']]   
            total_neighbors_pool[ind] += len(index_update)      
            positive_neighbors_pool[ind] +=  np.sum(index_update)
        self.total_neighbors_pool = np.array(total_neighbors_pool)
        self.positive_neighbors_pool = np.array(positive_neighbors_pool)

    # update information in the test set
        number_instances_test, number_features = np.shape(X_test)
        total_neighbors_test = self.total_neighbors_test
        positive_neighbors_test = self.positive_neighbors_test
        for ind in range(number_instances_test): 
            dists_update = [euclidean_distance(X_test[ind], X_batch[i], number_features) for i in range(batch_size)]  
            index_update = [Y_batch[i] for i in range(batch_size) if dists_update[i] <= prams['PW_value']]   
            total_neighbors_test[ind] += len(index_update)      
            positive_neighbors_test[ind] +=  np.sum(index_update)
        self.total_neighbors_test = np.array(total_neighbors_test)
        self.positive_neighbors_test = np.array(positive_neighbors_test)

        return [self.total_neighbors_pool, self.positive_neighbors_pool, self.total_neighbors_test, self.positive_neighbors_test]

    def score(self, x_test, y_test):
         total_neighbors = self.total_neighbors_test
         positive_neighbors = self.positive_neighbors_test
         prob_prediction = np.array([positive_neighbors[ind]/total_neighbors[ind] if total_neighbors[ind] > 0 else 3 for ind in range(np.shape(x_test)[0])])
         y_test_flip = 1 - y_test
         prob_prediction = np.where(prob_prediction == 3, y_test_flip, prob_prediction)
         
         binary_prediction = np.where(prob_prediction > 0.5, 1, 0)
         return accuracy_score(y_test, binary_prediction)   
