import os
import numpy as np
import UncertaintyM as unc
from sklearn.tree import DecisionTreeClassifier


def Tree_run(x_train, x_test, y_train, y_test, pram, unc_method, seed):
    np.random.seed(seed)
    model = DecisionTreeClassifier(
        # min_samples_leaf = 2,
        # criterion=pram['criterion'],
        max_depth=pram["max_depth"],
        random_state=seed)
    model.fit(x_train, y_train)

    if len(x_test) == 0:
        return 0, 0, 0, 0, model

    prediction = model.predict(x_test)
    if "ent" in unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_standard(np.array(porb_matrix))
    elif "rl" in unc_method:
        count_matrix = get_count_matrix(model, x_test, pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_avg(count_matrix)
    elif "credal" in unc_method:
        count_matrix = get_count_matrix(model, x_test, pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_credal_tree(count_matrix)
    elif "random" in unc_method:
        total_uncertainty = np.random.rand(len(x_test))
        epistemic_uncertainty = np.random.rand(len(x_test))
        aleatoric_uncertainty = np.random.rand(len(x_test))
    else:
        print(f"[Error] No implementation of {unc_method} for Tree")

    return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, model

def get_prob_matrix(model, x_test, laplace_smoothing, log=False):
    # populate the porb_matrix with the tree_prob
    tree_prob = model.predict_proba(x_test)
    if laplace_smoothing > 0:
        leaf_index_array = model.apply(x_test)
        for data_index, leaf_index in enumerate(leaf_index_array):
            leaf_values = model.tree_.value[leaf_index]
            leaf_samples = np.array(leaf_values).sum()
            for i,v in enumerate(leaf_values[0]):
                tree_prob[data_index][i] = (v + laplace_smoothing) / (leaf_samples + (len(leaf_values[0]) * laplace_smoothing))

    if log:
        print(f"----------------------------------------[get_prob_matrix - tree_prob]")
        print(tree_prob)

    return tree_prob

def get_count_matrix(model, x_test, laplace_smoothing=0):
    count_matrix = None #np.empty((len(x_test), n_estimators, 2))
    leaf_index_array = model.apply(x_test)
    tree_prob = model.tree_.value[leaf_index_array]
    tree_prob += laplace_smoothing 
    count_matrix = tree_prob.copy()
    return count_matrix.copy()
