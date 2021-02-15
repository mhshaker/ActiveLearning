import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

def create(mode):
    if mode == "test_ale":
        x, y = make_classification(
                    class_sep=.3,
                    flip_y=0.1, 
                    n_samples=500, 
                    n_features=2,
                    n_informative=2, 
                    n_redundant=0, 
                    n_repeated=0, 
                    n_classes=2, 
                    n_clusters_per_class=2, 
                    weights=None, 
                    hypercube=True, 
                    shift=0.0, 
                    scale=1.0, 
                    shuffle=True, 
                    random_state=1)

        x_test = np.array([[-0.2, 0.2],[0.5, -0.5],[-0.5, 0.6],[1,2],[-1, -2]])
    elif mode == "test_epist":
        x, y = make_classification(
                    class_sep=3,
                    flip_y=0.1, 
                    n_samples=500, 
                    n_features=2,
                    n_informative=2, 
                    n_redundant=0, 
                    n_repeated=0, 
                    n_classes=2, 
                    n_clusters_per_class=2, 
                    weights=None, 
                    hypercube=True, 
                    shift=0.0, 
                    scale=1.0, 
                    shuffle=True, 
                    random_state=1)

        x_test = np.array([[-3, -3],[-0.25, 0.16],[2.5, 0.1],[1,2],[4,4]])

    elif mode == "test_total":
        x, y = make_classification(
                    class_sep=0.3,
                    flip_y=0.8, 
                    n_samples=500, 
                    n_features=2,
                    n_informative=2, 
                    n_redundant=0, 
                    n_repeated=0, 
                    n_classes=2, 
                    n_clusters_per_class=2, 
                    weights=None, 
                    hypercube=True, 
                    shift=0.0, 
                    scale=1.0, 
                    shuffle=True, 
                    random_state=1)

        x_test = np.array([[0, -1],[-0.25, 0.16],[-0.1, 0.1],[1,2],[-1, 0]])

    y_test = [0,1,0,1,1]
    plt.scatter(x[:,0], x[:,1], c= y, alpha=0.7)
    plt.scatter(x_test[:,0], x_test[:,1], c='red',linewidths=4)
    for i in range(x_test.shape[0]):
        plt.text(x_test[i,0], x_test[i,1], str(i+1))
    plt.title(mode)
    plt.savefig(f"./pic/test_dataset.png")
    plt.close()
    return x, x_test, y, y_test





def create2(seed):
    x, y = make_classification(
                n_classes=3, 
                class_sep=3,
                flip_y=0.0, 
                n_samples=5000, 
                n_features=2,
                n_informative=2, 
                n_redundant=0, 
                n_repeated=0, 
                n_clusters_per_class=1, 
                weights=None, 
                hypercube=True, 
                shift=0.0, 
                scale=1.0, 
                shuffle=True, 
                random_state=seed)

    x_test = []
    f1 = np.arange(-10,10,0.1).reshape(-1,1)
    f2 = np.arange(-10,10,0.1).reshape(-1,1)
    for element in f1:
        data = np.append(f2,np.full((len(f2),1),element), axis=1)
        x_test.append(data)
    x_test = np.array(x_test).reshape(-1,2)

    plt.scatter(x[:,0], x[:,1], c= y, alpha=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.title(mode)
    plt.savefig(f"./pic/dataset.png")
    plt.close()
    return x, y, x_test


def moon_data(seed):
    x, y = make_moons(n_samples=1000, shuffle=True, noise=0.3, random_state=seed)
    x = x * 4
    x_test = []
    f1 = np.arange(-10,10,0.1).reshape(-1,1)
    f2 = np.arange(-10,10,0.1).reshape(-1,1)
    for element in f1:
        data = np.append(f2,np.full((len(f2),1),element), axis=1)
        x_test.append(data)
    x_test = np.array(x_test).reshape(-1,2)

    plt.scatter(x[:,0], x[:,1], c= y, alpha=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.title(mode)
    plt.savefig(f"./pic/dataset.png")
    plt.close()
    return x, y, x_test

def circles_data(seed):
    x, y = make_circles(n_samples=1000, shuffle=True, noise=0.1, random_state=seed, factor=0.8)
    x = x * 4
    x_test = []
    f1 = np.arange(-10,10,0.1).reshape(-1,1)
    f2 = np.arange(-10,10,0.1).reshape(-1,1)
    for element in f1:
        data = np.append(f2,np.full((len(f2),1),element), axis=1)
        x_test.append(data)
    x_test = np.array(x_test).reshape(-1,2)

    plt.scatter(x[:,0], x[:,1], c= y, alpha=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.title(mode)
    plt.savefig(f"./pic/dataset.png")
    plt.close()
    return x, y, x_test


moon_data(1)
# circles_data(1)