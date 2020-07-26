#%%
from copy import deepcopy
import numpy as np # linear algebra
from matplotlib import pyplot as plt
import tensorflow as tf

def KMeans(data_in,k=3,n=10):

    data = data_in.take(n)

    #k Number of clusters
    #n Number of training data
    
    # Number of features in the data
    c = data.__iter__().next()[0].shape
    print('c: '+ c)
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    m = tf.keras.metrics.MeanTensor()
    for img in data:
        _ = m.update_state(img[0])
    mean = m.result()
    print('mean: '+mean)

    var = tf.zeros(data.__iter__().next()[0].shape)
    for img in data:
        var += (img[0] - mean)**2
    std = tf.sqrt(var)
    print('std: '+std)

    centers = tf.random.normal(([],data.__iter__().next()[0].shape),mean,std,seed=0)

    centers_old = tf.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # Store new centers

    clusters = tf.zeros(n)
    distances = tf.zeros((n,k))

    error = tf.linalg.norm(centers_new - centers_old)

    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        # Measure the distance to every center
        for i in range(k):
            j =  0
            for img in data:
                distances[j,i] = tf.linalg.norm(data - centers[i])
                j += 1
        # Assign all training data to closest center
        clusters = tf.math.argmin(distances, axis = 1)
        
        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            m = tf.keras.metrics.MeanTensor()
            for img in data:
                if clusters == i:
                    _ = m.update_state(img[0])
            centers_new[i] = m.result()
        error = tf.linalg.norm(centers_new - centers_old)
    return centers_new
# %%
