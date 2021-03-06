# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from copy import deepcopy
import itertools
import time
import os
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow_addons.image import filters
# import TFAddons.filters #This module includes the median filter and some others
# - cut from tensorflow addons as it doesn't work on windows yet"""
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(
    include_top=True, weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to preprocess the image so that it can be inputted in MobileNetV2


def preprocess(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image, label

# Helper function to extract labels from probability vector

loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def get_gradient(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    return gradient
    # Get the sign of the gradients to create the perturbation


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    # label prediction
    return image, label


def display_images(image, description):
    probs = pretrained_model.predict(image)
    _, label, confidence = decode_predictions(probs, top=1)[0][0]
    plt.figure()
    plt.imshow(image[0]*0.5+0.5)
    plt.title('{} \n {} : {:.2f}% Confidence'.format(np.argmax(probs), label, confidence*100))
    plt.show()


# path to files
with open('rootpath.txt') as f:
    root_path = f.readline()

list_ds = tf.data.Dataset.list_files(
    root_path + '\\ILSVRC2012_img_val\\ILSVRC2012_val_000000??.JPEG', shuffle=False)
data = list_ds.map(parse_image)
# %%
# Iterative Fast Gradient Sign method


def IFGS(image, d_class, eps=0.001):
    #print('Target deceiving class: {}'.format(np.argmax(d_class)))
    perturbations = create_adversarial_pattern(image, d_class)
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    adv_probs = pretrained_model.predict(adv_x)
    iterCount = 1
    # Keep looping until we get the aversarial label
    while np.argmax(adv_probs) != np.argmax(d_class):
        perturbations += create_adversarial_pattern(adv_x, d_class)
        adv_x = image - eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        adv_probs = pretrained_model.predict(adv_x)
        '''print('Attempt: {} image label: {}, Index: {}'.format(
            iterCount, decode_predictions(adv_probs, top=1)[0][0][1], np.argmax(adv_probs)))'''
        # displays progress
        iterCount += 1
    return perturbations, iterCount


def IAN_generation(image, d_class, eps=0.005, increment=0.005):
    image_probs = pretrained_model.predict(image)
    perturbations, iterCount = IFGS(image, d_class, eps)
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    # adv_probs = pretrained_model.predict(adv_x)
    filtered_x = filters.median_filter2d(adv_x, (5, 5))
    filtered_probs = pretrained_model.predict(filtered_x)
    while np.argmax(filtered_probs) == np.argmax(image_probs):
        eps += increment
        perturbations, j = IFGS(image, d_class, eps)
        iterCount += j
        adv_x = image - eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        # adv_probs = pretrained_model.predict(adv_x)
        filtered_x = filters.median_filter2d(adv_x, (5, 5))
        filtered_probs = pretrained_model.predict(filtered_x)
        print('eps: {},iterCount: {}, Filtered image label: {} Index: {}'.format(
            eps, iterCount, decode_predictions(filtered_probs, top=1)[0][0][1], np.argmax(filtered_probs)))
    return eps*perturbations, eps, iterCount
# %%
#prepare clusters

n = 99
k = 3
# k Number of clusters
# n Number of training data
# c Number of features in the data
# Generate random centers, here we use sigma and mean to ensure it represent the whole data
m = tf.keras.metrics.MeanTensor()
for img in data:
    # print(img[1],img[0].shape)
    _ = m.update_state(img[0])
mean = m.result()
print('mean: {}'.format(mean))

var = tf.zeros(data.__iter__().next()[0].shape)
for img in data:
    var += (img[0] - mean)**2
std = tf.sqrt(var/n)
print('std: {}'.format(std))
centers = []
centers_old = []
for i in range(k):
    centers.append(tf.random.normal(data.__iter__().next()[0].shape, mean, std, seed=0))
    centers_old.append(tf.zeros(data.__iter__().next()[0].shape))
# centers_old = tf.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers)  # Store new centers

clusters = np.zeros(n)
distances = np.zeros((n, k))
error = (tf.linalg.norm(centers_new[k-1] - centers_old[k-1]))
for i in range(k-1):
    print(i)
    error += (tf.linalg.norm(centers_new[i] - centers_old[i]))
error /= k
print('error: {}'.format(error))
# When, after an update, the estimate of that center stays the same, exit loop
while error > 0.0001:
    # Measure the distance to every center
    for i in range(k):
        j = 0
        for img in data:
            distances[j, i] = tf.linalg.norm(img[0] - centers_new[i])
            j += 1
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis=1)
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        m = tf.keras.metrics.MeanTensor()
        j = 0
        for img in data:
            if clusters[j] == i:
                _ = m.update_state(img[0])
            j += 1
        centers_new[i] = m.result()
    # error = tf.linalg.norm(centers_new[i] - centers_old[i])
    error = tf.linalg.norm(centers_new[k-1] - centers_old[k-1])
    for i in range(k-1):
        print(i)
        error += tf.linalg.norm(centers_new[i] - centers_old[i])
    error /= k
    print('error: {}'.format(error))
np.savetxt('clusters.csv',clusters,delimiter=',')
print(centers_new)
#%%
import pandas as pd
mn_data = list_ds.map(preprocess)
file_list = []
idx_list = []
for img in mn_data:
    print(img[1])
    file_list.append(str(img[1]).split('_')[2].split('.')[0])
    idx_list.append(np.argmax(pretrained_model.predict(img[0])))

df = pd.DataFrame(index=file_list,columns=idx_list)
df = df.loc[:,~df.columns.duplicated()]
# %%
files_1 = []
files_2 = []
files_3 = []
i = 0
for filename in list_ds:
    if clusters[i] == 0:
        files_1.append(filename)
    elif clusters[i] == 1:
        files_2.append(filename)
    else:
        files_3.append(filename)
    i += 1
# %%
adv_success = 0
filter_success = 0
results = []
row = 0
for img in mn_data.shuffle(100).take(10):
    image = img[0]
    image_probs = pretrained_model.predict(image)
    label = tf.one_hot(np.argmax(image_probs), image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))
    display_images(image, 'input')
    column = 0
    for dNumber in df.columns:
        if np.argmax(image_probs) != dNumber:
            d_class = tf.one_hot(dNumber, 1000)
            d_class = tf.reshape(d_class, (1, 1000))
            print('decieving class: ',dNumber)
            delta, eps, df.iloc[row,column] = IAN_generation(image, d_class)
            adv_x = image - delta
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            adv_probs = pretrained_model.predict(adv_x)
    # results.append((np.argmax(image_probs),dNumber,np.argmax(adv_probs),np.argmax(filtered_probs),iterCount,eps))
            #display_images(adv_x, 'input')
            if np.argmax(adv_probs) != np.argmax(image_probs):
                print('Success')
                adv_success += 1
            else:
                print('Failed')
        column += 1
    print('label: {}\nmean: {}'.format(np.argmax(image_probs),df.mean(axis=1)))
    for i in range(k):
        print('cluster {}\n\
            mean: {}'.format(np.array(idx_list)[clusters == i],df[np.array(idx_list)[clusters == i]].mean(axis=1)))
    filtered_x = filters.median_filter2d(adv_x, (5, 5))
    filtered_probs = pretrained_model.predict(filtered_x)
    #display_images(filtered_x, 'input')

    if np.argmax(filtered_probs) == np.argmax(image_probs):
        filter_success += 1
    row += 1
print('{} {}'.format(adv_success, filter_success))
df.to_csv('Kmeanscenters.csv')

# %%
print(clusters)
#check second high value in prediction results

# %%
