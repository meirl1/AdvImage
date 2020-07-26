# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to preprocess the image so that it can be inputted in MobileNetV2


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

# Helper function to extract labels from probability vector


def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]

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
    #parts = tf.strings.split(filename, os.sep)
    #label = parts[-2]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    # label prediction
    return image,'no_label'

#path to files
with open('rootpath.txt') as f:
    root_path = f.readline()

list_ds = tf.data.Dataset.list_files(root_path + '\\ILSVRC2012_img_val\\ILSVRC2012_val_000000??.JPEG')
images_ds = list_ds.map(parse_image)

# %%
from copy import deepcopy
n = 10
k = 3
data = images_ds.take(n)
#k Number of clusters
#n Number of training data
#c Number of features in the data
c = data.__iter__().next()[0].shape
print('c: {}'.format(c))
# Generate random centers, here we use sigma and mean to ensure it represent the whole data
m = tf.keras.metrics.MeanTensor()
for img in data:
    _ = m.update_state(img[0])
mean = m.result()
print('mean: {}'.format(mean.shape))

var = tf.zeros(data.__iter__().next()[0].shape)
for img in data:
    var += (img[0] - mean)**2
std = tf.sqrt(var)
print('std: {}'.format(std.shape))

centers = tf.random.normal([[k],data.__iter__().next()[0].shape],mean,std,seed=0)
print('centers: {}'.format(centers.shape))
centers_old = tf.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

clusters = tf.zeros(n)
distances = tf.zeros((n,k))

error = tf.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        print('loop: {}'.format(i))
        j =  0
        for img in data:
            print('img[]: {}'.format(img[0].shape))
            distances[j,i] = tf.linalg.norm(img[0] - centers[i])
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

print(centers_new)
#%%
labels_dict = {}
for file in list_ds:
    image_raw = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image_raw)
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)
    filstr = str(tf.strings.split(file, os.sep)[-1])
    labels_dict[filstr] = np.argmax(image_probs)

#%%
for file in list_ds:
    print(labels_dict[str(tf.strings.split(file, os.sep)[-1])])

# %%
img_dt = images_ds.take(5)
for image in img_dt:
    print(image)

#%%
img_dt.__iter__().next()[0].shape[0]

# %%
num_clusters = 10
num_iter = 100

tf.compat.v1.enable_eager_execution()
def input_fn():
    images_ds = list_ds.map(parse_image)
    dataset = images_ds.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    return iterator.gen_next()
    # return grad_ds.take(10).repeat(1)


kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False, seed=1)
previous_centers = None
for _ in range(num_iter):
    kmeans.train(input_fn=input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
        print('delta:', cluster_centers - previous_centers)
    previous_centers = cluster_centers
    print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)

cluster_indices = list(kmeans.predict_cluster_index(grad_ds.repeat(1)))
for i, point in enumerate(grad_ds):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    print('point:', point, 'is in cluster',
          cluster_index, 'centered at', center)


# %%
# %%
eps = 0.001  # Has to be a small value, otherwise might render some images unreconizable by humans
adv_success = 0
labels = []
grads = []


def gen():
    for i in itertools.count(1):
        # print('file:'+root_path+f'{i:08}.JPEG')
        image_raw = tf.io.read_file(root_path+'\\ILSVRC2012_img_val\\ILSVRC2012_val_'+f'{i:08}.JPEG')
        image = tf.image.decode_image(image_raw)
        if image.shape[2] == 1:
            image = tf.image.grayscale_to_rgb(image)
        image = preprocess(image)
        #image_probs = pretrained_model.predict(image)
        #label = tf.one_hot(np.argmax(image_probs), image_probs.shape[-1])
        #label = tf.reshape(label, (1, image_probs.shape[-1]))
        # yield get_gradient(image,label)
        yield (image, i)
# %%
images_ds2 = tf.data.Dataset.from_generator(
    gen, (tf.float32, tf.int32), (tf.TensorShape([1, 224, 224, 3]), tf.TensorShape([])))

# %%
i = 0
for img in images_ds2:
    print(img)
    i += 1
    if i >= 30:
        break

# %%
