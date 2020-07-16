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

# %%
pretrained_model = tf.keras.applications.MobileNetV2(
    include_top=True, weights='imagenet')
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


# %%
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
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    # label prediction
    return image,'no_label'




# %%

list_ds = tf.data.Dataset.list_files(
    'Image folder')

# %%
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
images_ds = list_ds.map(parse_image)

# %%
for image in images_ds:
    print(image)


# %%
num_clusters = 10
num_iter = 100


def input_fn():
    return grad_ds.batch(10)
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
    print('score:', kmeans.score(grad_ds.repeat(1)))
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
    root_path = 'image folder\\ILSVRC2012_val_0000'
    for i in itertools.count(1):
        # print('file:'+root_path+f'{i:04}.JPEG')
        image_raw = tf.io.read_file(root_path+f'{i:04}.JPEG')
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
grad_ds = tf.data.Dataset.from_generator(
    gen, (tf.float32, tf.int32), (tf.TensorShape([1, 224, 224, 3]), tf.TensorShape([])))