# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tf_filters #This module includes the median filter and some others 
# - cut from tensorflow addons as it doesn't work on windows yet"""
import tensorflow_datasets as tfds
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False


# %%
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


# %%
import time
eps = 0.001 #Has to be a small value, otherwise might render some images unreconizable by humans
adv_success = 0
labels = []
grads = []
def gen():
    root_path = 'Enter here imagenet 2012 validation path \\ILSVRC2012_val_0000'
    for i in range(1,10):
        #print('file:'+root_path+f'{i:04}.JPEG')
        image_raw = tf.io.read_file(root_path+f'{i:04}.JPEG')
        image = tf.image.decode_image(image_raw)
        if image.shape[2] == 1:
            image = tf.image.grayscale_to_rgb(image)
        image = preprocess(image)
        image_probs = pretrained_model.predict(image)
        label = tf.one_hot(labels[i-1], image_probs.shape[-1])
    return get_gradient(image,label)


# %%
import time
root_path = 'Enter here imagenet 2012 validation path \\ILSVRC2012_val_0000'
eps = 0.001 #Has to be a small value, otherwise might render some images unreconizable by humans
adv_success = 0
labels = []
grads = []
start_time = time.time()
for i in range(3,4):
    #print('file:'+root_path+f'{i:04}.JPEG')
    image_raw = tf.io.read_file(root_path+f'{i:04}.JPEG')
    image = tf.image.decode_image(image_raw)
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)
    labels.append(np.argmax(image_probs))
    label = tf.one_hot(labels[0], image_probs.shape[-1])
    grads.append(get_gradient(image,label))
    for j in range(1,1001):
        grads.append(get_gradient(image,tf.one_hot(j,1000)))
print("--- %s seconds ---",time.time() - start_time)


# %%
tf.norm(grads[0]-grads[34])


# %%
for i in range(1,1001):
    if tf.norm(grads[0]-0)<tf.norm(grads[i]-0):
        print(tf.norm(abs(grads[i])-0))
        print(i)


# %%
num_clusters = 10
num_iter = 100
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters,use_mini_batch=False,seed=1)
previous_centers = None
for _ in range(num_iter):
    kmeans.train(tf.data.Dataset.from_tensors(grads).repeat(1))
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
        print('delta:', cluster_centers - previous_centers)
    previous_centers = cluster_centers
    print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)

cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)


# %%


