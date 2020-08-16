# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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
#Select model
modelChoice = 7
perturCo = 1

if modelChoice < 8:
    if modelChoice == 0:
        pretrained_model = tf.keras.applications.EfficientNetB0(include_top=True, weights='imagenet')        

    if modelChoice == 1:
        pretrained_model = tf.keras.applications.EfficientNetB1(include_top=True, weights='imagenet')

    if modelChoice == 2:
        pretrained_model = tf.keras.applications.EfficientNetB2(include_top=True, weights='imagenet')
    
    if modelChoice == 6:
        pretrained_model = tf.keras.applications.EfficientNetB6(include_top=True, weights='imagenet')

    if modelChoice == 7:
        pretrained_model = tf.keras.applications.EfficientNetB7(include_top=True, weights='imagenet')

    decode_predictions = tf.keras.applications.efficientnet.decode_predictions
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    perturCo = 255

pretrained_model.trainable = False
input_shape = pretrained_model.input_shape[1]
# Helper function to preprocess the image so that it can be inputted in MobileNetV2


def preprocess(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (input_shape, input_shape))
    image = preprocess_input(image)
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
    return perturCo*signed_grad


def get_gradient(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    return perturCo*gradient
    # Get the sign of the gradients to create the perturbation


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [input_shape, input_shape])
    # label prediction
    return image, label


def display_images(image):
    probs = pretrained_model.predict(image)
    _, label, confidence = decode_predictions(probs, top=1)[0][0]
    plt.figure()
    plt.imshow(image[0].numpy().astype("uint8"))
    plt.title('{} \n {} : {:.2f}% Confidence'.format(np.argmax(probs), label, confidence*100))
    plt.show()


# path to files
with open('rootpath.txt') as f:
    root_path = f.readline()

list_ds = tf.data.Dataset.list_files(
    root_path + '\\ILSVRC2012_img_val\\*.JPEG', shuffle=False)
data = list_ds.map(preprocess)
# %%
# Iterative Fast Gradient Sign method


def IFGS(image, d_class, eps=0.001):
    #print('Target deceiving class: {}'.format(np.argmax(d_class)))
    perturbations = create_adversarial_pattern(image, d_class)
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -perturCo, perturCo)
    adv_probs = pretrained_model.predict(adv_x)
    iterCount = 1
    # Keep looping until we get the aversarial label
    while np.argmax(adv_probs) != np.argmax(d_class):
        perturbations += create_adversarial_pattern(adv_x, d_class)
        adv_x = image - eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -perturCo, perturCo)
        adv_probs = pretrained_model.predict(adv_x)
        print('Attempt: {} image label: {}, Index: {}'.format(
            iterCount, decode_predictions(adv_probs, top=1)[0][0][1], np.argmax(adv_probs)))
        # displays progress
        iterCount += 1
    return perturbations, iterCount


def IAN_generation(image, d_class, eps=0.005, increment=0.005):
    image_probs = pretrained_model.predict(image)
    perturbations, iterCount = IFGS(image, d_class, eps)
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -perturCo, perturCo)
    # adv_probs = pretrained_model.predict(adv_x)
    filtered_x = filters.median_filter2d(adv_x, (5, 5))
    filtered_probs = pretrained_model.predict(filtered_x)
    while np.argmax(filtered_probs) == np.argmax(image_probs):
        eps += increment
        perturbations, j = IFGS(image, d_class, eps)
        iterCount += j
        adv_x = image - eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -perturCo, perturCo)
        # adv_probs = pretrained_model.predict(adv_x)
        filtered_x = filters.median_filter2d(adv_x, (5, 5))
        filtered_probs = pretrained_model.predict(filtered_x)
        print('eps: {},iterCount: {}, Filtered image label: {} Index: {}'.format(
            eps, iterCount, decode_predictions(filtered_probs, top=1)[0][0][1], np.argmax(filtered_probs)))
    return eps*perturbations, eps, iterCount

# %%
import random
adv_success = 0
filter_success = 0
results = []
iterRandCount = 0
iterOptCount = 0
for img in data.take(10):
    image = img[0]
    image_probs = pretrained_model.predict(image)
    print(decode_predictions(image_probs)[0][0][1])
    label = tf.one_hot(np.argmax(image_probs), image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))
    display_images(image)
    dNumber = random.randrange(1,1000)
    while np.argmax(image_probs) == dNumber:
        dNumber = random.randrange(1,1000)
    d_class = tf.one_hot(dNumber, 1000)
    d_class = tf.reshape(d_class, (1, 1000))
    print('decieving class: {} {}'.format(dNumber,decode_predictions(d_class.numpy(), top=1)[0][0][1]))
    delta, eps, iterNumRand = IAN_generation(image, d_class,eps=0.005,increment=0.005)
    adv_x = image - delta
    adv_x = tf.clip_by_value(adv_x, -perturCo, perturCo)
    display_images(adv_x)
    adv_probs = pretrained_model.predict(adv_x)
    iterRandCount += iterNumRand

    dNumberOpt = np.argsort(np.max(image_probs,axis=0))[-10]
    d_class = tf.one_hot(dNumberOpt, 1000)
    d_class = tf.reshape(d_class, (1, 1000))
    print('decieving class: {} {}'.format(dNumberOpt,decode_predictions(d_class.numpy(), top=1)[0][0][1]))
    delta, eps, iterNumOpt = IAN_generation(image, d_class,eps=0.005,increment=0.005)
    adv_x = image - delta
    adv_x = tf.clip_by_value(adv_x, -perturCo, perturCo)
    display_images(adv_x)
    adv_probs = pretrained_model.predict(adv_x)
    iterOptCount += iterNumOpt
    if np.argmax(adv_probs) != np.argmax(image_probs):
        print('Success')
        adv_success += 1
    else:
        print('Failed')
    filtered_x = filters.median_filter2d(adv_x, (5, 5))
    filtered_probs = pretrained_model.predict(filtered_x)
    #display_images(filtered_x, 'input')

    if np.argmax(filtered_probs) == np.argmax(image_probs):
        filter_success += 1
    print('iter rand: {}, iter opt: {}'.format(iterNumRand,iterNumOpt))
    with open('results01.csv',mode='a') as resf:
        resf.write('{},{},{},{}\n'.format(str(img[1]).split("'")[1],pretrained_model.name,iterNumRand,iterNumOpt))
print('{} {}'.format(adv_success, filter_success))
# %%
pretrained_model.input_shape[1]
# %%
