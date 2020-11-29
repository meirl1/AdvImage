#%%
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
modelChoice = 4
perturCo = 1
clipByTop = 1
clipByButtom = -1

if modelChoice < 10:
    if modelChoice < 8:
        if modelChoice == 0:
            pretrained_model = tf.keras.applications.EfficientNetB0(include_top=True, weights='imagenet')        
        if modelChoice == 1:
            pretrained_model = tf.keras.applications.EfficientNetB1(include_top=True, weights='imagenet')
        if modelChoice == 2:
            pretrained_model = tf.keras.applications.EfficientNetB2(include_top=True, weights='imagenet')
        if modelChoice == 4:
            pretrained_model = tf.keras.applications.EfficientNetB4(include_top=True, weights='imagenet')
        if modelChoice == 5:
            pretrained_model = tf.keras.applications.EfficientNetB5(include_top=True, weights='imagenet')
        if modelChoice == 6:
            pretrained_model = tf.keras.applications.EfficientNetB6(include_top=True, weights='imagenet')
        if modelChoice == 7:
            pretrained_model = tf.keras.applications.EfficientNetB7(include_top=True, weights='imagenet')

        decode_predictions = tf.keras.applications.efficientnet.decode_predictions
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        clipByTop = 255
        clipByButtom = 0
    else:
        if modelChoice == 8:
            pretrained_model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
            decode_predictions = tf.keras.applications.vgg19.decode_predictions
            preprocess_input = tf.keras.applications.vgg19.preprocess_input
        else:#9
            pretrained_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
            decode_predictions = tf.keras.applications.vgg16.decode_predictions
            preprocess_input = tf.keras.applications.vgg16.preprocess_input
        clipByTop = [255-103.939 ,255-116.779 ,255-123.68 ]
        clipByButtom = [-103.939 ,-116.779 ,-123.68 ]
    perturCo = 255

elif modelChoice == 10:
    pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,weights='imagenet')
    decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
elif modelChoice == 11:
    pretrained_model = tf.keras.applications.InceptionResNetV2(include_top=True, weights='imagenet')
    decode_predictions = tf.keras.applications.inception_resnet_v2.decode_predictions
    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

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
    #Fixing zero gradient
    '''if np.min(gradient) == 0 and np.max(gradient) == 0:
        print(np.min(gradient),np.max(gradient))
        gradient = tf.random.Generator.from_seed(1).normal(gradient.shape)'''
    
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad*max(perturCo-255/2,1)
 
 
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
    image = tf.image.resize(image, [input_shape, input_shape])
    # label prediction
    return image, label
 
 
def display_images(image):
    probs = pretrained_model.predict(image)
    _, label, confidence = decode_predictions(probs, top=1)[0][0]
    plt.figure()
    if modelChoice < 8:
        plt.imshow(image[0].numpy().astype("uint8"))
    elif modelChoice < 10:
        plt.imshow(tf.reverse(image[0]+[103.939 ,116.779 ,123.68 ],axis=[2]).numpy().astype("uint8"))
    else:
        plt.imshow(image[0]*0.5+0.5)
 
    plt.title('{} \n {} : {:.2f}% Confidence'.format(np.argmax(probs), label, confidence*100))
    plt.show()
#%%
# path to files
with open('rootpath.txt') as f:
    root_path = f.readline()
 
list_ds = tf.data.Dataset.list_files(root_path + '/ILSVRC2012_img_val/ILSVRC2012_val_0000????.JPEG', shuffle=False)
data = list_ds.map(preprocess)
 
# Iterative Fast Gradient Sign method
 
 
def IFGS(image, d_class, eps=0.001):
    #print('Target deceiving class: {}'.format(np.argmax(d_class)))
    perturbations = create_adversarial_pattern(image, d_class)
    iterCount = 1
    print(np.min(perturbations),np.max(perturbations))
    if np.min(perturbations) == 0 and np.max(perturbations) == 0:
        return perturbations, iterCount
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x, clipByButtom, clipByTop)
    adv_probs = pretrained_model.predict(adv_x)
    # Keep looping until we get the aversarial label
    while np.argmax(adv_probs) != np.argmax(d_class):
        perturbations += create_adversarial_pattern(adv_x, d_class)
        adv_x = image - eps*perturbations
        adv_x = tf.clip_by_value(adv_x, clipByButtom, clipByTop)
        adv_probs = pretrained_model.predict(adv_x)
        #if iterCount%10 == 0:
        #    display_images(adv_x)
        #print('Attempt: {} image label: {}, Index: {}'.format(iterCount, decode_predictions(adv_probs, top=1)[0][0][1], np.argmax(adv_probs)))
        # displays progress
        iterCount += 1
    return perturbations, iterCount
 
 
def IAN_generation(image, d_class, eps=0.005, increment=0.005):
    image_probs = pretrained_model.predict(image)
    perturbations, iterCount = IFGS(image, d_class, eps)
    if np.min(perturbations) == 0 and np.max(perturbations) == 0:
        return perturbations, eps, iterCount
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x, clipByButtom, clipByTop)
    # adv_probs = pretrained_model.predict(adv_x)
    filtered_x = filters.median_filter2d(adv_x, (5, 5))
    filtered_probs = pretrained_model.predict(filtered_x)
    while np.argmax(filtered_probs) == np.argmax(image_probs):
        eps += increment
        perturbations, j = IFGS(image, d_class, eps)
        iterCount += j
        adv_x = image - eps*perturbations
        adv_x = tf.clip_by_value(adv_x, clipByButtom, clipByTop)
        # adv_probs = pretrained_model.predict(adv_x)
        filtered_x = filters.median_filter2d(adv_x, (5, 5))
        filtered_probs = pretrained_model.predict(filtered_x)
        print('eps: {},iterCount: {}, Filtered image label: {} Index: {}'.format(
            eps, iterCount, decode_predictions(filtered_probs, top=1)[0][0][1], np.argmax(filtered_probs)))
    return eps*perturbations, eps, iterCount

#%%
#Exectution
import random
import time
adv_success = 0
filter_success = 0
skip_files = 433
distance = 20
results = []
st_time = time.time()
for img in data.skip(skip_files).take(1100 - skip_files):
    image = img[0]
    image_probs = pretrained_model.predict(image)
    print(str(img[1]).split("'")[1])
    print(decode_predictions(image_probs)[0][0][1])
    #display_images(image)
    #Random decieving class
    #dNumber = 'NA'
    #iterNumRand = 'NA'
    dNumber = random.randrange(1,1000)
    while np.argmax(image_probs) == dNumber:
        dNumber = random.randrange(1,1000)
    d_class = tf.one_hot(dNumber, 1000)
    d_class = tf.reshape(d_class, (1, 1000))
    print('decieving class: {} {}'.format(dNumber,decode_predictions(d_class.numpy(), top=1)[0][0][1]))
    delta, eps, iterNumRand = IAN_generation(image, d_class,eps=0.005,increment=0.005)
    adv_x = image - delta
    adv_x = tf.clip_by_value(adv_x, clipByButtom, clipByTop)
    #display_images(adv_x)
    adv_probs = pretrained_model.predict(adv_x)
    if np.argmax(adv_probs) != np.argmax(image_probs):
        print('Success')
    else:
        iterNumRand = 'failed'
        print(iterNumRand)
    
    distance = random.randint(15,50)
    print("distance: ",distance)
    #Distance based decieving class
    dNumberOpt = np.argsort(np.max(image_probs,axis=0))[-(distance+1)]
    d_class = tf.one_hot(dNumberOpt, 1000)
    d_class = tf.reshape(d_class, (1, 1000))
    print('decieving class: {} {}'.format(dNumberOpt,decode_predictions(d_class.numpy(), top=1)[0][0][1]))
    delta, eps, iterNumOpt = IAN_generation(image, d_class,eps=0.005,increment=0.005)
    adv_x = image - delta
    adv_x = tf.clip_by_value(adv_x, clipByButtom, clipByTop)
    #display_images(adv_x)
    adv_probs = pretrained_model.predict(adv_x)
    if np.argmax(adv_probs) != np.argmax(image_probs):
        print('Success')
    else:
        iterNumOpt = 'Failed'
        print(iterNumOpt)
    filtered_x = filters.median_filter2d(adv_x, (5, 5))
    filtered_probs = pretrained_model.predict(filtered_x)
    #display_images(filtered_x, 'input')
 
    if np.argmax(filtered_probs) == np.argmax(image_probs):
        filter_success += 1
    print('iter opt: {}, iter rand: {}'.format(iterNumOpt,iterNumRand))
    with open('EffiNet4_15_50.txt',mode='a') as resf:
        resf.write('{},{},{},{},{},{},{},{}\n'.format(str(img[1]).split("'")[1],pretrained_model.name,distance,np.argmax(image_probs),dNumber,dNumberOpt,iterNumRand,iterNumOpt))
    
en_time = st_time - time.time()
print('{} {}'.format(adv_success, filter_success))


# %%
