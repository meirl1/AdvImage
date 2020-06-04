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
import random
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False


# %%
"""Let's load the pretained MobileNetV2 model and the ImageNet class names."""

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


# %%
def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()


# %%
#Iterative Fast Gradient Sign
def IFGS(image,d_class,eps = 0.001):
  print('Target deceiving class: {}'.format(np.argmax(d_class)))
  perturbations = create_adversarial_pattern(image,d_class)
  adv_x = image - eps*perturbations
  adv_x = tf.clip_by_value(adv_x,0,1)
  adv_probs = pretrained_model.predict(adv_x)
  i=0
  while np.argmax(adv_probs) != np.argmax(d_class):#Keep looping until we get the aversarial label
    perturbations += create_adversarial_pattern(adv_x,d_class)
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x,0,1)
    adv_probs = pretrained_model.predict(adv_x)
    print('Attempt: {} image label: {}, Index: {}'.format(i,get_imagenet_label(adv_probs)[1],np.argmax(adv_probs)))
    #displays progress
    i+=1
  return perturbations ,i


# %%
def IAN_generation(image,d_class,eps=0.001,increment=0.001):
    image_probs = pretrained_model.predict(image)
    perturbations, i = IFGS(image,d_class,eps)
    adv_x = image - eps*perturbations
    adv_x = tf.clip_by_value(adv_x,0,1)
    adv_probs = pretrained_model.predict(adv_x)
    filtered_x = tf_filters.median_filter2d(adv_x,(5,5))
    filtered_probs = pretrained_model.predict(filtered_x)
    while np.argmax(filtered_probs) == np.argmax(image_probs):
        eps += increment
        perturbations, j = IFGS(image,d_class,eps)
        i += j
        adv_x = image - eps*perturbations
        adv_x = tf.clip_by_value(adv_x,0,1)
        adv_probs = pretrained_model.predict(adv_x)
        filtered_x = tf_filters.median_filter2d(adv_x,(5,5))
        filtered_probs = pretrained_model.predict(filtered_x)
        print('epsilon: {}, Filtered image label: {} Index: {}'.format(eps,get_imagenet_label(filtered_probs)[1],np.argmax(filtered_probs)))
    return eps*perturbations, eps, i


# %%
#example 

root_path = 'Enter here imagenet 2012 validation path \\ILSVRC2012_val_0000'

adv_success=0
filter_success=0
resHeaders = ['Original','Target','Actual','Filtered','FGS Iter','eps']
results = []
random.seed(0)
for i in range(1,10): #Image ILSVRC2012_val_00000003.jpeg is the same image used on the research paper
    print('file:'+root_path+f'{i:03}.JPEG')
    image_raw = tf.io.read_file(root_path+f'{i:03}.JPEG')#You may change it to any file you like
    image = tf.image.decode_image(image_raw)
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)
    label = tf.one_hot(np.argmax(image_probs), image_probs.shape[-1])
    label = tf.reshape(label,(1,image_probs.shape[-1]))
    print('Image label: {}, Index: {}'.format(get_imagenet_label(image_probs)[1],np.argmax(image_probs)))
    #display_images(image,'input')
    randnum = random.randint(1,1000)
    d_class = tf.one_hot(randnum,1000)
    d_class = tf.reshape(d_class,(1,1000))
    delta, eps, i = IAN_generation(image,d_class)
    adv_x = image - delta
    adv_x = tf.clip_by_value(adv_x,0,1)
    adv_probs = pretrained_model.predict(adv_x)
    print('Adv image label: {}, Index: {}'.format(get_imagenet_label(adv_probs)[1],np.argmax(adv_probs)))
    
    #display_images(adv_x,'input')

    if np.argmax(adv_probs) != np.argmax(image_probs):
        print('Success')
        adv_success+=1
    else:
        print('Failed')
    
    filtered_x = tf_filters.median_filter2d(adv_x,(5,5))
    filtered_probs = pretrained_model.predict(filtered_x)
    print('Filtered image label: {} Index: {}'.format(get_imagenet_label(filtered_probs)[1],np.argmax(filtered_probs)))
    #display_images(filtered_x,'input')

    if np.argmax(filtered_probs) == np.argmax(image_probs):
        filter_success+=1
    results.append((np.argmax(image_probs),randnum,np.argmax(adv_probs),np.argmax(filtered_probs),i,eps))
print('{} {}'.format(adv_success,filter_success))

#For any explanation please contact me


# %%
results


# %%
#decode_predictions(np.zeros((1,1000)),top=1000)[0][1]
decode_predictions(d_class.numpy())


# %%
results

