import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_parsing_ops import decode_raw
from tensorflow_addons.image import filters
import numpy as np
class NoiseGen:
    def __init__(self,clipByTop,clipByButtom,pretrained_model,decode_predictions,preprocess_input,loss_object):
        self.clipByTop = clipByTop
        self.clipByButtom = clipByButtom
        self.pretrained_model = pretrained_model
        self.decode_predictions = decode_predictions
        self.preprocess_input = preprocess_input
        self.pretrained_model.trainable = False
        self.input_shape = pretrained_model.input_shape[1]
        self.loss_object = loss_object

    def create_gradient(self,input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.pretrained_model(input_image)
            loss = self.loss_object(input_label, prediction)
            # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        return gradient

    #Fast Gradient Sign Method
    def adversarial_pattern_FGSM(gradient):
        return tf.sign(gradient) #invgrad/np.max(np.abs(invgrad))

    #Precise Gradient Method
    def adversarial_pattern_PGM(gradient):
        return gradient/tf.reduce_max(tf.abs(gradient))

    #Fast Gradient Non sign Method
    def adversarial_pattern_FGNM(gradient):
        return(tf.norm(tf.sign(gradient))/tf.norm(gradient))*gradient

    def display_images(self,image):
        probs = self.pretrained_model.predict(image)
        _, label, confidence = self.decode_predictions(probs, top=1)[0][0]
        plt.figure()
        plt.imshow(image[0]*0.5+0.5)
        plt.title('{} \n {} : {:.2f}% Confidence'.format(np.argmax(probs), label, confidence*100))
        plt.show()

    # Iterative noise generation
    def iterative_noise_generation(self,image, d_class, eps=0.001, spread=0, confidence=0.7,adversarial_pattern=adversarial_pattern_FGSM):
        perturbations = adversarial_pattern(self.create_gradient(image,d_class))
        #spread component
        perturbations_spread = eps*perturbations + spread*(tf.sign(perturbations) - perturbations)
        iterCount = 1
        adv_x = image - perturbations_spread
        adv_x = tf.clip_by_value(adv_x,self.clipByButtom, self.clipByTop)
        adv_probs = self.pretrained_model.predict(adv_x)
        # Keep looping until we get the aversarial label
        while (np.argmax(adv_probs) != np.argmax(d_class) or np.max(adv_probs) < confidence) and iterCount < 1000:
            perturbations += adversarial_pattern(self.create_gradient(image,d_class))
            #spread component
            perturbations_spread = eps*perturbations + spread*(tf.sign(perturbations) - perturbations)
            adv_x = image - perturbations_spread
            adv_x = tf.clip_by_value(adv_x, self.clipByButtom, self.clipByTop)
            adv_probs = self.pretrained_model.predict(adv_x)
            print('Iteration: {}'.format(iterCount), end='\r')
            iterCount += 1
        return perturbations_spread, iterCount

    #Adversarial Example generation algorithm
    def AE_generation(self,image, d_class, eps=0.005, increment=0.005,adversarial_pattern=adversarial_pattern_FGSM):
        image_probs = self.pretrained_model.predict(image)
        perturbations, iterCount = self.iterative_noise_generation(image, d_class, eps,adversarial_pattern=adversarial_pattern)
        adv_x = image - perturbations
        adv_x = tf.clip_by_value(adv_x, self.clipByButtom, self.clipByTop)
        # adv_probs = pretrained_model.predict(adv_x)
        filtered_x = filters.median_filter2d(adv_x, (5, 5))
        filtered_probs = self.pretrained_model.predict(filtered_x)
        spread = 0 #spread multiplier
        while np.argmax(filtered_probs) == np.argmax(image_probs) and iterCount < 1000:
            eps += increment
            if adversarial_pattern == self.adversarial_pattern_PGM:
                spread += increment #spread multiplier
            perturbations, j = self.iterative_noise_generation(image, d_class, eps, spread,adversarial_pattern=adversarial_pattern)
            iterCount += j
            adv_x = image - perturbations
            adv_x = tf.clip_by_value(adv_x, self.clipByButtom, self.clipByTop)
            # adv_probs = pretrained_model.predict(adv_x)
            filtered_x = filters.median_filter2d(adv_x, (5, 5))
            filtered_probs = self.pretrained_model.predict(filtered_x)
        return perturbations, eps, iterCount

    def preprocess(self,image):
        image = tf.image.resize(image, (224, 224))
        image = self.preprocess_input(image)
        image = image[None, ...]
        return image