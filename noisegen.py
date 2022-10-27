import tensorflow as tf
from tensorflow.python.ops.gen_parsing_ops import decode_raw
from tensorflow_addons.image import filters
import numpy as np
class NoiseGen:
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

    # Iterative noise generation
    def iterative_noise_generation(self,image, d_class, spread=0):
        perturbations = self.adversarial_pattern(self.create_gradient(image,d_class))
        #spread component
        perturbations_spread = self.eps*perturbations + spread*(tf.sign(perturbations) - perturbations)
        iterCount = 1
        adv_x = image - perturbations_spread
        adv_x = tf.clip_by_value(adv_x,self.clipByButtom, self.clipByTop)
        adv_probs = self.pretrained_model.predict(adv_x)
        # Keep looping until we get the aversarial label
        while (np.argmax(adv_probs) != np.argmax(d_class) or np.max(adv_probs) < self.confidence) and iterCount < 1000:
            perturbations += self.adversarial_pattern(self.create_gradient(image,d_class))
            #spread component
            perturbations_spread = self.eps*perturbations + spread*(tf.sign(perturbations) - perturbations)
            adv_x = image - perturbations_spread
            adv_x = tf.clip_by_value(adv_x, self.clipByButtom, self.clipByTop)
            adv_probs = self.pretrained_model.predict(adv_x)
            print('Iteration: {}'.format(iterCount), end='\r')
            iterCount += 1
        return perturbations_spread, iterCount

    #Adversarial Example generation algorithm
    def AE_generation(self,image, d_class):
        image_probs = self.pretrained_model.predict(image)
        perturbations, iterCount = self.iterative_noise_generation(image, d_class)
        adv_x = image - perturbations
        adv_x = tf.clip_by_value(adv_x, self.clipByButtom, self.clipByTop)
        # adv_probs = pretrained_model.predict(adv_x)
        filtered_x = filters.median_filter2d(adv_x, (5, 5))
        filtered_probs = self.pretrained_model.predict(filtered_x)
        spread = 0 #spread multiplier
        while np.argmax(filtered_probs) == np.argmax(image_probs) and iterCount < 1000:
            self.eps += self.increment
            if self.adversarial_pattern == self.adversarial_pattern_PGM:
                spread += self.increment #spread multiplier
            perturbations, j = self.iterative_noise_generation(image, d_class, spread)
            iterCount += j
            adv_x = image - perturbations
            adv_x = tf.clip_by_value(adv_x, self.clipByButtom, self.clipByTop)
            # adv_probs = pretrained_model.predict(adv_x)
            filtered_x = filters.median_filter2d(adv_x, (5, 5))
            filtered_probs = self.pretrained_model.predict(filtered_x)
        return perturbations, self.eps, iterCount
    
    def __init__(self,clipByTop,clipByButtom,pretrained_model,loss_object,adversarial_pattern=adversarial_pattern_FGSM,eps=0.005,increment=0.005,confidence=0.95):
        self.clipByTop = clipByTop
        self.clipByButtom = clipByButtom
        self.pretrained_model = pretrained_model
        self.loss_object = loss_object
        self.adversarial_pattern = adversarial_pattern
        self.eps = eps
        self.increment = increment
        self.confidence = confidence