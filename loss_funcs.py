### This module contains the loss function for the model ###

# We're going to experiment with different loss functions ###

### According to the paper by Shruti Jadon "A survey of loss functions for semantic segmentation" (2020)###


# Importing
import tensorflow as tf 
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K

class loss_function(object):
    """
    This class contains multiple loss functions includes dice loss, binary_cross entropy, and 
    """
    def __init__(self):
        print("Your loss function class initiate successfully!")
    
    
    def dice_coefficient(self, y_true, y_pred, control=1e-6):
        """
        Computer the dice coefficient formula: 
        Dice = 2 * (intersection between y_true and y_pred) / (y_true + y_pred)
        
        Params:
        @ y_true: true labels
        @ y_pred: predicted labels 
        @ control: a super small number to prevent possible division by zero error 
        """
        
        # Flattened y_true and y_pred is necessary since we are working on element-wise operations here
        # we're comparing for each individual pixel for the predicted mask to ...
        # each pixel of the ground truth mask 
        
        y_true_flattened = K.flatten(y_true) 
        y_pred_flattened = K.flatten(y_pred) 
        
        # Since this is a binary mask, the values on y_pred and y_true
        # ... is either 1 or 0
        # So, y_pred * y_true can either be 1 or 0 
        intersection = K.sum(y_true_flattened * y_pred_flattened) 
        
        dice_coefficient = 2 * (intersection + control) / (K.sum(y_true_flattened) + K.sum(y_pred_flattened) + control) 
        # add the control to prevent division by zero error 
        
        return dice_coefficient
        
        
    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coefficient(y_true, y_pred) 
        return loss
    
            
    def binary_cross_entropy(self, y_true, y_pred):
        """
        Binary corss-entropy function measures the differences between two probability distributions.
        
        Formula for binary cross-entropy: - (y * log(p) + (1 - y) * log(1 - p))
        
        Params:
        @ y_true: true labels
        @ y_pred: predicted labels 
        """
        bce = BinaryCrossentropy() # initiate a binary cross entropy function 
        loss = bce(y_true, y_pred) 
        return loss
        
    
    def weighted_binary_cross_entropy(self, y_true, y_pred, beta):
        """
        Weighted binary cross entropy includes an extrat beta value that can tune false negative and false positives. 
        This allow us to decide if we want to penalize false negative or false positive more? 
        
        Is false positive or false negative harm more in our task? 
        For our image segmentation mask, overestimating is more dangerous. i.e. predicted spots that should belong to the background, but classified to be involved into the mask. 
        
        Therefore, we want to penalize false positive more. 
        
        According to Shruti Jadon 2020, beta < 1 for weighted binary cross-entropy if we want to decrease false positives.  
        
        Params:
        @ y_true: true labels
        @ y_pred: predicted labels 
        @ beta: used to tune false positive or false negative 
        """
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits= y_pred, pos_weight=beta) 
        # normalize the loss
        loss = tf.reduce_mean(loss) 
        return loss 
    
    
    def bce_dice_loss(self, y_true, y_pred, beta, alpha):
        """
        This function combines the two loss function together: dice loss and binary cross-entropy.
        This equation is from Shruti jadon 2020 paper. The parameter alpha allows users to adjust the flexibility between
        using Dice loss or cross-entropy. 
        """
        loss = alpha * self.weighted_binary_cross_entropy(y_true, y_pred, beta) + (1 - alpha) * self.dice_loss(y_true, y_pred) 
        
        return loss 
    
    