## This file is similar to the jupyter notebook with the same name 
# Just that we can run bash script to generate predictions in batch 

# import
import nibabel as nib
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from loss_funcs import loss_function
import sys
import time

# config to use the gpu of the server 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Get input and output file paths from command-line arguments
input_nifti_path = sys.argv[1]  # First argument: input .nii file
output_nifti_path = sys.argv[2]  # Second argument: output .nii file


nifti = nib.load(input_nifti_path)
# retrieve nifti data
data = nifti.get_fdata()
data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
data = data.astype(np.uint8)  # Convert to 8-bit unsigned integers

# Create a temporary output directory to saven temporal png files
# This directory will overwrite everytime we call this .py
output_dir = "/home/smooi/vibes_auto_pipe/output" # tmp

# Iterate through the slices 
for i in range(data.shape[2]): #iter thru 1 to 48 
    slice_ = data[:, :, i] # 2D slices
    resized_slice = cv2.resize(slice_, (160, 160), interpolation=cv2.INTER_AREA)
    # rotate the image by 90 degrees counterclockwise 
    rotated_slice = cv2.rotate(resized_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)  
    # Scale to Full Grayscale Range [0, 255]
    img_max_range = rotated_slice.max()
    output_img = (rotated_slice / img_max_range) * 255
    output_path = os.path.join(output_dir, f"{i:03d}.png") 
    cv2.imwrite(output_path, output_img)
# debug message 
time.sleep(4)

print('Conversion to PNG files finished!')
    

def prepare_image_for_tf(img):
    brain_img = tf.io.read_file(img)
    brain_img = tf.io.decode_png(brain_img, channels=1) # brain img channels=1 greyscale
    brain_img = tf.image.resize(brain_img, [160, 160]) # Necessary since inputs may be varying in size from different trials
    brain_img = tf.cast(brain_img, tf.float32)  
    
    return brain_img

# make this to be a global variable 
store_filenames = []
def retrieve_n_sort(directory):
    # List Comprehensions
    filenames = [filename for filename in os.listdir(directory) if filename.endswith(".png")]
    # lambda func
    filenames.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))  # Sort by slice number
    return filenames


# Process all images in the directory
def process_all_images(directory):
    sorted_filenames = retrieve_n_sort(directory)
    image_tensors = [prepare_image_for_tf(os.path.join(directory, filename)) for filename in sorted_filenames]
    return tf.stack(image_tensors)  # Stack into a batch tensor

# Process images in the directory
image_batch = process_all_images(output_dir)
print(f"Processed Image Batch Shape: {image_batch.shape}") # Processed Image Batch Shape: (48, 160, 160, 1)  48 refers to the batch size 


# create the instance
my_loss_func = loss_function()

dice_coefficient = my_loss_func.dice_coefficient
dice_loss = my_loss_func.dice_loss


# Load the train model:
saved_model_path = "/home/smooi/vibes_auto_pipe/unet_w_aug.h5"
model = load_model(saved_model_path, custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient})
print("Model Loaded Successfully!")


# Make the predictions
predictions = model.predict(image_batch)
binary_predictions = (predictions > 0.5).astype(int)
predicted_mask = binary_predictions.squeeze()
predicted_mask_reordered = np.transpose(predicted_mask, (1, 2, 0))

# Save output NIfTI file
nib.save(nib.Nifti1Image(predicted_mask_reordered, nifti.affine), output_nifti_path)
print(f"Output NIfTI saved at {output_nifti_path}!")