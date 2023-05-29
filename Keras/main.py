import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import glob
from BCDUNet import BCDUNet

# Download a subset of the ADE20K dataset
img_path = "./data/train"

img_dir = os.path.join(os.path.dirname(img_path), "train/source")
ann_dir = os.path.join(os.path.dirname(img_path), "train/tactile")

# Load and preprocess the dataset
def load_image_and_annotation(image_path, annotation_path, input_size=(256, 256)):
    image = load_img(image_path, target_size=input_size)
    image = img_to_array(image) / 255.0
    
    annotation = load_img(annotation_path, target_size=input_size)
    annotation = img_to_array(annotation)
    annotation = img_to_array(annotation) / 255.0
    
    
    # annotation = load_img(annotation_path, target_size=input_size, color_mode="grayscale")
    # annotation = img_to_array(annotation)
    # annotation = np.where(annotation == 255, 0, annotation) - 1
    # annotation = to_categorical(annotation, num_classes=3)

    return image, annotation

image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
annotation_paths = sorted(glob.glob(os.path.join(ann_dir, "*.tiff")))

images = []
annotations = []

for img_path, ann_path in zip(image_paths, annotation_paths):
    img, ann = load_image_and_annotation(img_path, ann_path)
    images.append(img)
    annotations.append(ann)

images = np.array(images)
annotations = np.array(annotations)

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, annotations, train_size=0.8, test_size=0.2, random_state=42)

# Create the BCDUNet model
model = BCDUNet(input_size=(256, 256, 3), output_c=3)

# Train the model
model.fit(x_train, y_train, batch_size=4, epochs=10, validation_data=(x_val, y_val), verbose=2)

# Evaluate the model using Intersection over Union (IoU) metric
def calculate_iou(y_true, y_pred):
    o = np.array(y_pred)
    r = np.array(y_true)
    intersection = np.sum(o * r)
    union = np.sum(o**2 + r**2 - o*r)
    iou_score = intersection / union 
    
    # y_true = np.argmax(y_true, axis=-1)
    # y_pred = np.argmax(y_pred, axis=-1)

    # intersection = np.logical_and(y_true, y_pred)
    # union = np.logical_or(y_true, y_pred)
    # iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

y_pred = model.predict(x_val)
iou_score = calculate_iou(y_val, y_pred)
print("IoU score:", iou_score)
