# LIBRARIES

# General
import numpy as np
import json
from PIL import Image
import sys

# Visualizations
import matplotlib.pyplot as plt

# Parser
import argparse

# Neural Networks
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# DEFINITIONS

# Variables

batch_size = 32
image_size = 224
class_names = {}

# Functions

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image = image / 255
    return image.numpy()

def predict(file, model, top_k):
    
    image = Image.open(file)
    image = np.asarray(image)
    image = np.expand_dims(image,  axis=0)
    image = process_image(image)
    probabilities = model.predict(image)
    
    classes = []
    probabilities_list = []
    rank = probabilities[0].argsort()[::-1]

    for i in range(top_k):
        idx = rank[i] + 1
        cls = class_names[str(idx)]
        probabilities_list.append(probabilities[0][idx])
        classes.append(cls)
    
    return probabilities_list, classes


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('file_name', help='Image file to be classified')
    parser.add_argument('model', help='Saved model used for classification')
    parser.add_argument('--top_k', type=int, help='Number of top classes')
    parser.add_argument('--category_names',required=False, help='Class names')
    args = parser.parse_args()
    
    file = args.file_name
    
    model = tf.keras.models.load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer})
    
    top_k = args.top_k
    if top_k is None: 
        top_k = 5
        
    with open('label_map.json') as f:
        class_names = json.load(f)
        
    
    probabilities_list, classes = predict(file, model, top_k)
    
    print('Probabilities:',probabilities_list)
    print('Classes:',classes)
    
