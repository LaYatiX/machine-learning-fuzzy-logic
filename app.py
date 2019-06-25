import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

from flask import request
from flask import jsonify
from flask import Flask

import cv2

from PIL import Image
from io import StringIO
import base64

# disable GPU tensorflow
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import tensorflow as tf
global graph,model,model2
graph = tf.get_default_graph()

app = Flask(__name__)

# def get_model():
#     global model
#     model = load_model('VGG16_cats_and_dogs.h5')
#     print(" * Model loaded!")

def get_model2():
    global model2
    model2 = load_model('keras-multi-label\\models\\fashion.model')
    print(" * Model loaded!2")

# def preprocess_image(image, target_size):
#     if (image.mode != "RGB"):
#         image = image.convert("RGB")
#     image = image.resize(target_size)
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
    
#     return image

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def preprocess_image2(image, target_size):
    image = cv2.resize(image, (96, 96))
    # if (image.mode != "RGB"):
    #     image = image.convert("RGB")
    # image = image.resize(target_size)
    # image = image.astype("float") / 255.0
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    
    return image    

# print(" * Loading Keras model...")
# get_model()

print(" * Loading Keras model... 222222 ")
get_model2()


@app.route('/predict2', methods=['POST'])
def predict2():
    message = request.get_json(force=True)
    encoded = message['image']
    # print(encoded)
    # decoded = base64.b64decode(encoded)
    # image = Image.open(io.BytesIO(decoded))
    # processed_image = preprocess_image2(image, target_size=(96,96))
    image = stringToImage(encoded)
    image = toRGB(image)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    with graph.as_default():
        prediction = model2.predict(image)[0]

    print(prediction)

    response = {
        'prediction' : {
            'black' : str(prediction[0]),
            'blue' : str(prediction[1]),
            'dress' : str(prediction[2]),
            'jeans' : str(prediction[3]),
            'red' : str(prediction[4]),
            'shirt' : str(prediction[5])
        }
    }
    return jsonify(response)

@app.route('/sample')
def running():
    return 'Flask is running!'

@app.route('/hello', methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeteng' : 'Hello' + name + '!'
    }
    return jsonify(response)


# @app.route('/predict', methods=['POST'])
# def predict():
#     message = request.get_json(force=True)
#     encoded = message['image']
#     decoded = base64.b64decode(encoded)
#     image = Image.open(io.BytesIO(decoded))
#     processed_image = preprocess_image(image, target_size=(224,224))

#     with graph.as_default():
#         prediction = model.predict(processed_image).tolist()

#     response = {
#         'prediction' : {
#             'dog' : prediction[0][0],
#             'cat' : prediction[0][1]
#         }
#     }
#     return jsonify(response)