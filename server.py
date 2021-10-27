from flask import Flask, request, jsonify
from PIL import Image
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image

app = Flask(__name__)
MODEL1= tf.keras.models.load_model('handModelPoopVanila')
MODEL2=tf.keras.models.load_model('PoopingOrNotLast')
MODEL3=tf.keras.models.load_model('VanilaVsUnknown')
MODEL4=tf.keras.models.load_model('PoopVsUnknownUp')


@app.route("/runClassify", methods=["POST"])
def process_image():
    class_names=['poop', 'vanila']
    IMAGE_SHAPE = (224, 224)
    # model = tf.keras.models.load_model('testTransferLerning')
    file = request.files['image']
    # dog1= Image.open(file.stream).resize(IMAGE_SHAPE)
    with Image.open(file.stream) as dog1:
        dog1=dog1.resize(IMAGE_SHAPE)
        dog1 = np.array(dog1)/255.0
        result2 = MODEL1.predict(dog1[np.newaxis, ...])
        score2 = tf.nn.softmax(result2[0])
        msg = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score2)], 100 * np.max(score2))
        print(msg)
        return jsonify(
            {'msg':msg,'name':class_names[np.argmax(score2)],'score':100 * np.max(score2)})


@app.route("/runClassifyPoop", methods=["POST"])
def process_image_pooping():
    class_names=['pooping', 'not pooping']
    IMAGE_SHAPE = (224, 224)
    # model = tf.keras.models.load_model('PoopingOrNotLast')
    file = request.files['image']
    with Image.open(file.stream) as dog1:
        dog1=dog1.resize(IMAGE_SHAPE)
        dog1 = np.array(dog1)/255.0
        result2 = MODEL2.predict(dog1[np.newaxis, ...])
        score2 = tf.nn.softmax(result2[0])
        msg = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score2)], 100 * np.max(score2))
        print(msg)
        return jsonify(
            {'msg':msg,'poopOrNot':class_names[np.argmax(score2)],'score':100 * np.max(score2)})

@app.route("/vanilaOrNot", methods=["POST"])
def process_image_vanilaOrNot():
    class_names=['vanila', 'unknown']
    IMAGE_SHAPE = (224, 224)
    # model = tf.keras.models.load_model('PoopingOrNotLast')
    file = request.files['image']
    with Image.open(file.stream) as dog1:
        dog1=dog1.resize(IMAGE_SHAPE)
        dog1 = np.array(dog1)/255.0
        result2 = MODEL3.predict(dog1[np.newaxis, ...])
        score2 = tf.nn.softmax(result2[0])
        msg = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score2)], 100 * np.max(score2))
        print(msg)
        return jsonify(
            {'msg':msg,'vanilaOrNot':class_names[np.argmax(score2)],'score':100 * np.max(score2)})

@app.route("/poopOrNot", methods=["POST"])
def process_image_poopOrNot():
    class_names=['poop', 'unknown']
    IMAGE_SHAPE = (224, 224)
    # model = tf.keras.models.load_model('PoopingOrNotLast')
    file = request.files['image']
    with Image.open(file.stream) as dog1:
        dog1=dog1.resize(IMAGE_SHAPE)
        dog1 = np.array(dog1)/255.0
        result2 = MODEL4.predict(dog1[np.newaxis, ...])
        score2 = tf.nn.softmax(result2[0])
        msg = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score2)], 100 * np.max(score2))
        print(msg)
        return jsonify(
            {'msg':msg,'poopOrNot':class_names[np.argmax(score2)],'score':100 * np.max(score2)})    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8020)