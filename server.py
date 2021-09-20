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

@app.route("/runClassify", methods=["POST"])
def process_image():
    class_names=['poop', 'vanila']
    IMAGE_SHAPE = (224, 224)
    model = tf.keras.models.load_model('testTransferLerning')
    file = request.files['image']
    
    #recive image and save
    # img = Image.open(file.stream)
    # img.resize((img.width, img.height))
    # img.save('im-received.jpg')

    dog1= Image.open(file.stream).resize(IMAGE_SHAPE)
    dog1 = np.array(dog1)/255.0
    result2 = model.predict(dog1[np.newaxis, ...])
    score2 = tf.nn.softmax(result2[0])

    msg = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score2)], 100 * np.max(score2))
    return jsonify(
        {'msg':msg})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8020)
