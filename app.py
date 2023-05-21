from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf 

app = Flask(__name__)

classes = ['Normal Skin','Alpa','Tinea','Vitiligo']
with open('Best1Resnet50.tflite', 'rb') as fid:
    tflite_model = fid.read()
tflite_interpreter = tf.lite.Interpreter(model_content=tflite_model)
print("Model loaded ...")


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image=Image.open(image)
        image = image.resize((100, 100))
        image_arr = np.array(image.convert('RGB'))
        image_arr.shape = (1, 100, 100, 3)
        image_arr = image_arr.astype(np.float32)
        print("Model predicting 111...")
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        tflite_interpreter.resize_tensor_input(input_details[0]['index'], (1, 100, 100, 3))
        tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 4))
        tflite_interpreter.allocate_tensors()
        print("Model predicting 222...")
        tflite_interpreter.set_tensor(input_details[0]['index'], image_arr)
        tflite_interpreter.invoke()
        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        print("Model finished predicting ...")

        print(tflite_model_predictions)
        ind = np.argmax(tflite_model_predictions,)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)