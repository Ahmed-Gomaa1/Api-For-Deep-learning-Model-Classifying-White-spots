from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf 

app = Flask(__name__)

classes = ['Normal Skin','Pityriasis Alba','Tinea Versicolor','Vitiligo']
with open('Best1Resnet50.tflite', 'rb') as fid:
    tflite_model = fid.read()
tflite_interpreter = tf.lite.Interpreter(model_content=tflite_model)
print("Model loaded ...")

def preprossing(image):
    image=Image.open(image)
    image = image.resize((100, 100))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 100, 100, 3)
    image_arr = image_arr.astype(np.float32)
    return image_arr

@app.route('/')
def index():

    return render_template('index.html')


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("image loaded ...")
        image_arr = preprossing(image)
        print("predicting ...")
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.resize_tensor_input(input_details[0]['index'], (1, 100, 100, 3))
        tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 4))
        tflite_interpreter.allocate_tensors()
        tflite_interpreter.set_tensor(input_details[0]['index'], image_arr)
        tflite_interpreter.invoke()
        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        print("Model finished predicting ...")

        print(tflite_model_predictions)
        ind = np.argmax(tflite_model_predictions)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        image_arr= preprossing(image)
        print("predicting ...")
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.resize_tensor_input(input_details[0]['index'], (1, 100, 100, 3))
        tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 4))
        tflite_interpreter.allocate_tensors()
        tflite_interpreter.set_tensor(input_details[0]['index'], image_arr)
        tflite_interpreter.invoke()
        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        print("Model finished predicting ...")

        print(tflite_model_predictions)
        ind = np.argmax(tflite_model_predictions)
        prediction = classes[ind]
        print(prediction)
        return render_template('index.html', prediction=prediction, image='static/IMG/')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
