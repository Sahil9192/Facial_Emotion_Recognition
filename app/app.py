from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from utils import preprocess_image

app = Flask(__name__)
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'emotion_model.h5'))
model = load_model(MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded.", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No file selected.", 400

    # Save and read the image
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    roi = preprocess_image(img_path)
    prediction = model.predict(roi, verbose=0)
    emotion = emotion_labels[np.argmax(prediction)]

    return render_template('index.html', emotion=emotion, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
