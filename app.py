from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join('static', file.filename)
    os.makedirs('static', exist_ok=True)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)

    if result[0][0] > 0.5:
        prediction = "Tumor Detected"
    else:
        prediction = "No Tumor"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
