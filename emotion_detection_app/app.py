from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load your emotion detection model
model = load_model(r"C:\Users\KIIT0001\Desktop\Emotion Detection\model_best.keras")  
# Define the emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((48, 48))  # Resize the image as per your model's requirement
            img = img.convert('L')  # Convert to grayscale if your model expects grayscale images
            img = np.array(img)
            img = img / 255.0  # Normalize the image
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)  # If the model expects 4D input

            prediction = model.predict(img)
            emotion = EMOTIONS[np.argmax(prediction)]

            return render_template('index.html', prediction=emotion)
    return redirect(url_for('index'))

if __name__ == '_main_':
    app.run(debug=True)