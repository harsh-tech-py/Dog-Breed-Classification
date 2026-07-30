from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join('models', 'fine_tuned_inception.h5')
CLASS_NAMES_PATH = os.path.join('models', 'class_names.json')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# InceptionV3 input size used during training (see notebook: target_size=(224,224))
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload limit

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# class_names.json should look like: ["beagle", "labrador_retriever", "poodle", ...]
# The order MUST match train_generator.class_indices (or equivalent) from your notebook.
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
else:
    class_names = []
    print(f"WARNING: {CLASS_NAMES_PATH} not found. Predictions will show raw indices.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Matches notebook: simple rescale to [0, 1] (NOT InceptionV3's preprocess_input)
    img_array = img_array / 255.0
    return img_array


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG or JPEG.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)[0]

        top_n = 3
        top_indices = predictions.argsort()[-top_n:][::-1]

        results = []
        for i in top_indices:
            label = class_names[i] if class_names else f"Class {i}"
            results.append({
                'breed': label.replace('_', ' ').title(),
                'confidence': round(float(predictions[i] * 100), 2)
            })

        os.remove(filepath)

        return jsonify({'success': True, 'predictions': results})

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size is 8MB.'}), 413


if __name__ == '__main__':
    app.run(debug=True)