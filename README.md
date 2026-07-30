# 🐾 Dog Breed Detector

A Flask web app that identifies a dog's breed from an uploaded photo using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

Upload an image, and the model returns the top-3 most likely breeds with confidence scores — all through a clean, drag-and-drop UI.

---

## Features

- 🖼️ Drag-and-drop or click-to-upload image interface
- ⚡ Real-time predictions with confidence percentages
- 🏆 Top-3 breed predictions ranked by confidence
- 🎨 Modern, responsive UI with animated confidence bars
- ⚠️ Client- and server-side validation (file type, size limits)
- 🧹 Automatic cleanup of uploaded images after prediction

---

## Tech Stack

| Layer      | Technology                     |
|------------|---------------------------------|
| Backend    | Flask (Python)                  |
| ML Model   | TensorFlow / Keras (CNN)        |
| Frontend   | HTML, CSS, JavaScript (vanilla) |
| Image I/O  | Pillow, NumPy                   |

---

## Project Structure

```
Dog Breed Detection/
├── models/
│   ├── fine_tuned_inception.h5   # Fine-tuned InceptionV3 model
│   └── class_names.json          # Breed labels, in training order
├── templates/
│   └── index.html                # Frontend UI
├── uploads/                      # Temporary image storage (auto-created, gitignored)
├── app.py                        # Flask backend
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/harsh-tech-py/dog-breed-detection.git
cd dog-breed-detection
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your model files
Place the following inside the `models/` folder:
- `fine_tuned_inception.h5` — your fine-tuned InceptionV3 model
- `class_names.json` — already provided in this repo, containing the 8 trained breed classes in the correct order:
  ```json
  ["beagle", "bulldog", "dalmatian", "german-shepherd", "husky", "labrador-retriever", "poodle", "rottweiler"]
  ```

### 5. Run the app
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

---

## Model Details

- **Architecture:** Fine-tuned InceptionV3 (base frozen up to `mixed7`, custom dense head: Flatten → Dense(1024, ReLU) → Dropout(0.2) → Dense(8, Softmax))
- **Input size:** 224×224 RGB images
- **Output:** Softmax probabilities across 8 breed classes
- **Preprocessing:** Pixel values rescaled to `[0, 1]` (`img_array / 255.0`)
- **Classes:** `beagle`, `bulldog`, `dalmatian`, `german-shepherd`, `husky`, `labrador-retriever`, `poodle`, `rottweiler`

---

## API

### `POST /predict`

**Request:** `multipart/form-data` with a `file` field (PNG/JPG/JPEG, max 8MB)

**Response:**
```json
{
  "success": true,
  "predictions": [
    { "breed": "Labrador Retriever", "confidence": 92.4 },
    { "breed": "Beagle", "confidence": 5.1 },
    { "breed": "Rottweiler", "confidence": 1.3 }
  ]
}
```

---

## Future Improvements

- [ ] Deploy to a cloud platform (Render / Railway / AWS)
- [ ] Add breed information cards (temperament, size, origin)
- [ ] Support batch predictions for multiple images
- [ ] Add a confusion matrix / model evaluation page

---

## Author

**Harsh Vardhan**
GitHub: [@harsh-tech-py](https://github.com/harsh-tech-py)

---

## License

This project is open source and available under the [MIT License](LICENSE).