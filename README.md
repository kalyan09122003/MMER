# 🎭 MMER — Multimodal Emotion Recognition

Welcome to **MMER (Multimodal Emotion Recognition)** — a unified AI system that detects emotions from **text, audio, image, and video** 

---

## 📁 Folder Structure

The repository is organized as follows:

```
MMER/
├── app.py
├── audio_emotion.py
├── image_emotion.py
├── text_emotion.py
├── video_emotion.py
├── text_train.py
├── requirements.txt
├── models/
│   ├── text_model.h5
│   ├── audio_model.h5
│   ├── image_model.h5
│   ├── video_model.h5
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── index.html
│   ├── text.html
│   ├── audio.html
│   ├── image.html
│   └── video.html
```

### 🧩 File Descriptions

* **`app.py`** – Main Flask application that connects all modalities.
* **`text_emotion.py`** – Handles text emotion prediction using `text_model.h5`.
* **`audio_emotion.py`** – Extracts MFCCs and predicts emotion using `audio_model.h5`.
* **`image_emotion.py`** – Performs facial emotion recognition using `image_model.h5`.
* **`video_emotion.py`** – Analyzes sampled video frames and predicts overall emotion using `video_model.h5`.
* **`text_train.py`** – Script for training and saving the text-based emotion model.
* **`requirements.txt`** – Contains all dependencies.
* **`models/`** – Folder containing all pre-trained `.h5` models.
* **`static/` & `templates/`** – Frontend assets (CSS, JS, HTML templates).

---

## 🚀 Features

✅ **Multimodal Emotion Recognition**

* Text, audio, image, and video support
* Deep learning models (`.h5` loaded via TensorFlow/Keras)

✅ **Modular Architecture**

* Each modality handled by a dedicated Python module
* Easy to update or replace individual models

✅ **Flask-Based Web Interface**

* Upload files or enter text directly in browser
* Real-time emotion predictions with confidence levels

✅ **Scalable Design**

* Add new modalities easily (e.g., physiological sensors, EEG)
* Ready for future API or mobile app integration

---

## ⚙️ Getting Started

Follow these steps to run the MMER app locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kalyan09122003/MMER.git
cd MMER
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add Your Trained Models

Place your `.h5` model files inside the `models/` folder:

```
models/
├── text_model.h5
├── audio_model.h5
├── image_model.h5
└── video_model.h5
```

### 4️⃣ Run the Flask Application

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

## 🧠 Model Details

| Modality  | Model File       | Input Type      | Preprocessing                   |
| --------- | ---------------- | --------------- | ------------------------------- |
| **Text**  | `text_model.h5`  | Raw text        | Tokenization + Padding          |
| **Audio** | `audio_model.h5` | `.wav`, `.webm` | MFCC Extraction                 |
| **Image** | `image_model.h5` | `.jpg`, `.png`  | Face detection + Resize (48x48) |
| **Video** | `video_model.h5` | `.mp4`, `.avi`  | Frame sampling + Resize (48x48) |

---

## 🧩 API Endpoints

| Endpoint         | Method | Input                                    | Output                         |
| ---------------- | ------ | ---------------------------------------- | ------------------------------ |
| `/predict_text`  | POST   | JSON (`{"text": "I'm so happy today!"}`) | Emotion label + probabilities  |
| `/predict_audio` | POST   | Audio file                               | Emotion label + probabilities  |
| `/predict_image` | POST   | Image file                               | Emotion label + probabilities  |
| `/video`         | POST   | Video file                               | Dominant emotion across frames |

---



## 🤝 Contributing

We welcome contributions!
To contribute:

1. **Fork** this repository
2. **Create** your feature branch

   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit** your changes

   ```bash
   git commit -am "Add YourFeature"
   ```
4. **Push** to your branch

   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

---

## 📜 License

This project is released under the **MIT License**.
Feel free to use, modify, and distribute for educational and research purposes.

---



---
