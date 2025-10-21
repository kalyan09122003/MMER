from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import librosa
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import speech_recognition as sr

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained models (.h5)
text_model = load_model('models/text_model.h5')
audio_model = load_model('models/audio_model.h5')
image_model = load_model('models/image_model.h5')
video_model = load_model('models/video_model.h5')

# Define emotion labels (match your model output order)
EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']

# ---------------- TEXT EMOTION ----------------
def detect_emotion_from_text(text):
    # Simple text preprocessing
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=5000)
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100, padding='post')

    preds = text_model.predict(padded)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {'label': emotion, 'probabilities': dict(zip(EMOTIONS, preds.tolist()))}


# ---------------- AUDIO EMOTION ----------------
def detect_emotion_from_audio(audio_path):
    # Extract MFCC features
    y, sr_rate = librosa.load(audio_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)

    preds = audio_model.predict(mfcc)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {'label': emotion, 'probabilities': dict(zip(EMOTIONS, preds.tolist()))}


# ---------------- IMAGE EMOTION ----------------
def detect_emotion_from_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))  # or your model’s input size
    image = np.expand_dims(image, axis=0) / 255.0

    preds = image_model.predict(image)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {'label': emotion, 'probabilities': dict(zip(EMOTIONS, preds.tolist()))}


# ---------------- VIDEO EMOTION ----------------
def detect_emotion_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // 10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % sample_rate == 0:
            frame = cv2.resize(frame, (48, 48))
            frames.append(frame)
    cap.release()

    if not frames:
        return {'label': 'neutral', 'probabilities': {'neutral': 1.0}}

    frames = np.array(frames) / 255.0
    preds = video_model.predict(frames)
    avg_pred = np.mean(preds, axis=0)
    emotion = EMOTIONS[np.argmax(avg_pred)]
    confidence = float(np.max(avg_pred))

    return {'label': emotion, 'probabilities': dict(zip(EMOTIONS, avg_pred.tolist()))}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text', methods=['GET'])
def text_page():
    return render_template('text.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Get the emotion recognition result
        result = detect_emotion_from_text(text)
        
        # Format the response to match frontend expectations
        response = {
            'label': result.get('emotion', 'neutral'),
            'probabilities': {
                'happy': 0,
                'sad': 0,
                'angry': 0,
                'fear': 0,
                'surprise': 0,
                'disgust': 0,
                'neutral': 0
            }
        }
        
        # Set the confidence for the detected emotion
        if 'emotion' in result and 'confidence' in result:
            emotion = result['emotion']
            confidence = result['confidence']
            response['probabilities'][emotion] = confidence
            
            # If confidence is low, also set some neutral probability
            if confidence < 0.7:
                response['probabilities']['neutral'] = 0.3
        
        print(f"Returning response: {response}")  # Debug log
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio', methods=['GET'])
def audio_page():
    return render_template('audio.html')

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get file extension
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        
        # Save the audio file with original extension
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_audio{file_ext}')
        audio_file.save(temp_path)
        
        # Process the audio
        result = detect_emotion_from_audio(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"Error in predict_audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'label': 'neutral',
            'probabilities': {'neutral': 1.0}
        }), 500
        
    finally:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temp file {temp_path}: {str(e)}")

@app.route('/image', methods=['GET'])
def image_page():
    return render_template('image.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    temp_path = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save the image temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        image_file.save(temp_path)
        
        # Process the image
        result = detect_emotion_from_image(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify(result)
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/video', methods=['GET', 'POST'])
def video_page():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save the video temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
        video_file.save(temp_path)
        
        try:
            # Process video - analyze multiple frames
            cap = cv2.VideoCapture(temp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Analyze up to 10 frames (or all frames if video is short)
            sample_rate = max(1, frame_count // 10)
            frame_emotions = []
            
            for i in range(0, frame_count, sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = cap.read()
                if not success:
                    continue
                
                # Save frame as image and process
                frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame_{i}.jpg')
                cv2.imwrite(frame_path, frame)
                
                # Get emotion for this frame
                result = detect_emotion_from_image(frame_path)
                if 'emotion' in result and 'confidence' in result:
                    frame_emotions.append((result['emotion'], result['confidence']))
                
                # Clean up
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            cap.release()
            
            if not frame_emotions:
                return jsonify({
                    'error': 'No faces detected in video',
                    'label': 'neutral',
                    'probabilities': {'neutral': 1.0}
                })
            
            # Calculate average emotion across frames
            emotion_scores = {}
            for emotion, confidence in frame_emotions:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0
                emotion_scores[emotion] += confidence
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            total_frames = len(frame_emotions)
            
            # Format response to match frontend expectations
            response = {
                'label': dominant_emotion[0],
                'probabilities': {
                    'happy': 0,
                    'sad': 0,
                    'angry': 0,
                    'fear': 0,
                    'surprise': 0,
                    'disgust': 0,
                    'neutral': 0
                },
                'faces': total_frames
            }
            
            # Set probabilities
            for emotion, score in emotion_scores.items():
                if emotion in response['probabilities']:
                    response['probabilities'][emotion] = score / total_frames
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'label': 'neutral',
                'probabilities': {'neutral': 1.0}
            }), 500
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return render_template('video.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
