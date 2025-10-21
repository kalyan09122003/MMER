from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import speech_recognition as sr
from fer import FER
import cv2
import numpy as np
import tempfile
import base64

app = Flask(__name__)
CORS(app)
app.static_folder = 'static'
app.static_url_path = '/static'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion recognition models
emotion_detector = FER(mtcnn=True)

# Emotion keywords mapping - using 'joy' instead of 'happy' to match frontend expectations
EMOTION_KEYWORDS = {
    'joy': ["great", "awesome", "happy", "good", "nice", "love", "joy", "over the moon", "on cloud nine", "delighted", "ecstatic", "excited"],
    'sad': ['sad', 'unhappy', 'depressed', 'miserable', 'sorrow', 'upset'],
    'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
    'fear': ['fear', 'scared', 'afraid', 'terrified', 'worried', 'anxious'],
    'surprise': ['surprise', 'surprised', 'amazed', 'shocked', 'astonished'],
    'disgust': ['disgust', 'disgusted', 'revolted', 'sickened', 'repulsed'],
    'neutral': ['okay', 'fine', 'normal', 'alright', 'neutral', 'meh']
}

def detect_emotion_from_text(text):
    """Detect emotion from text using keyword matching with enhanced scoring"""
    if not text or not isinstance(text, str):
        return {'emotion': 'neutral', 'confidence': 0.5}
        
    text_lower = text.lower()
    emotion_scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    word_count = len(text_lower.split())
    
    # Calculate base scores from keywords
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            # Count occurrences of the keyword
            count = text_lower.count(keyword)
            if count > 0:
                # Longer matches get higher weight
                emotion_scores[emotion] += count * (1 + len(keyword) * 0.1)
    
    # Add intensity modifiers for punctuation and capitalization
    if '!' in text:
        for emotion in ['happy', 'angry', 'surprise']:
            emotion_scores[emotion] *= 1.5
    
    # Normalize scores
    total_score = sum(emotion_scores.values()) or 1  # Avoid division by zero
    normalized_scores = {e: s/total_score for e, s in emotion_scores.items()}
    
    # Get the dominant emotion
    detected_emotion = max(normalized_scores, key=normalized_scores.get)
    confidence = normalized_scores[detected_emotion]
    
    # If confidence is too low, default to neutral
    if confidence < 0.3:
        detected_emotion = 'neutral'
        confidence = 0.5
    
    print(f"Detected emotion: {detected_emotion} with confidence: {confidence}")
    return {
        'emotion': detected_emotion,
        'confidence': min(max(float(confidence), 0.1), 1.0)  # Ensure confidence is between 0.1 and 1.0
    }

def convert_webm_to_wav(webm_path, wav_path):
    """Convert WebM audio to WAV format using ffmpeg"""
    try:
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', webm_path, '-ac', '1', '-ar', '16000', wav_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        return True
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return False

def detect_emotion_from_audio(audio_path):
    """Convert speech to text and detect emotion"""
    recognizer = sr.Recognizer()
    temp_wav = None
    
    try:
        # If the file is WebM, convert it to WAV first
        if audio_path.lower().endswith('.webm'):
            import tempfile
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            if not convert_webm_to_wav(audio_path, temp_wav):
                raise Exception("Failed to convert WebM to WAV format")
            audio_path = temp_wav
        
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for the data (load audio to memory)
            audio_data = recognizer.record(source)
            
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio_data, language='en-US')
            
            # Get text emotion recognition result
            text_result = detect_emotion_from_text(text)
            
            # Format to match frontend expectations
            response = {
                'text': text,  # Ensure text is included in the response
                'label': text_result.get('emotion', 'neutral'),
                'probabilities': {
                    'happy': 0.0,
                    'sad': 0.0,
                    'angry': 0.0,
                    'fear': 0.0,
                    'surprise': 0.0,
                    'disgust': 0.0,
                    'neutral': 0.0
                },
                'emotion': text_result.get('emotion', 'neutral'),
                'confidence': float(text_result.get('confidence', 0.5))
            }
            
            # Set the confidence for the detected emotion
            if 'emotion' in text_result and 'confidence' in text_result:
                emotion = text_result['emotion']
                confidence = text_result['confidence']
                response['probabilities'][emotion] = confidence
                
                # If confidence is low, also set some neutral probability
                if confidence < 0.7:
                    response['probabilities']['neutral'] = 0.3
            
            return response
            
    except sr.UnknownValueError:
        return {
            'error': 'Could not understand the audio. Please speak clearly and try again.',
            'label': 'neutral',
            'probabilities': {'neutral': 1.0}
        }
    except sr.RequestError as e:
        return {
            'error': 'Could not request results from Google Speech Recognition service',
            'label': 'neutral',
            'probabilities': {'neutral': 1.0}
        }
    except Exception as e:
        import traceback
        print(f"Error in detect_emotion_from_audio: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': f'Error processing audio: {str(e)}',
            'label': 'neutral',
            'probabilities': {'neutral': 1.0}
        }
    finally:
        # Clean up temporary WAV file if it was created
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                print(f"Error removing temporary WAV file: {str(e)}")

def detect_emotion_from_image(image_path):
    """Detect emotion from image using FER"""
    try:
        # Read the image
        image = cv2.imread(image_path)
        print(f"[DEBUG] Loaded image: {type(image)}, shape: {getattr(image, 'shape', None)}")
        if image is None:
            print("[DEBUG] cv2.imread returned None")
            return {
                'error': 'Could not read image file',
                'label': 'neutral',
                'probabilities': {'neutral': 1.0},
                'faces': 0
            }
        # Detect emotions
        result = emotion_detector.detect_emotions(image)
        print(f"[DEBUG] FER result: {result}")
        if not result:
            print("[DEBUG] No faces detected by FER")
            return {
                'label': 'neutral',
                'probabilities': {'neutral': 1.0},
                'faces': 0
            }
        # Aggregate emotions for all faces
        face_count = len(result)
        emotion_sums = {'happy': 0, 'sad': 0, 'angry': 0, 'fear': 0, 'surprise': 0, 'disgust': 0, 'neutral': 0}
        for face in result:
            for emotion, value in face['emotions'].items():
                if emotion in emotion_sums:
                    emotion_sums[emotion] += value
        # Average emotions
        emotion_avgs = {k: v / face_count for k, v in emotion_sums.items()}
        # Dominant emotion
        label = max(emotion_avgs.items(), key=lambda x: x[1])[0]
        response = {
            'label': label,
            'probabilities': emotion_avgs,
            'faces': face_count
        }
        print(f"[DEBUG] Response: {response}")
        return response
    except Exception as e:
        print(f"[ERROR] Exception in detect_emotion_from_image: {e}")
        return {'error': str(e), 'faces': 0}

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
