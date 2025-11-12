"""
Emotion Detection Web Application - Backend
Flask application for detecting human emotions from images and live webcam capture
"""

from flask import Flask, render_template, request, jsonify, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io
import os
from datetime import datetime
import sqlite3
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Global variables for model and face detector
model = None
face_cascade = None

# Database setup
DATABASE = 'emotion_detection.db'

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT,
            emotion TEXT NOT NULL,
            confidence REAL,
            detection_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def save_to_database(name, image_path, emotion, confidence, detection_type):
    """Save detection result to database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (name, image_path, emotion, confidence, detection_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, image_path, emotion, confidence, detection_type))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

def load_emotion_model():
    """Load the trained emotion detection model"""
    global model, face_cascade
    
    try:
        # Load the emotion detection model
        if os.path.exists('emotion_model.h5'):
            model = load_model('emotion_model.h5')
            print("Emotion detection model loaded successfully!")
        else:
            print("Warning: emotion_model.h5 not found. Please train the model first.")
            model = None
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier")
        else:
            print("Face cascade loaded successfully!")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def detect_emotion(image):
    """
    Detect emotion from an image
    Returns: emotion label, confidence, and annotated image
    """
    if model is None:
        return None, 0, image, "Model not loaded"
    
    if face_cascade is None:
        return None, 0, image, "Face detector not loaded"
    
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, 0, image, "No face detected"
        
        # Process the first detected face
        (x, y, w, h) = faces[0]
        
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to model input size (48x48)
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize pixel values
        face_roi = face_roi.astype('float32') / 255.0
        
        # Reshape for model input
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        
        # Predict emotion
        predictions = model.predict(face_roi, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = EMOTIONS[emotion_idx]
        
        # Add emotion text to image
        text = f"{emotion}: {confidence*100:.1f}%"
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (0, 255, 0), 2)
        
        return emotion, confidence, image, None
        
    except Exception as e:
        return None, 0, image, f"Error: {str(e)}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect_from_upload', methods=['POST'])
def detect_from_upload():
    """Handle image upload and detect emotion"""
    try:
        print("\n=== Upload Detection Request ===")
        
        # Get user name
        name = request.form.get('name', 'Anonymous')
        print(f"User name: {name}")
        
        # Check if image was uploaded
        if 'image' not in request.files:
            print("ERROR: No image in request.files")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        print(f"Image bytes read: {len(image_bytes)} bytes")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("ERROR: Failed to decode image")
            return jsonify({'error': 'Invalid image file'}), 400
        
        print(f"Image decoded successfully: {image.shape}")
        
        # Detect emotion
        print("Starting emotion detection...")
        emotion, confidence, annotated_image, error = detect_emotion(image)
        
        if error:
            print(f"ERROR: {error}")
            return jsonify({'error': error}), 400
        
        print(f"Emotion detected: {emotion} ({confidence*100:.2f}%)")
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, annotated_image)
        print(f"Image saved to: {filepath}")
        
        # Save to database
        db_success = save_to_database(name, filepath, emotion, confidence, 'upload')
        print(f"Database save: {'success' if db_success else 'failed'}")
        
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"Image encoded to base64: {len(img_base64)} characters")
        
        response_data = {
            'success': True,
            'emotion': emotion,
            'confidence': f"{confidence*100:.2f}",
            'image': img_base64,
            'message': f'Detected emotion: {emotion} ({confidence*100:.1f}% confidence)'
        }
        print("Sending success response")
        print("=== Request Complete ===\n")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"EXCEPTION in detect_from_upload: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=== Request Failed ===\n")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_from_webcam', methods=['POST'])
def detect_from_webcam():
    """Handle webcam capture and detect emotion"""
    try:
        print("\n=== Webcam Detection Request ===")
        
        # Get user name and image data
        data = request.get_json()
        name = data.get('name', 'Anonymous')
        image_data = data.get('image', '')
        print(f"User name: {name}")
        print(f"Image data length: {len(image_data)}")
        
        if not image_data:
            print("ERROR: No image data received")
            return jsonify({'error': 'No image data received'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        print(f"Decoded {len(image_bytes)} bytes")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("ERROR: Failed to decode image")
            return jsonify({'error': 'Invalid image data'}), 400
        
        print(f"Image decoded successfully: {image.shape}")
        
        # Detect emotion
        print("Starting emotion detection...")
        emotion, confidence, annotated_image, error = detect_emotion(image)
        
        if error:
            print(f"ERROR: {error}")
            return jsonify({'error': error}), 400
        
        print(f"Emotion detected: {emotion} ({confidence*100:.2f}%)")
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_webcam_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, annotated_image)
        print(f"Image saved to: {filepath}")
        
        # Save to database
        db_success = save_to_database(name, filepath, emotion, confidence, 'webcam')
        print(f"Database save: {'success' if db_success else 'failed'}")
        
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"Image encoded to base64: {len(img_base64)} characters")
        
        response_data = {
            'success': True,
            'emotion': emotion,
            'confidence': f"{confidence*100:.2f}",
            'image': img_base64,
            'message': f'Detected emotion: {emotion} ({confidence*100:.1f}% confidence)'
        }
        print("Sending success response")
        print("=== Request Complete ===\n")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"EXCEPTION in detect_from_webcam: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=== Request Failed ===\n")
        return jsonify({'error': str(e)}), 500

@app.route('/get_history')
def get_history():
    """Get detection history from database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, emotion, confidence, detection_type, timestamp
            FROM detections
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'name': row[0],
                'emotion': row[1],
                'confidence': f"{row[2]*100:.2f}" if row[2] else "N/A",
                'type': row[3],
                'timestamp': row[4]
            })
        
        return jsonify({'history': history})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'running',
        'model_loaded': model is not None,
        'face_detector_loaded': face_cascade is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    print("=" * 50)
    print("EMOTION DETECTION WEB APPLICATION")
    print("=" * 50)
    
    # Initialize database
    init_db()
    
    # Load model
    load_emotion_model()
    
    # Run the app
    print("\nStarting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)