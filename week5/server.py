from flask import Flask, render_template, request, jsonify
from transformers import ViTImageProcessor, ViTForImageClassification, pipeline
import torch
import numpy as np
import io
import cv2
from PIL import Image

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Load the ViT model and processor
model = None
processor = None
emotion_labels = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']  # ViT model order

# Load Haar cascade for face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    # Fallback if cv2.data is not available
    face_cascade = None
    print("Warning: Haar cascade not loaded, face detection disabled")

def load_emotion_model():
    global model, processor
    try:
        # Load the ViT model and processor from Hugging Face
        model_name = "dima806/facial_emotions_image_detection"
        model = ViTForImageClassification.from_pretrained(model_name)
        processor = ViTImageProcessor.from_pretrained(model_name)
        print("ViT model loaded successfully")
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        model = None
        processor = None

def detect_and_crop_face(image):
    """
    Detect face in PIL image and crop to face region
    Returns cropped PIL image or original if no face found
    """
    # Convert PIL to numpy array for OpenCV
    img_array = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        # Use the largest face found
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        # Add some padding around the face
        padding = int(0.1 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)

        # Crop the face
        face_crop = img_array[y:y+h, x:x+w]
        # Convert back to PIL
        return Image.fromarray(face_crop)

    # Return original image if no face detected
    return image

# Load model at startup
load_emotion_model()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or processor is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    try:
        # Get the image from the request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        # Read image from bytes
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Try face detection if available
        if face_cascade is not None:
            try:
                image = detect_and_crop_face(image)
            except Exception as e:
                print(f"Face detection failed, using original image: {e}")

        # Process image with ViT processor
        inputs = processor(images=image, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()

        emotion = emotion_labels[predicted_class]

        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    # Check if we're in production (Render sets this)
    import os
    is_production = os.environ.get('RENDER') == 'true'
    
    if is_production:
        # Production settings - let gunicorn handle the server
        print("Running in production mode with gunicorn")
    else:
        # Development settings
        app.run(debug=True)
