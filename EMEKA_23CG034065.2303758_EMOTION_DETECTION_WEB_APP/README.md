# 🎭 Emotion Detection Web Application

A comprehensive AI-powered web application that detects human emotions from images and live webcam captures using deep learning and computer vision.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Model Training](#model-training)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Database Schema](#database-schema)
- [Technologies Used](#technologies-used)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a real-time emotion detection system that can identify seven different emotions:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😮 Surprise
- 😐 Neutral

The application uses a Convolutional Neural Network (CNN) trained on facial expressions to predict emotions with high accuracy.

## ✨ Features

- **Image Upload Detection**: Upload images and detect emotions from photos
- **Live Webcam Capture**: Real-time emotion detection using your webcam
- **User Management**: Track who used the application
- **Database Storage**: Store all detection results with timestamps
- **Detection History**: View recent emotion detection results
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Visual Feedback**: Color-coded results with emotion emojis
- **Confidence Scores**: Shows prediction confidence percentages

## 📁 Project Structure

```
STUDENT_MAT.12345678/
│
├── app.py                          # Flask backend application
├── emotion_model_training.py       # Model training script
├── emotion_model.h5                # Trained model file (generated)
├── emotion_detection.db            # SQLite database (auto-created)
├── requirements.txt                # Python dependencies
├── link_to_my_web_app.txt         # Hosting information
├── README.md                       # This file
│
├── templates/
│   └── index.html                  # Frontend HTML interface
│
├── static/
│   └── style.css                   # CSS styling
│
└── uploads/                        # Stored detection images (auto-created)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (for live capture feature)
- Git (optional, for version control)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd STUDENT_MAT.12345678

# Or simply download and extract the folder
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues installing TensorFlow or OpenCV, try:

```bash
pip install --upgrade pip
pip install tensorflow==2.13.0
pip install opencv-python==4.8.0.76
```

## 🧠 Model Training

### Option 1: Use Pre-configured Model Architecture

The quickest way to get started:

```bash
python emotion_model_training.py
# Select option 2 when prompted
```

This creates the model architecture. For production use, you'll need to train it with actual data.

### Option 2: Train with FER2013 Dataset (Recommended)

1. **Download the FER2013 Dataset**:
   - Visit: https://www.kaggle.com/datasets/msambare/fer2013
   - Download and extract to the project root
   - Ensure folder structure: `fer2013/train/` and `fer2013/test/`

2. **Train the Model**:
   ```bash
   python emotion_model_training.py
   # Select option 1 when prompted
   ```

3. **Training Parameters**:
   - Image Size: 48x48 pixels (grayscale)
   - Batch Size: 64
   - Epochs: 50 (with early stopping)
   - Optimizer: Adam (learning rate: 0.0001)

4. **Output Files**:
   - `emotion_model.h5` - Final trained model
   - `emotion_model_best.h5` - Best model checkpoint
   - `training_history.png` - Training/validation curves

### Model Architecture

```
Input Layer: 48x48x1 (grayscale image)
├── Conv2D Block 1: 64 filters
├── Conv2D Block 2: 128 filters
├── Conv2D Block 3: 256 filters
├── Conv2D Block 4: 512 filters
├── Flatten Layer
├── Dense Layer: 512 units
├── Dense Layer: 256 units
└── Output Layer: 7 units (softmax)
```

## 🖥️ Running the Application

### Start the Flask Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

### Default Configuration

- **Host**: 0.0.0.0 (accessible from network)
- **Port**: 5000
- **Debug Mode**: Enabled (disable for production)

## 📖 Usage Guide

### 1. Image Upload Mode

1. Enter your name in the input field
2. Click "Choose Image" button
3. Select an image file (JPG, PNG, etc.)
4. The app will automatically detect and display the emotion
5. Results are saved to the database

### 2. Webcam Capture Mode

1. Enter your name in the input field
2. Click "Start Webcam" button
3. Allow browser to access your webcam
4. Position your face in the camera view
5. Click "Capture & Detect" button
6. View the detected emotion and result
7. Results are saved to the database

### 3. View Detection History

1. Scroll to the "Recent Detections" section
2. Click "Load History" button
3. View the last 10 detection records
4. Each record shows: Name, Emotion, Confidence, Type, Timestamp

## 🗄️ Database Schema

The application uses SQLite with the following schema:

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    image_path TEXT,
    emotion TEXT NOT NULL,
    confidence REAL,
    detection_type TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Database File

- **Location**: `emotion_detection.db` (project root)
- **Auto-created**: Yes (on first run)
- **Backup**: Recommended for production use

### Querying the Database

```bash
sqlite3 emotion_detection.db

# View all detections
SELECT * FROM detections;

# View recent detections
SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10;

# Count detections by emotion
SELECT emotion, COUNT(*) FROM detections GROUP BY emotion;
```

## 🛠️ Technologies Used

### Backend
- **Flask 2.3.3** - Web framework
- **TensorFlow 2.13.0** - Deep learning framework
- **Keras 2.13.1** - Neural network API
- **OpenCV 4.8.0** - Computer vision library
- **SQLite3** - Database

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **JavaScript** - Interactivity
- **WebRTC** - Webcam access

### Machine Learning
- **CNN Architecture** - Custom emotion detection model
- **Haar Cascade** - Face detection
- **ImageDataGenerator** - Data augmentation

## 🌐 Deployment

### Option 1: Render.com (Recommended)

1. Create account at https://render.com
2. Connect your GitHub repository
3. Create a new Web Service
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Deploy and get your public URL

### Option 2: Heroku

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: PythonAnywhere

1. Upload project files
2. Configure virtual environment
3. Set up WSGI configuration
4. Access at `https://yourusername.pythonanywhere.com`

### Option 4: Local Network (Development)

```bash
python app.py
# Access from other devices: http://your-ip:5000
```

### Option 5: Ngrok (Temporary Public URL)

```bash
# Terminal 1
python app.py

# Terminal 2
ngrok http 5000
# Use the provided public URL
```

## 📸 Screenshots

### Main Interface
- Clean, modern design with gradient backgrounds
- Two detection modes: Upload and Webcam
- User-friendly input fields

### Detection Results
- Annotated images with bounding boxes
- Emotion labels with confidence scores
- Large emoji representations
- Color-coded feedback

### Detection History
- Tabular view of recent detections
- Sortable columns
- Real-time updates

## 🔧 Troubleshooting

### Model Not Found Error

```
Warning: emotion_model.h5 not found. Please train the model first.
```

**Solution**: Run the training script:
```bash
python emotion_model_training.py
```

### Webcam Access Denied

**Solution**: 
- Check browser permissions (camera access)
- Use HTTPS in production (required by some browsers)
- Try different browser (Chrome recommended)

### Face Not Detected

**Possible causes**:
- Poor lighting conditions
- Face too far from camera
- Face partially obscured
- Low image quality

**Solutions**:
- Improve lighting
- Move closer to camera
- Ensure face is clearly visible
- Use higher quality images

### Installation Issues

**TensorFlow Installation**:
```bash
pip install tensorflow==2.13.0 --no-cache-dir
```

**OpenCV Installation**:
```bash
pip install opencv-python-headless
# Use headless version if GUI not needed
```

### Port Already in Use

```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Database Locked Error

**Solution**: 
- Close other connections to the database
- Restart the application
- Check file permissions

## 📊 Performance Tips

### Model Optimization
- Use model quantization for faster inference
- Implement model caching
- Use TensorFlow Lite for mobile deployment

### Application Optimization
- Implement image compression
- Add request rate limiting
- Use CDN for static files
- Enable gzip compression

### Database Optimization
- Add indexes on frequently queried columns
- Implement pagination for large datasets
- Regular database maintenance

## 🔒 Security Considerations

### For Production Deployment

1. **Secret Key**: Change the Flask secret key
   ```python
   app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
   ```

2. **File Upload**: Validate file types and sizes
3. **Database**: Use PostgreSQL for production
4. **HTTPS**: Always use SSL/TLS in production
5. **Input Validation**: Sanitize all user inputs
6. **CORS**: Configure Cross-Origin Resource Sharing
7. **Rate Limiting**: Implement to prevent abuse

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is created for educational purposes as part of CSC334 coursework.

## 👨‍💻 Author

**Student Name**: [Your Name]
**Matriculation Number**: [Your Matric Number]
**Course**: CSC334 - Artificial Intelligence
**Institution**: [Your Institution Name]

## 🙏 Acknowledgments

- FER2013 Dataset creators
- TensorFlow and Keras teams
- Flask framework developers
- OpenCV community
- Course instructors and teaching assistants

## 📞 Support

For issues or questions:
- Check the Troubleshooting section
- Review closed issues on GitHub
- Contact course instructor
- Email: [your-email@example.com]

## 🔄 Version History

### Version 1.0.0 (Current)
- Initial release
- Image upload detection
- Webcam capture detection
- SQLite database integration
- Responsive web interface
- Detection history viewer

### Planned Features
- Multiple face detection
- Emotion intensity analysis
- Export detection reports
- User authentication
- Real-time video stream analysis
- Mobile app version

---

**Note**: Remember to rename the folder from `STUDENT_MAT.12345678` to your actual surname and matriculation number in the format: `SURNAME_MAT.XXXXXX`

**Happy Emotion Detecting! 🎭😊**