# Emotion Detection Web App

A Flask web application that uses a Vision Transformer (ViT) model to detect emotions from facial images captured via webcam.

## Features

- Real-time emotion detection using ViT model
- Webcam integration for live photo capture
- Face detection preprocessing for better accuracy
- Clean, responsive web interface
- REST API for emotion prediction

## Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python server.py
   ```

3. **Open your browser to** `http://localhost:5000`

## Deployment to Render

### Option 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**

2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Click "New" → "Blueprint" or "Web Service"
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` file

3. **Configure the service:**
   - Name: `emotion-detection-app`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT wsgi:app`

### Option 2: Manual Configuration

If render.yaml doesn't work, configure manually in Render dashboard:

- **Runtime:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn --bind 0.0.0.0:$PORT wsgi:app`
- **Environment Variables:**
  - `PYTHON_VERSION`: `3.13.0`

## Model Information

- **Model:** ViT (Vision Transformer) from Hugging Face
- **Source:** `dima806/facial_emotions_image_detection`
- **Emotions Detected:** sad, disgust, angry, neutral, fear, surprise, happy
- **Accuracy:** ~67% on test dataset

## API Usage

### Predict Emotion
```bash
curl -X POST -F "image=@photo.jpg" https://your-app-url/predict
```

**Response:**
```json
{
  "success": true,
  "emotion": "happy",
  "confidence": 0.85
}
```

## Troubleshooting

### Common Render Deployment Issues

1. **Model Download Timeout:**
   - The ViT model (~340MB) downloads on first startup
   - Increase timeout in Render dashboard if needed

2. **Memory Issues:**
   - Free tier has 512MB RAM limit
   - Consider upgrading to paid tier for ML models

3. **Port Binding:**
   - Render automatically sets the `PORT` environment variable
   - The app binds to `0.0.0.0:$PORT`

### Local Testing

Test the production setup locally:
```bash
# Set production environment
export RENDER=true
export PORT=8000

# Run with gunicorn
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

## Project Structure

```
week5/
├── server.py          # Flask application
├── wsgi.py           # WSGI entry point for gunicorn
├── train.py          # Model training script
├── requirements.txt  # Python dependencies
├── render.yaml       # Render deployment config
├── pyproject.toml    # Project metadata
├── templates/        # HTML templates
├── static/          # CSS/JS assets
└── data/            # Training data (if any)
```