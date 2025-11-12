# 🚀 QUICKSTART GUIDE - Emotion Detection Web App

Get your emotion detection app up and running in 5 minutes!

## ⚡ Fast Track Setup

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd STUDENT_MAT.12345678

# Install required packages
pip install -r requirements.txt
```

**Note**: If you encounter errors, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Step 2: Verify Setup (30 seconds)

```bash
python setup_and_test.py
```

This script will:
- ✓ Check Python version
- ✓ Verify all dependencies
- ✓ Initialize database
- ✓ Test configuration

### Step 3: Create Model (1 minute)

```bash
python emotion_model_training.py
```

When prompted, choose **Option 2** for quick setup:
- Creates model architecture
- Ready to use immediately
- No dataset download required

**For production**: Choose Option 1 and train with FER2013 dataset

### Step 4: Run the App (30 seconds)

```bash
python app.py
```

You'll see:
```
EMOTION DETECTION WEB APPLICATION
==================================================
Starting Flask server...
Access the app at: http://localhost:5000
==================================================
```

### Step 5: Open Browser

Navigate to: **http://localhost:5000**

## 🎯 Quick Usage

### Upload an Image
1. Enter your name
2. Click "Choose Image"
3. Select a photo with a face
4. View emotion result instantly!

### Use Webcam
1. Enter your name
2. Click "Start Webcam"
3. Allow camera access
4. Click "Capture & Detect"
5. See your emotion detected!

## 🔧 Troubleshooting Quick Fixes

### Problem: Model not found
```bash
python emotion_model_training.py
# Choose option 2
```

### Problem: Port already in use
Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Problem: Webcam not working
- Allow camera permissions in browser
- Try Chrome or Firefox
- Check if another app is using camera

### Problem: Package installation fails
```bash
# Try one at a time
pip install flask
pip install tensorflow
pip install opencv-python
pip install numpy pillow
```

## 📁 What You Need

### Minimum Files to Run:
- ✓ `app.py` - Backend server
- ✓ `templates/index.html` - Frontend
- ✓ `emotion_model.h5` - Trained model
- ✓ `emotion_detection.db` - Database (auto-created)

### Optional Files:
- `static/style.css` - Enhanced styling
- `emotion_model_training.py` - For retraining
- `setup_and_test.py` - Verification script

## 🌟 First-Time Demo

Want to test without training? The model architecture will still load, but you'll need actual trained weights for accurate predictions.

**Quick Demo Steps:**
1. Run `python emotion_model_training.py` → Choose option 2
2. Run `python app.py`
3. Open http://localhost:5000
4. Try uploading test images
5. Note: Predictions may not be accurate without training data

## 📊 Sample Commands Reference

```bash
# Setup
pip install -r requirements.txt
python setup_and_test.py

# Create/Train Model
python emotion_model_training.py

# Run Application
python app.py

# Access Database
sqlite3 emotion_detection.db
SELECT * FROM detections;

# Check Database Records
python -c "import sqlite3; conn = sqlite3.connect('emotion_detection.db'); print(conn.execute('SELECT COUNT(*) FROM detections').fetchone()[0]); conn.close()"
```

## 🎓 For Submission

Before submitting, ensure you have:

1. **Rename folder** to your actual details:
   ```
   SURNAME_MAT.XXXXXX
   ```

2. **Verify all files exist**:
   - ✓ app.py
   - ✓ emotion_model_training.py
   - ✓ templates/index.html
   - ✓ static/style.css
   - ✓ requirements.txt
   - ✓ link_to_my_web_app.txt
   - ✓ emotion_model.h5
   - ✓ emotion_detection.db

3. **Update hosting link**:
   - Edit `link_to_my_web_app.txt`
   - Add your deployment URL

4. **Test everything works**:
   ```bash
   python setup_and_test.py
   python app.py
   # Test both upload and webcam modes
   ```

## 🚀 Deploy Online (5 minutes)

### Using Render.com (Recommended - Free):

1. Push code to GitHub
2. Go to https://render.com
3. Create new "Web Service"
4. Connect repository
5. Build command: `pip install -r requirements.txt`
6. Start command: `gunicorn app:app`
7. Deploy!
8. Copy your public URL to `link_to_my_web_app.txt`

### Using Ngrok (Quick Demo):

```bash
# Terminal 1
python app.py

# Terminal 2
ngrok http 5000
# Copy the https URL
```

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Install dependencies | 2-3 minutes |
| Setup verification | 30 seconds |
| Create model | 1 minute |
| Run application | 30 seconds |
| Test features | 2 minutes |
| **TOTAL** | **~6 minutes** |

## 💡 Pro Tips

1. **Use virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```

2. **Keep terminal open**: Flask runs in foreground
3. **Camera permissions**: Allow when prompted
4. **Internet needed**: Only for initial package installation
5. **Model training**: Optional for quick demo, required for accuracy

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Run `python setup_and_test.py` for diagnostics
3. Review error messages carefully
4. Check Python version (need 3.8+)

## ✅ Success Checklist

Before you're done, verify:

- [ ] All packages installed successfully
- [ ] Database created and accessible
- [ ] Model file exists (emotion_model.h5)
- [ ] App runs without errors
- [ ] Can access http://localhost:5000
- [ ] Image upload works
- [ ] Webcam capture works
- [ ] History shows saved detections
- [ ] Folder renamed to your details
- [ ] Hosting link updated (if deployed)

## 🎉 You're Ready!

Your emotion detection app is now running!

**Next Steps:**
- Try different facial expressions
- Test with multiple images
- Check the detection history
- Deploy online for remote access
- Train with real data for better accuracy

---

**Remember**: This is a CSC334 project. Make sure all required files are included in your submission!

**Happy Detecting! 😊🎭**