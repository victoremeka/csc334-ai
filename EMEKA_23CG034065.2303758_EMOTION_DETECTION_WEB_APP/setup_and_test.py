"""
Setup and Test Script for Emotion Detection Web Application
This script verifies installation, initializes the database, and tests model loading
"""

import sys
import os
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking required packages...")
    
    packages = [
        'flask',
        'tensorflow',
        'keras',
        'cv2',
        'numpy',
        'PIL',
        'sqlite3',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            installed_packages.append(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed")
        return True

def check_directory_structure():
    """Verify project directory structure"""
    print("\nChecking directory structure...")
    
    required_dirs = ['templates', 'static', 'uploads']
    required_files = [
        'app.py',
        'emotion_model_training.py',
        'requirements.txt',
        'link_to_my_web_app.txt',
        'templates/index.html',
        'static/style.css'
    ]
    
    all_ok = True
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory '{directory}' exists")
        else:
            print(f"✗ Directory '{directory}' is missing")
            os.makedirs(directory, exist_ok=True)
            print(f"  Created directory '{directory}'")
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ File '{file}' exists")
        else:
            print(f"✗ File '{file}' is missing")
            all_ok = False
    
    return all_ok

def initialize_database():
    """Initialize the SQLite database"""
    print("\nInitializing database...")
    
    try:
        import sqlite3
        
        DATABASE = 'emotion_detection.db'
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
        
        # Check if table was created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
        result = cursor.fetchone()
        
        if result:
            print("✓ Database initialized successfully")
            
            # Get record count
            cursor.execute("SELECT COUNT(*) FROM detections")
            count = cursor.fetchone()[0]
            print(f"  Current records in database: {count}")
        else:
            print("✗ Failed to create database table")
            conn.close()
            return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False

def check_model_file():
    """Check if model file exists"""
    print("\nChecking for trained model...")
    
    model_files = ['emotion_model.h5', 'emotion_model_best.h5']
    found = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✓ Model file '{model_file}' found ({size_mb:.2f} MB)")
            found = True
        else:
            print(f"✗ Model file '{model_file}' not found")
    
    if not found:
        print("\n⚠ No trained model found!")
        print("  To create/train a model, run:")
        print("  python emotion_model_training.py")
    
    return found

def test_model_loading():
    """Test loading the emotion detection model"""
    print("\nTesting model loading...")
    
    try:
        from tensorflow.keras.models import load_model
        import cv2
        
        # Try to load model
        if os.path.exists('emotion_model.h5'):
            model = load_model('emotion_model.h5')
            print("✓ Model loaded successfully")
            print(f"  Model input shape: {model.input_shape}")
            print(f"  Model output shape: {model.output_shape}")
            
            # Test face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if not face_cascade.empty():
                print("✓ Face detection cascade loaded successfully")
            else:
                print("✗ Failed to load face detection cascade")
                return False
            
            return True
        else:
            print("⚠ Model file not found - skipping load test")
            return False
            
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def create_test_record():
    """Create a test record in the database"""
    print("\nCreating test database record...")
    
    try:
        import sqlite3
        
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (name, image_path, emotion, confidence, detection_type)
            VALUES (?, ?, ?, ?, ?)
        ''', ('Test User', 'test/test.jpg', 'Happy', 0.95, 'test'))
        
        conn.commit()
        
        # Verify insertion
        cursor.execute("SELECT * FROM detections WHERE name='Test User' ORDER BY id DESC LIMIT 1")
        record = cursor.fetchone()
        
        if record:
            print("✓ Test record created successfully")
            print(f"  Record ID: {record[0]}")
            print(f"  Name: {record[1]}")
            print(f"  Emotion: {record[3]}")
            print(f"  Confidence: {record[4]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to create test record: {e}")
        return False

def generate_report(results):
    """Generate a summary report"""
    print_header("SETUP VERIFICATION REPORT")
    
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    print("Check Results:")
    print("-" * 70)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<50} {status}")
    
    print("-" * 70)
    print(f"\nTotal: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! Your application is ready to run.")
        print("\nTo start the application:")
        print("  python app.py")
        print("\nThen open your browser to:")
        print("  http://localhost:5000")
    else:
        print("\n⚠ Some checks failed. Please review the errors above.")
        print("   Fix the issues before running the application.")
    
    return passed_checks == total_checks

def main():
    """Main setup and test routine"""
    print_header("EMOTION DETECTION WEB APP - SETUP VERIFICATION")
    print("This script will verify your installation and setup.\n")
    
    results = {}
    
    # Run all checks
    results['Python Version'] = check_python_version()
    results['Dependencies'] = check_dependencies()
    results['Directory Structure'] = check_directory_structure()
    results['Database Initialization'] = initialize_database()
    results['Model File Check'] = check_model_file()
    
    # Optional tests (don't fail if model not present)
    if results['Model File Check']:
        results['Model Loading Test'] = test_model_loading()
    
    # Test database operations
    if results['Database Initialization']:
        results['Database Test Record'] = create_test_record()
    
    # Generate final report
    all_passed = generate_report(results)
    
    # Additional recommendations
    if not results.get('Model File Check', False):
        print("\n📝 RECOMMENDATION:")
        print("   Train or create the emotion detection model by running:")
        print("   python emotion_model_training.py")
    
    print("\n" + "=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)