// Global variables
let video = null;
let canvas = null;
let stream = null;

// DOM elements
const startCameraBtn = document.getElementById('startCamera');
const capturePhotoBtn = document.getElementById('capturePhoto');
const stopCameraBtn = document.getElementById('stopCamera');
const resultContainer = document.getElementById('resultContainer');
const loadingIndicator = document.getElementById('loadingIndicator');

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');

    // Add event listeners
    startCameraBtn.addEventListener('click', startCamera);
    capturePhotoBtn.addEventListener('click', capturePhoto);
    stopCameraBtn.addEventListener('click', stopCamera);
});

// Start the camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });

        video.srcObject = stream;

        startCameraBtn.disabled = true;
        capturePhotoBtn.disabled = false;
        stopCameraBtn.disabled = false;

    } catch (error) {
        console.error('Camera error:', error);
        showError('Camera access denied');
    }
}

// Stop the camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;

        startCameraBtn.disabled = false;
        capturePhotoBtn.disabled = true;
        stopCameraBtn.disabled = true;
    }
}

// Capture photo from video stream
function capturePhoto() {
    if (!stream) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        if (blob) await sendImageToServer(blob);
    }, 'image/jpeg', 0.9);
}

// Send image to server for classification
async function sendImageToServer(imageBlob) {
    showLoading();

    const formData = new FormData();
    formData.append('image', imageBlob, 'capture.jpg');

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        hideLoading();

        if (data.success) {
            displayResult(data.emotion, data.confidence, imageBlob);
        } else {
            showError(data.error || 'Detection failed');
        }

    } catch (error) {
        console.error('Server error:', error);
        hideLoading();
        showError('Connection failed');
    }
}

// Display the emotion detection result
function displayResult(emotion, confidence, imageBlob) {
    const imageUrl = URL.createObjectURL(imageBlob);

    const resultHTML = `
        <div class="result-content">
            <img src="${imageUrl}" alt="Captured" class="captured-image">
            <div class="emotion-result">
                <div class="emotion-label">${emotion}</div>
                <div class="emotion-confidence">${(confidence * 100).toFixed(1)}% confident</div>
            </div>
        </div>
    `;

    resultContainer.innerHTML = resultHTML;
}

// Show loading indicator
function showLoading() {
    loadingIndicator.style.display = 'flex';
    resultContainer.style.display = 'none';
}

// Hide loading indicator
function hideLoading() {
    loadingIndicator.style.display = 'none';
    resultContainer.style.display = 'flex';
}

// Show error message
function showError(message) {
    const errorHTML = `<div class="error-message">${message}</div>`;
    resultContainer.innerHTML = errorHTML;
    resultContainer.style.display = 'flex';
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});