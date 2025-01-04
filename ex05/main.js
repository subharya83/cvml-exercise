// Constants
const MAX_GESTURE_DELAY = 500;
const MAX_FACE_DELAY = 5000;
const AUDIO_COOLDOWN = 5000;
const EXPRESSION_CHECK_INTERVAL = 100;

// Models and Elements
let handposeModel;
let blazefaceModel;
let video;
let canvas;
let lastFaceDetected = true;
let userLeftTimeout = null;

// Status Icons for Expressions
const statusIcons = {
    default: { emoji: '😐', color: '#02c19c' },
    neutral: { emoji: '😐', color: '#54adad' },
    happy: { emoji: '😀', color: '#148f77' },
    sad: { emoji: '😥', color: '#767e7e' },
    angry: { emoji: '😠', color: '#b64518' },
    fearful: { emoji: '😨', color: '#90931d' },
    disgusted: { emoji: '🤢', color: '#1a8d1a' },
    surprised: { emoji: '😲', color: '#1230ce' }
};

// Audio Elements
const audioFiles = {
    thumbsUp: new Audio('/audio/thumbs_up.mp3'),
    handRaised: new Audio('/audio/hand_raised.mp3'),
    userLeft: new Audio('/audio/user_left.mp3'),
    thumbsDown: new Audio('/audio/thumbs_down.mp3')
};

let lastAudioPlay = {
    thumbsUp: 0,
    handRaised: 0,
    userLeft: 0,
    thumbsDown: 0
};

// Helper Functions
function canPlayAudio(audioType) {
    const now = Date.now();
    if (now - lastAudioPlay[audioType] > AUDIO_COOLDOWN) {
        lastAudioPlay[audioType] = now;
        return true;
    }
    return false;
}

function playAudio(audioType) {
    if (canPlayAudio(audioType)) {
        audioFiles[audioType].play().catch(e => console.error('Error playing audio:', e));
    }
}

function showAlert(message, audioType) {
    const alertsDiv = document.getElementById('alerts');
    const alert = document.createElement('div');
    alert.className = 'alert';
    alert.textContent = message;
    alertsDiv.appendChild(alert);

    if (audioType) {
        playAudio(audioType);
    }

    setTimeout(() => {
        alertsDiv.removeChild(alert);
    }, 500);
}

async function setupCamera() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing camera:', error);
        showAlert('Error accessing camera', null);
        throw error;
    }

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

async function loadModels() {
    try {
        // Load hand and face detection models
        handposeModel = await handpose.load();
        blazefaceModel = await blazeface.load();
        
        // Load face-api.js models
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
            faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
            faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
            faceapi.nets.faceExpressionNet.loadFromUri('./models')
        ]);
        
        console.log('All models loaded successfully');
    } catch (error) {
        console.error('Error loading models:', error);
        showAlert('Error loading detection models', null);
    }
}

// Existing gesture detection functions
function isHandRaised(predictions) {
    if (predictions.length > 0) {
        const wrist = predictions[0].annotations.palmBase[0];
        const fingers = [
            predictions[0].annotations.middleFinger[3],
            predictions[0].annotations.indexFinger[3],
            predictions[0].annotations.ringFinger[3],
            predictions[0].annotations.pinky[3]
        ];
        return fingers.every(finger => finger[1] < wrist[1] - 50);
    }
    return false;
}

function isThumbsUp(predictions) {
    if (predictions.length > 0) {
        const thumbTip = predictions[0].annotations.thumb[3];
        const indexBase = predictions[0].annotations.indexFinger[0];
        return thumbTip[1] < indexBase[1];
    }
    return false;
}

function isThumbsDown(predictions) {
    if (predictions.length > 0) {
        const thumbTip = predictions[0].annotations.thumb[3];
        const indexBase = predictions[0].annotations.indexFinger[0];
        const wrist = predictions[0].annotations.palmBase[0];
        const fingers = [
            predictions[0].annotations.middleFinger[3],
            predictions[0].annotations.ringFinger[3],
            predictions[0].annotations.pinky[3]
        ];
        return thumbTip[1] > indexBase[1] + 50 && fingers.every(finger => finger[1] > wrist[1] - 50);
    }
    return false;
}

// Expression detection setup
function setupExpressionDetection() {
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(
            video,
            new faceapi.TinyFaceDetectorOptions()
        ).withFaceExpressions();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

        updateExpressionUI(detections);
    }, EXPRESSION_CHECK_INTERVAL);
}

function updateExpressionUI(detections) {
    const emoji = document.getElementById('emoji');
    const textStatus = document.getElementById('textStatus');
    const app = document.getElementById('app');

    if (detections.length > 0) {
        detections.forEach((element) => {
            let status = '';
            let valueStatus = 0.0;
            
            for (const [key, value] of Object.entries(element.expressions)) {
                if (value > valueStatus) {
                    status = key;
                    valueStatus = value;
                }
            }

            emoji.innerHTML = statusIcons[status].emoji;
            textStatus.innerHTML = status;
            app.style.backgroundColor = statusIcons[status].color;
        });
    } else {
        emoji.innerHTML = statusIcons['default'].emoji;
        textStatus.innerHTML = '...';
        app.style.backgroundColor = statusIcons['default'].color;
    }
}

async function detectGestures() {
    try {
        const handPredictions = await handposeModel.estimateHands(video);

        if (isThumbsDown(handPredictions)) {
            showAlert('👎 Thumbs down detected!', 'thumbsDown');
        } else if (isThumbsUp(handPredictions)) {
            showAlert('👍 Thumbs up detected!', 'thumbsUp');
        } else if (isHandRaised(handPredictions)) {
            showAlert('✋ Hand raised detected!', 'handRaised');
        }

        const faces = await blazefaceModel.estimateFaces(video, false);
        const faceDetected = faces.length > 0;

        if (faceDetected !== lastFaceDetected) {
            if (!faceDetected) {
                userLeftTimeout = setTimeout(() => {
                    showAlert('⚠️ User left the frame!', 'userLeft');
                }, MAX_FACE_DELAY);
            } else {
                clearTimeout(userLeftTimeout);
            }
            lastFaceDetected = faceDetected;
        }
    } catch (error) {
        console.error('Error in detection:', error);
    }

    setTimeout(detectGestures, MAX_GESTURE_DELAY);
}

function preloadAudio() {
    Object.values(audioFiles).forEach(audio => audio.load());
}

async function init() {
    try {
        preloadAudio();
        await setupCamera();
        await loadModels();
        setupExpressionDetection();
        detectGestures();
    } catch (error) {
        console.error('Error initializing:', error);
        showAlert('Error initializing camera or models', null);
    }
}

init();