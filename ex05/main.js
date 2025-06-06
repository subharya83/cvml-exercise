// Constants
const MAX_GESTURE_DELAY = 500; // General detection interval
const MAX_FACE_DELAY = 5000; // Face visibility wait interval
const AUDIO_COOLDOWN = 5000; // 5 seconds

// Models and Video Element
let handposeModel;
let blazefaceModel;
let video;
let lastFaceDetected = true;
let userLeftTimeout = null;

// Audio Elements
const audioFiles = {
    thumbsUp: new Audio('/audio/thumbs_up.mp3'),
    handRaised: new Audio('/audio/hand_raised.mp3'),
    userLeft: new Audio('/audio/user_left.mp3'),
    thumbsDown: new Audio('/audio/thumbs_down.mp3'),
    happy: new Audio('/audio/happy.mp3'),
    sad: new Audio('/audio/sad.mp3'),
    surprised: new Audio('/audio/surprised.mp3'),
    angry: new Audio('/audio/angry.mp3'),
    disgusted: new Audio('/audio/disgusted.mp3'),
    fearful: new Audio('/audio/fearful.mp3'),
    neutral: new Audio('/audio/neutral.mp3')
};

let lastAudioPlay = {
    thumbsUp: 0,
    handRaised: 0,
    userLeft: 0,
    thumbsDown: 0,
    happy: 0,
    sad: 0,
    surprised: 0,
    angry: 0,
    disgusted: 0,
    fearful: 0,
    neutral: 0
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
        handposeModel = await handpose.load();
        blazefaceModel = await blazeface.load();
        await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
        await faceapi.nets.faceExpressionNet.loadFromUri('/models');
        console.log('Models loaded successfully');
    } catch (error) {
        console.error('Error loading models:', error);
        showAlert('Error loading detection models', null);
    }
}

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

        // Facial Expression Detection
        const expressions = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceExpressions();
        
        if (expressions.length > 0) {
            const expression = expressions[0].expressions;
            const maxExpression = Object.keys(expression).reduce((a, b) => expression[a] > expression[b] ? a : b);

            switch (maxExpression) {
                case 'happy':
                    showAlert('😊 Happy detected!', 'happy');
                    break;
                case 'sad':
                    showAlert('😢 Sad detected!', 'sad');
                    break;
                case 'angry':
                    showAlert('😠 Angry detected!', 'angry');
                    break;
                case 'surprised':
                    showAlert('😮 Surprised detected!', 'surprised');
                    break;
                case 'disgusted':
                    showAlert('🤢 Disgusted detected!', 'disgusted');
                    break;
                case 'fearful':
                    showAlert('😨 Fearful detected!', 'fearful');
                    break;
                case 'neutral':
                    showAlert('😐 Neutral detected!', 'neutral');
                    break;
                default:
                    break;
            }
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
        detectGestures();
    } catch (error) {
        console.error('Error initializing:', error);
        showAlert('Error initializing camera or models', null);
    }
}

init();