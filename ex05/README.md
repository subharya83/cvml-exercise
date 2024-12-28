# Situationally Aware Virtual evitaR (SAVR)


## Code Organization
```shell

├── app
│   ├── public
│   │   ├── index.html
│   │   ├── lib
│   │   │   ├── face-api.min.js
│   │   │   ├── posenet.min.js
│   │   │   ├── tf.min.js
│   │   │   └── webgazer.min.js
│   │   ├── models
│   │   │   ├── face_expression_model-shard1
│   │   │   ├── face_expression_model-weights_manifest.json
│   │   │   ├── face_landmark_68_model-shard1
│   │   │   ├── face_landmark_68_model-weights_manifest.json
│   │   │   ├── tiny_face_detector_model-shard1
│   │   │   └── tiny_face_detector_model-weights_manifest.json
│   │   └── script.js
│   └── server.js
├── Dockerfile
└── package.json

4 directories, 15 files

```
## Downloading dependencies

### Getting model weights
```shell
mkdir -p app/public/models
cd app/public/models

# Download face-api.js model files
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-shard1
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model-weights_manifest.json
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model-shard1
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_expression_model-weights_manifest.json
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_expression_model-shard1
```
### Getting other .js libraries
```shell
cd app/public/lib
wget https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js -O tf.min.js
wget https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet/dist/posenet.min.js -O posenet.min.js
wget https://cdn.jsdelivr.net/npm/face-api.js/dist/face-api.min.js -O face-api.min.js
wget https://webgazer.cs.brown.edu/webgazer.js -O webgazer.min.js
```

### Docker build and run
```shell
# TODO : Update this line after main branch integration
git checkout UI-DS-6f388e08 
cd savr-v1
docker build -t savr .
docker run -p 3000:3000 savr
```
### Test client
```shell
curl localhost:3000
<!DOCTYPE html>
<html>
<head>
    <title>ML Webcam Tracker</title>
    <!-- Load dependencies from local files -->
    <script src="/lib/tf.min.js"></script>
    <script src="/lib/posenet.min.js"></script>
    <script src="/lib/face-api.min.js"></script>
    <script src="/lib/webgazer.min.js"></script>
    <style>
        .container {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            padding: 20px;
        }
        .video-container {
            position: relative;
            width: 640px;
            height: 480px;
        }
        #videoElement {
            position: absolute;
            border: 2px solid #333;
            border-radius: 8px;
        }
        #faceOverlay, #gazeOverlay, #poseOverlay {
            position: absolute;
            top: 0;
            left: 0;
        }
        .metrics {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            z-index: 1000;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
        }
        button:hover {
            background-color: #45a049;
        }
        .loading {
            color: white;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 8px;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div class="loading" id="loadingIndicator">Loading ML models...</div>
    <div class="container">
        <div class="video-container">
            <video id="videoElement" width="640" height="480" autoplay></video>
            <canvas id="faceOverlay" width="640" height="480"></canvas>
            <canvas id="gazeOverlay" width="640" height="480"></canvas>
            <canvas id="poseOverlay" width="640" height="480"></canvas>
            <div class="metrics" id="metrics"></div>
        </div>
        <button id="submitButton" disabled>Submit</button>
    </div>
    <script src="script.js"></script>
</body>
</html>
```

### Test server response
```shell
$ docker container ls
CONTAINER ID   IMAGE     COMMAND                  CREATED          STATUS          PORTS                                       NAMES
9ea4df366504   savr   "docker-entrypoint.s…"   25 minutes ago   Up 25 minutes   0.0.0.0:3000->3000/tcp, :::3000->3000/tcp   jovial_kapitsa

$ docker exec -it 9ea4df366504 ls -l /usr/src/app/
total 72
-rw-rw-r--  1 root root   257 Nov 12 23:06 Dockerfile
drwxrwxr-x  1 root root  4096 Nov 12 23:00 app
-rw-r--r--  1 root root   608 Nov 12 23:28 clicks_1731454053911.csv
drwxr-xr-x 67 root root  4096 Nov 12 21:49 node_modules
-rw-r--r--  1 root root 47184 Nov 12 21:49 package-lock.json
-rw-rw-r--  1 root root   158 Nov 12 21:42 package.json


$ docker exec -it 9ea4df366504 cat /usr/src/app/clicks_1731454053911.csv
TIMESTAMP,SESSION_ID,EXPRESSION,GAZE_X,GAZE_Y,POSE_DATA
2024-11-12T23:28:06.103Z,1731454053911,neutral,1168.9146981468934,557.7042636810066,"{""nose"":{""x"":368.21683648595945,""y"":285.7076626656705},""leftEye"":{""x"":412.6522584340874,""y"":222.95162462642932},""rightEye"":{""x"":320.59538319032765,""y"":230.53476440683474}}"
2024-11-12T23:28:11.183Z,1731454053911,neutral,859.0277071626986,505.4985871661163,"{""nose"":{""x"":369.35293974258974,""y"":282.30602748180877},""leftEye"":{""x"":413.92163733424337,""y"":217.04155872368764},""rightEye"":{""x"":315.63599965873635,""y"":223.9914048288072}}"
```

