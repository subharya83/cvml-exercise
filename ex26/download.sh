# Download model configuration and weights
mkdir -p weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg -O weights/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
