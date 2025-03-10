# Download model configuration and weights
mkdir -p weights

# CNN based model
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg -O weights/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# Transformer based model
wget https://huggingface.co/qualcomm/DETR-ResNet50/tree/main/detr-resnet50.onnx -O weights/detr-resnet50.onnx