#!/bin/bash
# Transcode input video to canonical form
IP=data/home_gym.webm
OP=data/home_gym.mp4
ffmpeg -hide_banner -y -loglevel warning -thread_queue_size 8192 -i $IP -r 25000/1001 -vcodec libx264 -preset slower -profile:v high -crf 24 -pix_fmt yuv420p -shortest $OP

mkdir -p output
mkdir -p tmp
# Running pyscenedetect to sample frames for evaluation
scenedetect -i data/home_gym.mp4 list-scenes output/scenes.csv save-images -o tmp

# Running OCR on detected keyframes
python3 processdata.py -m 0 -i tmp/ > output/ocr.log

# Pruning relevant information for NER tasks
cat output/ocr.log |grep -v DEBUG|cut -d' ' -f10- > output/analysis.txt

# Running nlp for named entity recognition
python3 processdata.py -m 1 -i output/analysis.txt > output/tokens.json