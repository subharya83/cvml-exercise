## Video search with image query
The objective of this exercise is to understand inner workings of an image search computational pipeline.
For the scope of this exercise, we are assuming our query is a single image, and our database is either 
a video or a directory containing the frames extracted from the video.

Query Image/Frame |  Video (Shown as GIF) | Result |
------------------|-----------------------|--------|
![Query](../../data/ex01/testframes/000100.jpg "Frame 100") | ![Video](../../assets/ex01-test.gif "Video") | Frame Number: 99 | 

### Data Preparation

1. Downloading query and dataset

```shell
# Download test data (video file)
yt-dlp -S ext:mp4:m4a https://www.youtube.com/watch?v=rTDaZoDDW5g -o rTDaZoDDW5g.mp4
```

```shell
# Extract all frames from video
ffmpeg -hide_banner -thread_queue_size 8192 -i rTDaZoDDW5g.mp4 frames/%06d.jpg
```

```shell
# Create small testset for quick testing
ffmpeg -i rTDaZoDDW5g.mp4 -ss 00:00:20 -to 00:00:30 test.mp4
ffmpeg -i test.mp4 testframes/%06d.jpg 
```
2. Data Organization
```shell
tree -L 1 

├── frames
├── rTDaZoDDW5g.mp4
├── testframes
├── testIndex-f.json
├── testIndex-v.json
└── test.mp4

2 directories, 4 files
```

3. Useful transcode for visualization 
```shell
# Generating GIF from frames
OPTS="fps=25000/10001,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
IP="/path/to/input.mp4"
OP="/path/to/output.gif"
ffmpeg -ss 05 -t 2 -i $IP -vf $OPTS -loop 0 $OP
```

### Code Organization
```shell
tree

├── buildIndex.py
├── embeddings.py
├── getSimilarity.py
├── __pycache__
│   └── embeddings.cpython-310.pyc
├── queryDataset.py
└── README.md

1 directory, 6 files

```
### Test cases

1. Image Similarity

```shell
python3 getSimilarity.py -q ../../data/ex01/testframes/000010.jpg -t ../../data/ex01/testframes/000011.jpg
Image Similarity: 0.9946704506874084
```
2. Building index for future query
```shell
# Using sequence of images as input
python3 buildIndex.py -i ../../data/ex01/testframes/ -o ../../data/ex01/testIndex-f.json
Image sequence processed
Processed 300 frames. Index saved to ../../data/ex01/testIndex-f.json

# Using video as input
python3 buildIndex.py -i ../../data/ex01/test.mp4 -o ../../data/ex01/testIndex-v.json
Video processed
Processed 300 frames. Index saved to ../../data/ex01/testIndex-v.json
```
3. Search and retrieval

```shell
# Query against frame based imdex
python3 queryDataset.py -q ../../data/ex01/testframes/000100.jpg -i ../../data/ex01/testIndex-v.json
Most Similar Frame:
Frame Number: 99
Cosine Similarity: 0.8103252220706153

# Query against video based index 
python3 queryDataset.py -q ../../data/ex01/testframes/000100.jpg -i ../../data/ex01/testIndex-f.json
Most Similar Frame:
Frame Number: 99
Cosine Similarity: 1.0000000046178263
```

### Further Optimizations and improvements