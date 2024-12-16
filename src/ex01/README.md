## Video search with image query
The objective of this exercise is to understand inner workings of an image search computational pipeline.
For the scope of this exercise, we are assuming our query is a single image, and our database is either 
a video or a directory containing the frames extracted from the video.

Query Image/Frame |  Video (Shown as GIF) | Result |
------------------|-----------------------|--------|
[!Query](../../data/testframes/000100.jpg "Frame 100") | [!Video](../../assets/ex01-test.gif "Video") | () | 

### Data Preparation

```shell
# Generating GIF from frames
OPTS="fps=25000/10001,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
IP="/path/to/input.mp4"
OP="/path/to/output.gif"
ffmpeg -ss 05 -t 2 -i $IP -vf $OPTS -loop 0 $OP
```

### Code Organization

### Test cases

### Further Optimizations and improvements