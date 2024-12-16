# Download test data (video file)
```shell
yt-dlp -S ext:mp4:m4a https://www.youtube.com/watch?v=rTDaZoDDW5g -o rTDaZoDDW5g.mp4
```

# Extract all frames from video
```shell
ffmpeg -hide_banner -thread_queue_size 8192 -i rTDaZoDDW5g.mp4 frames/%06d.jpg
```

# Create small testset for quick testing
```shell
ffmpeg -i rTDaZoDDW5g.mp4 -ss 00:00:20 -to 00:00:30 test.mp4
ffmpeg -i test.mp4 testframes/%06d.jpg 
```
