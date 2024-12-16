# Download test data (video file)
```shell
yt-dlp -S ext:mp4:m4a https://www.youtube.com/watch?v=rTDaZoDDW5g -o rTDaZoDDW5g.mp4
```

# Extract all frames from video
```shell
ffmpeg -hide_banner -thread_queue_size 8192 -i rTDaZoDDW5g.mp4 frames/%06d.jpg
```
