


```shell
# Compile video corruptor, SpatiotemPoral Anomalous pixel DEtector
g++ -std=c++11 -o videoCorruptor videoCorruptor.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++11 -o Spade Spade.cpp `pkg-config --cflags --libs opencv4`
```

```shell
./Spade -i output/c-hummer.mp4 -o output/c-hummer.csv -v output/o-hummer.mp4
````

### Multi-threaded version improvements

Key Changes:
- Thread Pool: A thread pool is created using std::thread and std::mutex to manage concurrent processing of frames.
- Frame Queue: A queue (std::queue) is used to store frames that need to be processed. Threads will pop frames from this queue and process them.
- Mutexes: std::mutex is used to synchronize access to shared resources like the frame queue and the defects data structures.
- Worker Function: A lambda function (worker) is defined to process frames. Each thread runs this function, which processes frames until the queue is empty.

Notes:
- The number of threads is determined by `std::thread::hardware_concurrency()`, which returns the number of concurrent threads supported by the hardware.
- The `update_defects()`  is protected by a mutex to ensure thread safety when updating the `active_defects_` and `completed_defects_` structures.
- The `visualize_defects()` is also protected by a mutex to ensure thread-safe access to the active_defects_ structure.
