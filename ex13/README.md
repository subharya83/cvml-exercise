


```shell
# Compile video corruptor, SpatiotemPoral Anomalous pixel DEtector
g++ -std=c++11 -o videoCorruptor videoCorruptor.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++11 -o Spade Spade.cpp `pkg-config --cflags --libs opencv4`
```

```shell
./Spade -i output/c-hummer.mp4 -o output/c-hummer.csv -v output/o-hummer.mp4
````