# Audio Pattern Matching Tool

This tool finds occurrences of a short audio pattern (query) within a longer audio file using cross-correlation. It outputs timestamps where matches are found along with their correlation scores.

## Dependencies

- C++ compiler with C++11 support
- [AudioFile](https://github.com/adamstark/AudioFile) (header-only library for audio file handling)

## Building

```bash
g++ -std=c++11 main.cpp -o audio_matcher
```

## Usage

```bash
./audio_matcher -i <input_audio> -q <query_audio> -o <output_csv>
```

Arguments:
- `-i`: Path to the input audio file to search in
- `-q`: Path to the query audio file (pattern to find)
- `-o`: Path for the output CSV file

## Output Format

The program generates a CSV file with two columns:
1. Timestamp (seconds): The time point where a match was found
2. Correlation: The correlation score (0-1) indicating match confidence

## Example

```bash
./audio_matcher -i long_recording.wav -q gunshot.wav -o detections.csv
```

## Technical Details

- The program uses normalized cross-correlation to find matches
- Default correlation threshold is 0.7 (can be modified in the code)
- Audio files are converted to mono if necessary
- Supports common audio formats (WAV, AIFF)

## Notes

- For best results, ensure the query audio is clean and contains only the pattern you're looking for
- Higher correlation scores indicate better matches
- Multiple detections close to each other might indicate the same event