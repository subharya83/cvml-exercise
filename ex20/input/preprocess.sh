#!/bin/bash

# Check if input file and prefix are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_mp3_file> <output_prefix>"
    exit 1
fi

INPUT_FILE="$1"
PREF="$2"
SEGMENT_DURATION=240  # 4 minutes in seconds

# Get the total duration of the input file in seconds
TOTAL_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE" | cut -d. -f1)

# Function to zero-pad a number to 5 digits
zero_pad() {
    printf "%05d" "$1"
}

# Function to process a segment
process_segment() {
    START="$1"
    END="$2"
    START_PADDED=$(zero_pad "$START")
    END_PADDED=$(zero_pad "$END")
    OUTPUT_FILE="${PREF}_${START_PADDED}-${END_PADDED}.wav"

    echo "Processing segment: $START - $END"

    # Extract the segment, amplify speech, suppress noise, and resample
    ffmpeg -i "$INPUT_FILE" -ss "$START" -to "$END" \
        -af "highpass=f=200,lowpass=f=3000,afftdn=nf=-25,volume=2.0" \
        -ar 16000 -ac 1 \
        "$OUTPUT_FILE"
}

# Loop through the file in 4-minute segments
START=0
while [ "$START" -lt "$TOTAL_DURATION" ]; do
    END=$((START + SEGMENT_DURATION))
    if [ "$END" -gt "$TOTAL_DURATION" ]; then
        END="$TOTAL_DURATION"
    fi

    process_segment "$START" "$END"

    START=$((START + SEGMENT_DURATION))
done

echo "Segmentation and enhancement complete!"