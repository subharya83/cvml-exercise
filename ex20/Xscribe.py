import os
import whisper
import srt
from datetime import timedelta

# Ensure the weights directory exists
os.makedirs("weights", exist_ok=True)

def transcribe_audio_to_srt(input_audio_path, output_srt_path, language_code="bn"):
    # Load the Whisper model (it will download weights if not already present)
    model = whisper.load_model("large", download_root="weights")

    # Transcribe the audio file
    result = model.transcribe(input_audio_path, language=language_code)

    # Generate SRT subtitles
    subtitles = []
    for i, segment in enumerate(result["segments"]):
        start_time = timedelta(seconds=segment["start"])
        end_time = timedelta(seconds=segment["end"])
        text = segment["text"]
        subtitle = srt.Subtitle(index=i + 1, start=start_time, end=end_time, content=text)
        subtitles.append(subtitle)

    # Write the SRT file
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

if __name__ == "__main__":
    # Input and output file paths
    input_audio = input("Enter the path to the input audio file: ")
    output_srt = input("Enter the path for the output SRT file: ")

    # Language code (e.g., "bn" for Bengali)
    language_code = input("Enter the language code (e.g., 'bn' for Bengali): ")

    # Transcribe and generate SRT
    transcribe_audio_to_srt(input_audio, output_srt, language_code)
    print(f"SRT file generated successfully at {output_srt}")
