import os
import whisper
import srt
import argparse
from datetime import timedelta
from pyannote.audio import Pipeline

# Ensure the weights directory exists
os.makedirs("weights", exist_ok=True)

def transcribe_audio_with_diarization(input_audio_path, output_srt_path, language_code="bn"):
    # Load the Whisper model (it will download weights if not already present)
    print("Loading Whisper model...")
    model = whisper.load_model("large", download_root="weights")
    model = model.to("cpu")  # Move the model to CPU

    # Load the pyannote.audio speaker diarization pipeline
    print("Loading speaker diarization model...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # Perform speaker diarization
    print("Running speaker diarization...")
    diarization = diarization_pipeline(input_audio_path)

    # Transcribe the audio file
    print("Transcribing audio...")
    transcription = model.transcribe(input_audio_path, language=language_code)

    # Align diarization results with transcription segments
    subtitles = []
    speaker_labels = {}
    speaker_counter = 1

    for segment in transcription["segments"]:
        start_time = timedelta(seconds=segment["start"])
        end_time = timedelta(seconds=segment["end"])
        text = segment["text"]

        # Find the speaker for this segment
        speaker = None
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            if turn.start <= segment["start"] <= turn.end:
                speaker = speaker_id
                break

        # Assign a speaker label (e.g., Speaker 01, Speaker 02)
        if speaker not in speaker_labels:
            speaker_labels[speaker] = f"Speaker {speaker_counter:02d}"
            speaker_counter += 1

        # Add the speaker label to the subtitle text
        subtitle_text = f"{speaker_labels[speaker]}: {text}"

        # Create the subtitle
        subtitle = srt.Subtitle(
            index=len(subtitles) + 1,
            start=start_time,
            end=end_time,
            content=subtitle_text
        )
        subtitles.append(subtitle)

    # Write the SRT file
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    print(f"SRT file generated successfully at {output_srt_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization and generate an SRT file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input audio file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output SRT file.")
    parser.add_argument("-l", "--language", default="bn", help="Language code for transcription (e.g., 'bn' for Bengali).")
    args = parser.parse_args()

    # Transcribe and generate SRT with speaker diarization
    transcribe_audio_with_diarization(args.input, args.output, args.language)