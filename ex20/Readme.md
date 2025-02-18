# Audio to SRT Transcription using Whisper

This Python script transcribes an input audio file into an SRT subtitle file using OpenAI's Whisper model. It supports multiple languages and generates subtitles in English. The model weights are downloaded locally to a `weights` directory.

## Features
- Transcribes audio files to SRT subtitles.
- Supports multiple languages (e.g., Bengali (`bn`), English, etc.).
- Uses the state-of-the-art Whisper model for accurate transcription.
- Downloads and stores model weights locally for offline use.

## Prerequisites
- Python 3.7 or higher.
- `ffmpeg` installed on your system.

