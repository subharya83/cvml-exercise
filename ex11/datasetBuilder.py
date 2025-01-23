import os
import csv
import hashlib
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

def download_srt(youtube_link):
    """Download SRT file for a given YouTube link."""
    try:
        video_id = YouTube(youtube_link).video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        srt_filename = f"{video_id}.srt"
        with open(srt_filename, "w", encoding="utf-8") as f:
            for line in transcript:
                f.write(f"{line['text']}\n")
        return srt_filename, video_id
    except Exception as e:
        print(f"Error downloading SRT for {youtube_link}: {e}")
        return None, None

def generate_qa_from_srt(srt_filename):
    """Generate Q/A pairs from an SRT file."""
    qa_pairs = []
    with open(srt_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(len(lines) - 1):
            question = lines[i].strip()
            answer = lines[i + 1].strip()
            qa_pairs.append((question, answer))
    return qa_pairs

def hash_qa(question, answer):
    """Generate a unique hash for a Q/A pair."""
    qa_string = f"{question}{answer}".encode("utf-8")
    return hashlib.md5(qa_string).hexdigest()

def save_dataset(qa_pairs, video_id, output_dir="dataset"):
    """Save Q/A dataset and mapping CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_filename = os.path.join(output_dir, f"{video_id}_qa.txt")
    mapping_filename = os.path.join(output_dir, "mapping.csv")

    with open(dataset_filename, "w", encoding="utf-8") as f:
        for q, a in qa_pairs:
            f.write(f"Q: {q}\nA: {a}\n\n")

    with open(mapping_filename, "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        for q, a in qa_pairs:
            qa_hash = hash_qa(q, a)
            writer.writerow([video_id, qa_hash, q, a])

def process_youtube_links(youtube_links):
    """Process a list of YouTube links to generate Q/A dataset."""
    for link in youtube_links:
        srt_filename, video_id = download_srt(link)
        if srt_filename and video_id:
            qa_pairs = generate_qa_from_srt(srt_filename)
            save_dataset(qa_pairs, video_id)
            print(f"Processed {link} and saved dataset for video ID: {video_id}")

if __name__ == "__main__":
    youtube_links = [
        "https://www.youtube.com/watch?v=example1",
        "https://www.youtube.com/watch?v=example2",
        # Add more YouTube links here
    ]
    process_youtube_links(youtube_links)
