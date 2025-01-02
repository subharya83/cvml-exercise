import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from scenedetect import detect, ContentDetector
from PIL import Image
import pytesseract
from transformers import pipeline
import spacy
from dateutil import parser
import re
import os
import torch
from huggingface_hub import hf_hub_download

class NewsVideoAnalyzer:
    def __init__(self, weights_dir="weights"):
        # Create weights directory if it doesn't exist
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.image_captioner = self._setup_image_captioner()
        self.nlp = spacy.load("en_core_web_sm")
        
    def _setup_image_captioner(self):
        """Setup image captioning model with local weight management"""
        # Using LAVIS GIT-base model as an open-source alternative
        # It has similar performance to BLIP but with an open license
        model_id = "microsoft/git-base-textcaps"
        
        # Create model-specific directory
        model_path = self.weights_dir / "git-base-textcaps"
        model_path.mkdir(exist_ok=True)
        
        # Download model files if they don't exist
        files_to_download = [
            "config.json",
            "pytorch_model.bin",
            "vocab.json",
            "merges.txt"
        ]
        
        for file in files_to_download:
            if not (model_path / file).exists():
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=file,
                        local_dir=model_path
                    )
                except Exception as e:
                    print(f"Error downloading {file}: {e}")
                    raise
        
        # Initialize the pipeline with local weights
        return pipeline(
            "image-to-text",
            model=str(model_path),
            local_files_only=True
        )
    
    def detect_scenes(self, video_path):
        """Detect scene changes in the video"""
        scenes = detect(video_path, ContentDetector())
        return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]
    
    def extract_frame(self, video_path, timestamp):
        """Extract a frame from video at given timestamp"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def perform_ocr(self, frame):
        """Perform OCR on the frame"""
        pil_image = Image.fromarray(frame)
        text = pytesseract.image_to_string(pil_image)
        return text

    def generate_image_caption(self, frame):
        """Generate caption for the frame"""
        pil_image = Image.fromarray(frame)
        caption = self.image_captioner(pil_image)[0]['generated_text']
        return caption

    def extract_location(self, text):
        """Extract location from text using SpaCy"""
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        return locations[0] if locations else None

    def extract_date(self, text):
        """Extract date from text"""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    return parser.parse(matches[0]).strftime('%Y-%m-%d')
                except:
                    continue
        return None

    def generate_headline(self, text):
        """Generate a short headline from the text"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if sentences:
            first_sent = str(sentences[0])
            return first_sent[:100] if len(first_sent) > 100 else first_sent
        return None

    def analyze_video(self, video_path, output_path):
        """Main function to analyze the video and generate JSON output"""
        scenes = self.detect_scenes(video_path)
        results = []
        
        for scene_start, scene_end in scenes:
            mid_timestamp = (scene_start + scene_end) / 2
            frame = self.extract_frame(video_path, mid_timestamp)
            
            if frame is None:
                continue
                
            ocr_text = self.perform_ocr(frame)
            image_caption = self.generate_image_caption(frame)
            
            headline = self.generate_headline(ocr_text)
            location = self.extract_location(ocr_text)
            date = self.extract_date(ocr_text)
            
            scene_data = {
                "timestamp": {
                    "start": scene_start,
                    "end": scene_end
                },
                "headline": headline if headline else image_caption,
                "summary": ocr_text.strip(),
                "date": date,
                "location": location,
                "image_caption": image_caption
            }
            
            results.append(scene_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze news video and generate JSON output')
    parser.add_argument('-i', help='Path to input video file')
    parser.add_argument('-o', help='Path for the output JSON file')
    parser.add_argument('--weights-dir', default='weights', help='Directory to store model weights')
    args = parser.parse_args()
    
    analyzer = NewsVideoAnalyzer(weights_dir=args.weights_dir)
    analyzer.analyze_video(args.i, args.o)

if __name__ == "__main__":
    main()