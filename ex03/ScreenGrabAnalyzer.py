import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from scenedetect import detect, ContentDetector
from PIL import Image
import easyocr
from transformers import pipeline
import spacy
from dateutil import parser
import re
import os
import torch
from huggingface_hub import hf_hub_download

class NewsVideoAnalyzer:
    def __init__(self, weights_dir="weights", debug_dir=None):
        # Initialize debug setting
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"Debug mode enabled. Output directory: {self.debug_dir}")
        
        # Create weights directory if it doesn't exist
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        print("Initializing models and loading weights...")
        
        # Initialize models
        self.image_captioner = self._setup_image_captioner()
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize EasyOCR reader
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("Models initialized successfully")
        
    def _setup_image_captioner(self):
        """Setup image captioning model with local weight management"""
        print("Setting up image captioner...")
        model_id = "microsoft/git-base-textcaps"
        
        model_path = self.weights_dir / "git-base-textcaps"
        model_path.mkdir(exist_ok=True)
        
        files_to_download = [
            "config.json",
            "pytorch_model.bin",
            "vocab.txt",
            "generation_config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",  
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt"
        ]
        
        for file in files_to_download:
            if not (model_path / file).exists():
                print(f"Downloading {file}...")
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=file,
                        local_dir=model_path
                    )
                except Exception as e:
                    print(f"Error downloading {file}: {e}")
                    raise
        
        print("Image captioner setup complete")
        return pipeline("image-to-text", model=str(model_path))
    
    def detect_scenes(self, video_path):
        """Detect scene changes in the video"""
        print("Detecting scenes in video...")
        scenes = detect(video_path, ContentDetector(threshold=18))
        scene_list = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]
        print(f"Number of scenes detected: {len(scene_list)}")
        return scene_list
    
    def extract_frame(self, video_path, timestamp):
        """Extract a frame from video at given timestamp"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.debug_dir:
                frame_path = self.debug_dir / f"frame_{timestamp:.2f}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                print(f"Saved frame at timestamp {timestamp:.2f}s to {frame_path}")
            return frame_rgb
            
        print(f"Failed to extract frame at timestamp {timestamp:.2f}s")
        return None

    def perform_ocr(self, frame):
        """Perform OCR on the frame using EasyOCR"""
        print("Performing OCR on frame...")
        # EasyOCR works with numpy arrays directly
        results = self.reader.readtext(frame)
        
        # Extract text from results and combine
        text = ' '.join([result[1] for result in results])
        
        if self.debug_dir:
            ocr_path = self.debug_dir / f"ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            ocr_path.write_text(text)
            print(f"Saved OCR text to {ocr_path}")
            
        return text

    def generate_image_caption(self, frame):
        """Generate caption for the frame"""
        print("Generating image caption...")
        pil_image = Image.fromarray(frame)
        caption = self.image_captioner(pil_image)[0]['generated_text']
        
        if self.debug_dir:
            caption_path = self.debug_dir / f"caption_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            caption_path.write_text(caption)
            print(f"Saved image caption to {caption_path}")
            
        return caption

    def extract_location(self, text):
        """Extract location from text using SpaCy"""
        print("Extracting locations from text...")
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        print(f"Number of locations found: {len(locations)}")
        return locations[0] if locations else None

    def extract_date(self, text):
        """Extract date from text"""
        print("Extracting dates from text...")
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    date = parser.parse(matches[0]).strftime('%Y-%m-%d')
                    print(f"Date found: {date}")
                    return date
                except:
                    continue
        print("No valid dates found in text")
        return None

    def generate_headline(self, text):
        """Generate a short headline from the text"""
        print("Generating headline...")
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if sentences:
            first_sent = str(sentences[0])
            headline = first_sent[:100] if len(first_sent) > 100 else first_sent
            print(f"Generated headline: {headline}")
            return headline
        print("No sentences found for headline generation")
        return None

    def analyze_video(self, video_path, output_path):
        """Main function to analyze the video and generate JSON output"""
        print(f"\nStarting video analysis for: {video_path}")
        scenes = self.detect_scenes(video_path)
        results = []
        
        for i, (scene_start, scene_end) in enumerate(scenes, 1):
            print(f"\nProcessing scene {i}/{len(scenes)}")
            print(f"Scene timestamp: {scene_start:.2f}s - {scene_end:.2f}s")
            
            mid_timestamp = (scene_start + scene_end) / 2
            frame = self.extract_frame(video_path, mid_timestamp)
            
            if frame is None:
                print(f"Skipping scene {i} due to frame extraction failure")
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
            
            if self.debug_dir:
                scene_debug_path = self.debug_dir / f"scene_{i}_data.json"
                with open(scene_debug_path, 'w', encoding='utf-8') as f:
                    json.dump(scene_data, f, indent=2, ensure_ascii=False)
                print(f"Saved scene {i} debug data to {scene_debug_path}")
            
            results.append(scene_data)
        
        print(f"\nAnalysis complete. Processed {len(scenes)} scenes.")
        print(f"Writing results to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze news video and generate JSON output')
    parser.add_argument('-i', required=True, help='Path to input video file')
    parser.add_argument('-o', required=True, help='Path for the output JSON file')
    parser.add_argument('-d', required=False, help='Debug directory if to store intermediate information')
    
    args = parser.parse_args()
    
    analyzer = NewsVideoAnalyzer(weights_dir="./weights", debug_dir=args.d)
    analyzer.analyze_video(args.i, args.o)

if __name__ == "__main__":
    main()