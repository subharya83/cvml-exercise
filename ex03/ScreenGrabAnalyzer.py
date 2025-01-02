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

class NewsVideoAnalyzer:
    def __init__(self):
        # Initialize models
        self.image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        self.nlp = spacy.load("en_core_web_sm")
        
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
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        # Perform OCR
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
        # Look for common date patterns
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
        # Split into sentences
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if sentences:
            # Return first sentence if it's not too long, otherwise return first 100 chars
            first_sent = str(sentences[0])
            return first_sent[:100] if len(first_sent) > 100 else first_sent
        return None

    def analyze_video(self, video_path, output_path):
        """Main function to analyze the video and generate JSON output"""
        scenes = self.detect_scenes(video_path)
        results = []
        
        for scene_start, scene_end in scenes:
            # Extract middle frame from scene
            mid_timestamp = (scene_start + scene_end) / 2
            frame = self.extract_frame(video_path, mid_timestamp)
            
            if frame is None:
                continue
                
            # Perform analysis
            ocr_text = self.perform_ocr(frame)
            image_caption = self.generate_image_caption(frame)
            
            # Extract information
            headline = self.generate_headline(ocr_text)
            location = self.extract_location(ocr_text)
            date = self.extract_date(ocr_text)
            
            # Create scene data
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
        
        # Save results to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze news video and generate JSON output')
    parser.add_argument('-i', help='Path to input video file')
    parser.add_argument('-o', help='Path for the output JSON file')
    args = parser.parse_args()
    
    # Create analyzer and process video
    analyzer = NewsVideoAnalyzer()
    analyzer.analyze_video(args.i, args.o)

if __name__ == "__main__":
    main()