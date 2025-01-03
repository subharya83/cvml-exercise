import argparse
import pypdf
import spacy
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from typing import List, Dict, Tuple
import os
from huggingface_hub import snapshot_download
import json
from bs4 import BeautifulSoup
import chardet
import srt
import datetime
from difflib import SequenceMatcher
import csv

class DocumentQA:
    def __init__(self):
        self.weights_dir = "./weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        
        if not spacy.util.is_package("en_core_web_sm"):
            os.system("python -m spacy download en_core_web_sm")
        
        self.nlp = spacy.load("en_core_web_sm")
        
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        model_path = os.path.join(self.weights_dir, model_name)
        
        if not os.path.exists(model_path):
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pdf"]
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                return "\n".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def extract_text_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except Exception as e:
            print(f"TXT extraction error: {e}")
            return ""

    def extract_text_from_srt(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
            with open(file_path, 'r', encoding=encoding) as file:
                srt_data = list(srt.parse(file))
                return " ".join(sub.content for sub in srt_data)
        except Exception as e:
            print(f"SRT extraction error: {e}")
            return ""

    def extract_text_from_html(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                
            encodings = [chardet.detect(raw_data)['encoding'], 'utf-8', 'latin-1', 'cp1252', 'ascii']
            
            text = None
            for encoding in encodings:
                try:
                    text = raw_data.decode(encoding, errors='ignore')
                    break
                except (UnicodeDecodeError, TypeError):
                    continue
                    
            if text is None:
                print(f"HTML extraction error: Could not decode file with any encoding")
                return ""
                
            soup = BeautifulSoup(text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ")
            
        except Exception as e:
            print(f"HTML extraction error: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?:;()\-\'\"]+', '', text)
        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        doc = self.nlp(text)
        segments = []
        current_segment = ""
        
        for sent in doc.sents:
            current_segment += sent.text + " "
            if len(current_segment.split()) > 100 or sent.text.endswith('.'):
                segments.append(current_segment.strip())
                current_segment = ""
        
        if current_segment:
            segments.append(current_segment.strip())
        return segments

    def process_document(self, file_path: str, output_format: str = "json") -> List[Dict[str, str]]:
        """Process any supported document type and generate QA pairs."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        extractors = {
            '.pdf': self.extract_text_from_pdf,
            '.txt': self.extract_text_from_txt,
            '.srt': self.extract_text_from_srt,
            '.html': self.extract_text_from_html,
            '.htm': self.extract_text_from_html
        }
        
        extractor = extractors.get(file_extension)
        if not extractor:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        text = extractor(file_path)
        if not text:
            return []
            
        text = self.clean_text(text)
        segments = self.segment_text(text)
        qa_pairs = self.generate_qa_pairs(segments)
        
        # Return the raw QA pairs instead of formatted string
        return [{
            "instruction": pair["question"],
            "input": "",
            "output": pair["answer"]
        } for pair in qa_pairs]

    def detect_duplicates(self, qa_pairs: List[Dict[str, str]], similarity_threshold: float = 0.85) -> List[Dict[str, str]]:
        """
        Detect and remove near-duplicate QA pairs based on similarity threshold.
        
        Args:
            qa_pairs: List of dictionaries containing QA pairs
            similarity_threshold: Threshold for considering entries as duplicates (0.0 to 1.0)
            
        Returns:
            List of unique QA pairs
        """
        def calculate_similarity(str1: str, str2: str) -> float:
            """Calculate string similarity using SequenceMatcher."""
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        
        def is_duplicate(current: Dict[str, str], others: List[Dict[str, str]]) -> bool:
            """Check if current QA pair is similar to any in others."""
            current_q = current["instruction"]
            current_a = current["output"]
            
            for other in others:
                other_q = other["instruction"]
                other_a = other["output"]
                
                # Check both question and answer similarity
                q_similarity = calculate_similarity(current_q, other_q)
                a_similarity = calculate_similarity(current_a, other_a)
                
                # Consider it duplicate if either question or answer is very similar
                if q_similarity > similarity_threshold or a_similarity > similarity_threshold:
                    return True
            return False
        
        unique_pairs = []
        duplicates_found = 0
        
        # Process each QA pair
        for qa_pair in qa_pairs:
            if not is_duplicate(qa_pair, unique_pairs):
                unique_pairs.append(qa_pair)
            else:
                duplicates_found += 1
        
        print(f"Found and removed {duplicates_found} duplicate entries")
        return unique_pairs
    
    def generate_qa_pairs(self, segments: List[str]) -> List[Dict[str, str]]:
        qa_pairs = []
        for segment in segments:
            doc = self.nlp(segment)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            for ent in entities:
                if ent[1] in ['PERSON', 'ORG', 'GPE']:
                    qa_pairs.append({
                        "question": f"Who or what is {ent[0]}?",
                        "answer": self.extract_context(segment, ent[0])
                    })
                elif ent[1] in ['DATE', 'TIME']:
                    qa_pairs.append({
                        "question": f"When did the events related to {ent[0]} occur?",
                        "answer": self.extract_context(segment, ent[0])
                    })
            
            summary = self.generate_summary_question(segment)
            if summary:
                qa_pairs.append(summary)
        
        return qa_pairs

    def extract_context(self, text: str, entity: str, window: int = 100) -> str:
        pos = text.find(entity)
        if pos == -1:
            return text
        
        start = max(0, pos - window)
        end = min(len(text), pos + len(entity) + window)
        
        while start > 0 and text[start] != ' ':
            start -= 1
        while end < len(text) and text[end] != ' ':
            end += 1
        
        return text[start:end].strip()

    def generate_summary_question(self, text: str) -> Dict[str, str]:
        doc = self.nlp(text)
        main_topics = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        
        if main_topics:
            topic = main_topics[0]
            return {
                "question": f"What is the main point about {topic}?",
                "answer": text
            }
        return None

def write_output(data: List[Dict[str, str]], output_path: str, similarity_threshold: float = None):
    """Write the QA pairs to either JSON or CSV file based on the file extension."""
    if similarity_threshold is not None:
        qa_processor = DocumentQA()
        data = qa_processor.detect_duplicates(data, similarity_threshold)
    
    file_extension = os.path.splitext(output_path)[1].lower()
    
    if file_extension == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    elif file_extension == '.csv':
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["instruction", "input", "output"])
            for item in data:
                writer.writerow([item["instruction"], item["input"], item["output"]])
                
    else:
        raise ValueError(f"Unsupported output format: {file_extension}")
    
    print(f"Successfully wrote data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse document to generate QA')
    parser.add_argument('-i', required=True, help='Path to input document file')
    parser.add_argument('-o', required=True, help='Path for the output csv/json file')
    parser.add_argument('-d', type=float, nargs='?', const=0.85, metavar='SIMILARITY',
                       help='Enable duplicate detection with optional similarity threshold (0.0 to 1.0, default: 0.85)')

    args = parser.parse_args()
    
    if not os.path.exists(args.i):
        print(f"Error: Input file {args.i} does not exist")
        exit(1)
        
    try:
        converter = DocumentQA()
        print(f"\nProcessing {args.i}...")
        qa_data = converter.process_document(args.i)
        
        if not qa_data:
            print("No QA pairs were generated from the document")
            exit(1)
            
        write_output(qa_data, args.o, similarity_threshold=args.d)
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        exit(1)