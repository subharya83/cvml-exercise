import pypdf
import spacy
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from typing import List, Dict, Tuple
import os
from huggingface_hub import snapshot_download

class PDFtoQA:
    def __init__(self):
        # Create weights directory if it doesn't exist
        self.weights_dir = "./weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Download SpaCy model
        if not spacy.util.is_package("en_core_web_sm"):
            os.system("python3 -m spacy download en_core_web_sm")
        
        # Load SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Download and load BERT model and tokenizer
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

    # Rest of the class implementation remains the same
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        pdf_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() + "\n"
            return self.clean_text(pdf_text)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and special characters."""
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?:;()\-\'\"]+', '', text)
        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        """Split text into meaningful segments/paragraphs."""
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

    def generate_qa_pairs(self, segments: List[str]) -> List[Dict[str, str]]:
        """Generate question-answer pairs from text segments."""
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
        """Extract relevant context around an entity."""
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
        """Generate a summary-type question for the text."""
        doc = self.nlp(text)
        main_topics = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        
        if main_topics:
            topic = main_topics[0]
            return {
                "question": f"What is the main point about {topic}?",
                "answer": text
            }
        return None

    def format_for_fine_tuning(self, qa_pairs: List[Dict[str, str]], output_format: str = "json") -> str:
        """Format QA pairs for fine-tuning in specified format."""
        if output_format.lower() == "json":
            import json
            formatted_data = [{
                "instruction": pair["question"],
                "input": "",
                "output": pair["answer"]
            } for pair in qa_pairs]
            return json.dumps(formatted_data, indent=2)
        
        elif output_format.lower() == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["instruction", "input", "output"])
            for pair in qa_pairs:
                writer.writerow([pair["question"], "", pair["answer"]])
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def process_pdf(self, pdf_path: str, output_format: str = "json") -> str:
        """Process PDF and generate formatted QA pairs."""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return ""
            
        segments = self.segment_text(text)
        qa_pairs = self.generate_qa_pairs(segments)
        return self.format_for_fine_tuning(qa_pairs, output_format)

def main():
    converter = PDFtoQA()
    qa_data = converter.process_pdf("example.pdf", "json")
    print(qa_data)

if __name__ == "__main__":
    main()