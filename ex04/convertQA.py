import pypdf
import spacy
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from typing import List, Dict, Tuple

class PDFtoQA:
    def __init__(self):
        # Load SpaCy model for NER and sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize BERT tokenizer and model for question generation
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        
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
        # Remove extra newlines and whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?:;()\-\'\"]+', '', text)
        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        """Split text into meaningful segments/paragraphs."""
        doc = self.nlp(text)
        # Group sentences into paragraphs based on content similarity
        segments = []
        current_segment = ""
        
        for sent in doc.sents:
            current_segment += sent.text + " "
            # Start new segment if current one is long enough or contains complete thought
            if len(current_segment.split()) > 100 or sent.text.endswith('.'):
                segments.append(current_segment.strip())
                current_segment = ""
        
        if current_segment:  # Add any remaining text
            segments.append(current_segment.strip())
            
        return segments

    def generate_qa_pairs(self, segments: List[str]) -> List[Dict[str, str]]:
        """Generate question-answer pairs from text segments."""
        qa_pairs = []
        
        for segment in segments:
            # Process each segment to identify key information
            doc = self.nlp(segment)
            
            # Extract entities and generate questions
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            for ent in entities:
                # Generate different types of questions based on entity type
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
                
            # Generate general questions about the content
            summary = self.generate_summary_question(segment)
            if summary:
                qa_pairs.append(summary)
                
        return qa_pairs

    def extract_context(self, text: str, entity: str, window: int = 100) -> str:
        """Extract relevant context around an entity."""
        # Find the position of the entity in the text
        pos = text.find(entity)
        if pos == -1:
            return text
            
        # Extract window of text around the entity
        start = max(0, pos - window)
        end = min(len(text), pos + len(entity) + window)
        
        # Ensure we don't cut words in half
        while start > 0 and text[start] != ' ':
            start -= 1
        while end < len(text) and text[end] != ' ':
            end += 1
            
        return text[start:end].strip()

    def generate_summary_question(self, text: str) -> Dict[str, str]:
        """Generate a summary-type question for the text."""
        # Extract main topics using NLP
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
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return ""
            
        # Segment the text
        segments = self.segment_text(text)
        
        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(segments)
        
        # Format for fine-tuning
        return self.format_for_fine_tuning(qa_pairs, output_format)

def main():
    # Example usage
    converter = PDFtoQA()
    qa_data = converter.process_pdf("example.pdf", "json")
    print(qa_data)

if __name__ == "__main__":
    main()