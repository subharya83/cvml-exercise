# DocumentQA: Automated Question-Answer Pair Generator

A Python tool that automatically generates question-answer pairs from various document formats using NLP and transformer models.

## Problem Design

### Objectives
- Extract meaningful text from multiple document formats (PDF, TXT, SRT, HTML)
- Generate relevant question-answer pairs from the extracted text
- Support different output formats (JSON, CSV) for fine-tuning language models
- Maintain context-awareness when generating QA pairs

### Constraints
- Documents must be in supported formats (PDF, TXT, SRT, HTML)
- Text extraction quality depends on document formatting
- Question generation is limited to entity-based and summary-based questions
- Memory limitations based on document size
- Requires internet connection for initial model downloads

### Success Criteria
- Successful text extraction from supported document formats
- Generation of contextually relevant QA pairs
- Proper handling of different languages and encodings
- Accurate entity recognition and context preservation
- Clean output in specified format (JSON/CSV)

## Data Preparation

### Input Processing
1. **Document Loading**
   - Multiple encoding support (UTF-8, ASCII, Latin-1, CP1252)
   - Format-specific extractors for PDF, TXT, SRT, and HTML
   - Error handling for corrupted or unsupported files

2. **Text Cleaning**
   - Removal of excessive whitespace and newlines
   - Standardization of punctuation and special characters
   - Preservation of sentence structure and context

3. **Text Segmentation**
   - Breaking text into manageable chunks (max 100 words)
   - Maintaining sentence integrity
   - Preserving contextual relationships

### Model Preparation
- Automatic download of required models
- SpaCy model for NER and text processing
- BERT model for question answering
- Weights management and storage

## Code Organization

### Code structure
```
├── autopunc.py
├── convertQA.py
├── input
│   ├── 3-Kings.pdf
│   └── Argo.htm
├── output
│   ├── Argo.csv
│   └── Argo.json
├── prepro_ytxscrips.py
├── README.md
└── weights
    └── bert-large-uncased-whole-word-masking-finetuned-squad
        ├── config.json
        ├── model.safetensors
        ├── pytorch_model.bin
        ├── README.md
        ├── saved_model.tar.gz
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.txt
```

### Class Structure
```
DocumentQA/
├── __init__()              # Model initialization and setup
├── Extractors/
│   ├── extract_text_from_pdf()
│   ├── extract_text_from_txt()
│   ├── extract_text_from_srt()
│   └── extract_text_from_html()
├── Text Processing/
│   ├── clean_text()
│   └── segment_text()
├── QA Generation/
│   ├── process_document()
│   ├── generate_qa_pairs()
│   ├── extract_context()
│   └── generate_summary_question()
└── Output Handling/
    └── write_output()
```

### Key Components
1. **Document Processing**
   - Format-specific text extractors
   - Encoding detection and handling
   - Error management

2. **NLP Pipeline**
   - Entity recognition
   - Context extraction
   - Question generation

3. **Output Generation**
   - Format conversion
   - File writing
   - Error handling

## Test Cases

### Document Loading Tests
```python
def test_pdf_extraction():
    # Test PDF with text content
    # Test PDF with images
    # Test corrupted PDF
    # Test empty PDF

def test_encoding_handling():
    # Test UTF-8 documents
    # Test documents with special characters
    # Test documents with multiple encodings
```

### QA Generation Tests
```python
def test_qa_generation():
    # Test with short text
    # Test with long text
    # Test with multiple entities
    # Test with no entities
    # Test with edge cases (empty text, special characters)
```

### Output Format Tests
```python
def test_output_formats():
    # Test JSON output
    # Test CSV output
    # Test with Unicode characters
    # Test with large datasets
```

## Further Optimizations and Improvements

### Performance Enhancements
1. **Processing Speed**
   - Implement batch processing for large documents
   - Add multiprocessing for parallel text extraction
   - Optimize memory usage for large files

2. **QA Quality**
   - Implement more sophisticated question generation strategies
   - Add support for different question types
   - Improve context selection algorithm

### Feature Additions
1. **Document Support**
   - Add support for more document formats (DOCX, EPUB)
   - Implement image text extraction (OCR)
   - Add support for tables and structured data

2. **Output Options**
   - Add support for more output formats
   - Implement streaming output for large files
   - Add customizable output templates

### Model Improvements
1. **Language Support**
   - Add multilingual support
   - Implement language-specific question generation
   - Add custom entity recognition

2. **Quality Control**
   - Add confidence scores for generated QA pairs
   - Implement quality filters