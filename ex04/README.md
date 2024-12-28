# Convert text content into Q&A key-value pairs for LLM fine-tuning

## Convert PDF 
1. **Setup Requirements**:
```bash
pip install pypdf spacy transformers
python -m spacy download en_core_web_sm
```

2. **Key Features**:
- Extracts text from PDF while preserving structure
- Segments text into meaningful chunks
- Generates natural questions using NER and content analysis
- Creates context-aware answers
- Outputs in common fine-tuning formats (JSON/CSV)

3. **Example Usage**:
```python
converter = PDFtoQA()
qa_data = converter.process_pdf("your_pdf.pdf", "json")
```

4. **Output Format**:
```json
[
  {
    "instruction": "Who is John Smith?",
    "input": "",
    "output": "John Smith is the lead researcher who developed..."
  },
  {
    "instruction": "What is the main point about neural networks?",
    "input": "",
    "output": "Neural networks are computational models..."
  }
]
```

5. **Key Improvements You Can Make**:
- Add more question types based on content patterns
- Implement better context window selection
- Add support for tables and structured data
- Implement question quality filtering
- Add support for multiple PDFs processing

## Convert YouTube Transcripts

1. Get list of YouTube Ids from a specific channel
```shell
youtube-dl --flat-playlist https://www.youtube.com/@samharrisorg/videos --get-id > example/example.ytid
```

2. Download text and subtitle files (corresponding to the IDs)
```shell
for i in $(cat example/example.ytid);do 
    youtube_transcript_api $i --format text > example/txt/YTID_$i.txt; 
    youtube_transcript_api $i --format srt > example/srt/YTID_$i.srt; 
done
```

3. Preprocessing the collected transcripts 

```shell
python3 prepro_ytxscrips.py -i example/txt -o example -p
Generating dataset from txt files in example
Saved 32768 tokens to example/val.bin
Saved 1827300 tokens to example/train.bin
```
