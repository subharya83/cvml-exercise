# Screen Capture Video Analysis

A Python-based tool for analyzing screen recordings of news browsing sessions, extracting key information, and generating structured data output.
## Rough Problem sketch 
I have a video of screen capture that records a user browsing news articles from different websites that may contain images, videos and text. The screen recording at minimum last 5 minutes. I want a set of python/shell scripts that will do the following:
* Perform scene detection using pySceneDetect 
* Perform OCR
* Perform Image/video captioning if required
* Create a json output file that  have the following - a short headline indicating a story, a summary description, timestamp/or date of story, location where the story happened

## Problem Design

### Objectives
- Extract meaningful information from screen recordings of news browsing sessions
- Generate structured data about news stories including headlines, summaries, and metadata
- Process various media types including text, images, and video content

### Choice of model
- GIT (Generative Image-to-text Transformer) is open-source and has comparable performance to BLIP
- The textcaps variant is specifically trained for caption generation
- Has a permissive license for commercial use

### Constraints
- Minimum video duration: 5 minutes
- Input must be screen capture footage of news browsing
- Videos may contain mixed media (text, images, videos)
- Output must be in JSON format

### Success Criteria
1. Accurate scene detection of different news articles
2. Reliable text extraction from news content
3. Meaningful image/video caption generation
4. Accurate extraction of metadata (dates, locations)
5. Well-structured JSON output containing all required information

## Data Preparation

### Input Requirements
- Video Format: MP4, AVI, or other common video formats
- Resolution: Minimum 720p recommended for optimal OCR performance
- Frame Rate: Standard 30fps or higher
- Video Quality: Clear, stable footage without excessive motion blur

### Preprocessing Steps
1. Scene Detection
   - Content-aware scene detection using PySceneDetect
   - Frame extraction at optimal intervals
   - Frame quality assessment

2. Frame Processing
   - Color space conversion (BGR to RGB)
   - Image enhancement for OCR
   - Resolution standardization
   - Noise reduction if needed

3. Text Extraction
   - OCR optimization for different text layouts
   - Text cleaning and normalization
   - Removal of irrelevant text elements

## Code Organization

```
news_video_analyzer/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── analyzer.py        # Main NewsVideoAnalyzer class
│   ├── scene_detector.py  # Scene detection utilities
│   ├── ocr_processor.py   # OCR related functions
│   ├── caption_gen.py     # Image captioning utilities
│   └── utils/
│       ├── __init__.py
│       ├── text_processor.py
│       └── video_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py
│   ├── test_scene_detector.py
│   └── test_ocr.py
└── examples/
    ├── sample_videos/
    └── sample_output/
```

### Key Components
1. NewsVideoAnalyzer: Main class orchestrating the analysis pipeline
2. Scene Detection: Handles video segmentation
3. OCR Processing: Manages text extraction
4. Caption Generation: Handles image description
5. Utilities: Common helper functions and tools

## Test Cases

### Unit Tests
1. Scene Detection
   - Test scene boundary detection accuracy
   - Verify frame extraction quality
   - Test handling of various video formats

2. OCR Processing
   - Test text extraction accuracy
   - Verify handling of different fonts and layouts
   - Test multilingual text support

3. Caption Generation
   - Test image caption quality
   - Verify handling of different image types
   - Test caption relevance

4. Metadata Extraction
   - Test date parsing accuracy
   - Verify location extraction
   - Test handling of different date formats

### Integration Tests
1. End-to-End Pipeline
   - Test complete workflow with sample videos
   - Verify JSON output structure
   - Test error handling and recovery

2. Performance Tests
   - Test processing speed
   - Memory usage monitoring
   - Resource utilization analysis

## Further Optimizations and Improvements

### Performance Enhancements
1. Parallel Processing
   - Implement multiprocessing for scene analysis
   - Optimize frame extraction pipeline
   - Add batch processing capabilities

2. Memory Management
   - Implement streaming for large videos
   - Optimize image processing pipeline
   - Add memory-efficient data structures

### Feature Enhancements
1. Advanced Text Analysis
   - Sentiment analysis of news content
   - Topic classification
   - Named entity recognition improvements

2. Enhanced Media Processing
   - Support for more video formats
   - Improved image caption quality
   - Better handling of low-quality videos

3. Output Flexibility
   - Support for multiple output formats
   - Customizable JSON structure
   - Export to different databases

### User Experience
1. Command Line Interface
   - Progress bars and status updates
   - Better error messages and logging
   - Configuration file support

2. Documentation
   - API documentation
   - Usage examples
   - Troubleshooting guide

### Monitoring and Logging
1. Performance Metrics
   - Processing time tracking
   - Resource usage monitoring
   - Quality metrics for OCR and captions

2. Error Handling
   - Comprehensive error logging
   - Recovery mechanisms
   - Validation checks