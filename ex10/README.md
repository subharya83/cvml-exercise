# Image Captioning with ResNet50 and LSTM

This project implements an image captioning system using a ResNet50 backbone for feature 
extraction and an LSTM network for caption generation.

## Problem Design

The system aims to generate natural language descriptions of images by:
- Using transfer learning with a pre-trained ResNet50 model
- Implementing sequence generation with LSTM architecture
- Supporting variable-length caption generation
- Maintaining a vocabulary system for word encoding/decoding

Key constraints:
- Input images must be RGB format
- Images are resized to 224x224 pixels
- Captions are limited to 20 words maximum
- Minimum word frequency threshold of 5 for vocabulary inclusion

## Data Preparation

### Input Processing
1. Image preprocessing:
   - Resize to 224x224
   - Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Convert to tensor format

### Vocabulary Processing
1. Caption text processing:
   - Tokenization
   - Special token handling (<pad>, <start>, <end>, <unk>)
   - Frequency thresholding
   - Word-to-index mapping

## Code Organization

```
image_captioning/
├── models/
│   ├── encoder.py        # ResNet50 backbone
│   └── decoder.py        # LSTM architecture
├── utils/
│   ├── vocabulary.py     # Vocabulary management
│   └── data_loader.py    # Dataset handling
├── config.py            # Configuration parameters
└── train.py            # Training script
```

Key Components:
- ImageCaptioner: Main model combining ResNet50 and LSTM
- Vocabulary: Handles word indexing and vocabulary management
- Data processing utilities: Image transformation and caption preprocessing

## Computational flow
```
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Image Input      | ----> |  Image Preprocess | ----> |  ResNet50 Feature |
|  (PIL Image)      |       |  (Resize, Norm)   |       |  Extraction       |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
        |                                                      |
        |                                                      |
        v                                                      v
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Linear Layer     | ----> |  LSTM for         | <---- |  Linear Layer     |
|  (Embedding)      |       |  Sequence Gen     |       |  (Word Prediction)|
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
        |                           |                           |
        |                           |                           |
        v                           v                           v
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Vocabulary       | <---- |  Caption          | <---- |  LSTM Hidden      |
|  (idx2word)       |       |  Generation       |       |  States           |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
```
## Test Cases

1. Vocabulary Creation:
```python
vocab = Vocabulary(freq_threshold=5)
assert len(vocab) == 4  # Initial special tokens
vocab.build_vocabulary(["test caption", "another caption"])
```

2. Image Processing:
```python
image = Image.open("test.jpg")
processed = transform(image)
assert processed.shape == (3, 224, 224)
```

3. Caption Generation:
```python
model = ImageCaptioner(embed_size=256, hidden_size=512, vocab_size=len(vocab))
caption = model.generate_caption(image, vocab)
assert isinstance(caption, str)
```

## Further Optimizations

Potential improvements:
1. Model Architecture:
   - Implement attention mechanism
   - Add beam search for caption generation
   - Experiment with different backbone networks

2. Training Enhancements:
   - Implement curriculum learning
   - Add data augmentation
   - Implement mixed precision training

3. Evaluation Metrics:
   - Add BLEU score calculation
   - Implement METEOR metric
   - Add CIDEr score evaluation

4. Production Readiness:
   - Model quantization
   - ONNX export support
   - API endpoint implementation