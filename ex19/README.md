# Efficient Subwindow Search (ESS) Template Matching

## Overview

This project implements the Efficient Subwindow Search (ESS) algorithm for template matching, inspired by the research paper "Beyond Sliding Windows: Object Localization by Efficient Subwindow Search" by Lampert et al.

The implementation provides an alternative to traditional sliding window template matching by using a branch-and-bound search strategy to efficiently locate template images within a target image.

## Features

- Efficient object localization using branch-and-bound search
- Normalized cross-correlation for template matching
- Finds multiple template locations
- Guaranteed global optimum search
- Implemented without external computer vision libraries (except OpenCV)

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ess-template-matching.git
cd ess-template-matching
```

2. Install dependencies:
```bash
pip install opencv-python numpy
```

## Usage

```python
import cv2
from ess_template_matching import locate_template_ess

# Load images
target_image = cv2.imread('target.jpg')
template_image = cv2.imread('template.jpg')

# Find template locations
matches = locate_template_ess(target_image, template_image)

# Draw rectangles around matches
for (x, y, w, h) in matches:
    cv2.rectangle(target_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Matches', target_image)
cv2.waitKey(0)
```

## Algorithm Details

The Efficient Subwindow Search (ESS) algorithm:
- Uses a priority queue to search image regions
- Splits search space along largest coordinate range
- Computes upper and lower bounds of template correlation
- Guarantees finding globally optimal matches

## Performance Considerations

- More computationally intensive than standard template matching
- Best for precise, complex template matching requirements
- Slower than built-in OpenCV methods
- Ideal for scenarios requiring guaranteed global optimum

## Limitations

- Does not handle rotation or scale variations
- Performance degrades with large images
- Less efficient for simple template matching tasks
