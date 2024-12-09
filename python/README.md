# Jackknife

A Python implementation of the Jackknife gesture recognition library, converted from the original C++ version.

## Overview
This library implements various DTW-based measurement techniques for gesture recognition. It supports different approaches that can be configured based on your specific needs.

## Features
- Dynamic Time Warping (DTW) based gesture recognition
- Configurable distance measures (Euclidean or inner product)
- Gesture Path Stochastic Resampling (GPSR) for synthetic data generation
- Comprehensive evaluation framework
- Support for user-independent and session-based evaluation

## Requirements
- Python 3.6+
- NumPy
- SciPy

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Basic usage example:
```python
from jackknife import JackknifeConfig
from jackknife.recognizer import JackknifeRecognizer
from jackknife.template import Sample
from jackknife.vector import Vector
import numpy as np

# Create configuration
config = JackknifeConfig(
    resample_count=16,
    radius=2,
    use_euclidean_distance=True
)

# Create recognizer
recognizer = JackknifeRecognizer(config)

# Add templates
template_points = [Vector(np.array([x, y])) for x, y in gesture_data]
template = Sample(gesture_id=0, points=template_points)
recognizer.add_template(template)

# Recognize gesture
gesture_id, confidence = recognizer.recognize(input_points)
```

For a complete example, see `main.py`.

## Structure
- `jackknife/`: Core gesture recognition implementation
  - `recognizer.py`: Main recognition engine
  - `template.py`: Gesture template management
  - `mathematics.py`: Core mathematical operations
  - `features.py`: Feature extraction
  - `filters.py`: Signal filtering utilities
  - `evaluate.py`: Evaluation framework
  - `dataset.py`: Dataset management
  - `train.py`: Training utilities

## Command Line Interface
The library includes a command-line tool for evaluation:

```bash
python main.py --data /path/to/dataset --resample 16 --radius 2 --euclidean
```

Arguments:
- `--data`: Path to dataset directory
- `--resample`: Number of points to resample to (default: 16)
- `--radius`: Sakoe-Chiba band radius (default: 2)
- `--euclidean`: Use Euclidean distance (default: inner product)

## License
Subject to the terms and conditions of the Florida Public Educational Institution non-exclusive software license.
