"""
Copyright 2017 the University of Central Florida Research Foundation, Inc.
All rights reserved.

    Eugene M. Taranta II <etaranta@gmail.com>
    Amirreza Samiei <samiei@knights.ucf.edu>
    Mehran Maghoumi <mehran@cs.ucf.edu>
    Pooya Khaloo <pooya@cs.ucf.edu>
    Corey R. Pittman <cpittman@knights.ucf.edu>
    Joseph J. LaViola Jr. <jjl@cs.ucf.edu>

Subject to the terms and conditions of the Florida Public Educational
Institution non-exclusive software license.
"""

"""Jackknife gesture recognition library."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class JackknifeConfig:
    """Configuration for Jackknife's DTW-based measurement techniques."""
    
    # Number of points to resample the trajectory to
    resample_count: int = 16
    
    # Sakoe-Chiba band size (typically 10% of resample_count)
    radius: int = 2
    
    # Distance measure flags (mutually exclusive)
    use_euclidean_distance: bool = True
    use_inner_product: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_euclidean_distance and self.use_inner_product:
            raise ValueError("Cannot use both Euclidean distance and inner product")
        if not self.use_euclidean_distance and not self.use_inner_product:
            raise ValueError("Must use either Euclidean distance or inner product")

# Import main components for easier access
from .recognizer import JackknifeRecognizer
from .template import Sample, JackknifeTemplate
from .dataset import Dataset, load_dataset
from .train import train_recognizer
from .evaluate import EvaluationResults, ConfusionMatrices
from .vector import Vector

# Version information
__version__ = '1.0.0'
__author__ = 'Original C++ by UCF Research Foundation, Python port by Codeium'

# Export main components
__all__ = [
    'JackknifeConfig',
    'JackknifeRecognizer',
    'Sample',
    'JackknifeTemplate',
    'Dataset',
    'load_dataset',
    'train_recognizer',
    'EvaluationResults',
    'ConfusionMatrices',
    'Vector',
]
