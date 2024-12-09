"""Training utilities for the Jackknife gesture recognizer."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .vector import Vector
from .template import Sample, JackknifeTemplate
from .mathematics import gpsr
from tqdm import tqdm

@dataclass
class Distributions:
    """Helper class to manage statistics when calculating thresholds."""
    
    # Range of the histogram [0.0, max_score]
    max_score: float
    
    # Distribution of negative and positive sample scores
    neg: Vector
    pos: Vector
    
    # Number of bins in histogram
    bin_count: int
    
    def __init__(self, max_score: float, bin_count: int = 100):
        """
        Initialize distributions.
        
        Args:
            max_score: Maximum score value
            bin_count: Number of bins in histogram
        """
        self.max_score = max_score
        self.bin_count = bin_count
        self.neg = Vector(np.zeros(bin_count))
        self.pos = Vector(np.zeros(bin_count))
        
    def add_score(self, score: float, is_positive: bool) -> None:
        """
        Add a score to either positive or negative distribution.
        
        Args:
            score: Score to add
            is_positive: True if score is from positive sample
        """
        if score > self.max_score:
            score = self.max_score
            
        # Convert score to bin index
        bin_idx = int((score / self.max_score) * (self.bin_count - 1))
        
        if is_positive:
            self.pos.data[bin_idx] += 1
        else:
            self.neg.data[bin_idx] += 1
    
    def find_threshold(self, beta: float) -> float:
        """
        Find optimal threshold based on distributions.
        
        Args:
            beta: Weight factor for false positives vs false negatives
            
        Returns:
            Optimal threshold value
        """
        # Normalize distributions
        pos_sum = np.sum(self.pos.data)
        neg_sum = np.sum(self.neg.data)
        
        if pos_sum > 0:
            self.pos.data /= pos_sum
        if neg_sum > 0:
            self.neg.data /= neg_sum
            
        # Find threshold that minimizes weighted error
        min_error = float('inf')
        best_threshold = 0.0
        
        for i in range(self.bin_count):
            threshold = (i / self.bin_count) * self.max_score
            
            # Calculate errors
            false_neg = np.sum(self.pos.data[i:])  # positive samples below threshold
            false_pos = np.sum(self.neg.data[:i])  # negative samples above threshold
            
            # Compute weighted error
            error = false_neg + beta * false_pos
            
            if error < min_error:
                min_error = error
                best_threshold = threshold
                
        return best_threshold

def train_recognizer(recognizer, gpsr_n: int = 10, gpsr_r: int = 2, beta: float = 1.0) -> None:
    """
    Train the Jackknife recognizer by generating synthetic samples and computing thresholds.
    
    Args:
        recognizer: JackknifeRecognizer instance to train
        gpsr_n: Number of synthetic samples to generate per template
        gpsr_r: Number of points to remove in GPSR
        beta: Weight factor for false positives vs false negatives
    """
    # Generate synthetic samples using GPSR
    synthetic_templates = []
    for template in tqdm(recognizer.templates, desc="Generating Synthetic Templates"):
    # for template in recognizer.templates:
        # Generate variations
        for _ in range(gpsr_n):
            points = gpsr(
                template.points.copy(),
                recognizer.config.resample_count,
                0.25,  # variance
                gpsr_r
            )
            
            # Create synthetic template
            synthetic = JackknifeTemplate(
                sample=Sample(
                    gesture_id=template.gesture_id,
                    points=points
                ),
                gesture_id=template.gesture_id
            )
            synthetic.preprocess(recognizer.config)
            synthetic_templates.append(synthetic)
    
    # Collect score distributions
    distributions = Distributions(max_score=1.0)
    
    # Compare each template against all others
    for i, template in enumerate(tqdm(recognizer.templates + synthetic_templates, desc="Comparing Templates")):
    # for i, template in enumerate(recognizer.templates + synthetic_templates):
        for j, other in enumerate(tqdm(recognizer.templates + synthetic_templates, desc="Inner Comparing Templates")):
        # for j, other in enumerate(recognizer.templates + synthetic_templates):
            if i == j:
                continue
                
            # Compute DTW distance
            dist = recognizer._dtw(template.points, other.points)
            
            # Convert distance to similarity score
            score = 1.0 / (1.0 + dist)
            
            # Add to distributions
            is_positive = template.gesture_id == other.gesture_id
            distributions.add_score(score, is_positive)
    
    # Find optimal threshold
    threshold = distributions.find_threshold(beta)
    
    # Set threshold in recognizer
    recognizer.threshold = threshold
