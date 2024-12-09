"""Jackknife gesture recognizer implementation."""

from typing import List, Tuple, Optional
import numpy as np
from .vector import Vector
from .template import JackknifeTemplate, Sample

class JackknifeRecognizer:
    """Main gesture recognizer class implementing the Jackknife algorithm."""
    
    def __init__(self, config):
        """Initialize the recognizer with given configuration."""
        self.config = config
        self.templates: List[JackknifeTemplate] = []
    
    def add_template(self, sample: Sample) -> None:
        """Add a new template to the recognizer."""
        template = JackknifeTemplate(
            sample=sample,
            gesture_id=sample.gesture_id
        )
        template.preprocess(self.config)
        self.templates.append(template)
    
    def recognize(self, points: List[Vector]) -> Tuple[int, float]:
        """
        Recognize a gesture from input points.
        
        Returns:
            Tuple of (gesture_id, confidence_score)
        """
        from .mathematics import resample, z_normalize
        
        # Preprocess input
        query = resample(points, self.config.resample_count)
        z_normalize(query)
        
        # Find best match using DTW
        best_dist = float('inf')
        best_gesture = -1
        
        for template in self.templates:
            dist = self._dtw(query, template.points)
            if dist < best_dist:
                best_dist = dist
                best_gesture = template.gesture_id
        
        # Convert distance to confidence score (0 to 1)
        confidence = 1.0 / (1.0 + best_dist) if best_dist != float('inf') else 0.0
        
        return best_gesture, confidence
    
    def _dtw(self, v1: List[Vector], v2: List[Vector]) -> float:
        """
        Compute Dynamic Time Warping distance between two sequences.
        
        Uses Sakoe-Chiba band to constrain the warping path.
        """
        n, m = len(v1), len(v2)
        
        # Initialize cost matrix
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        
        # Fill cost matrix using dynamic programming
        for i in range(1, n + 1):
            for j in range(
                max(1, i - self.config.radius),
                min(m + 1, i + self.config.radius + 1)
            ):
                # Find minimum cost path to extend
                min_cost = min(
                    cost[i-1, j],     # repeat v1 element
                    cost[i, j-1],     # repeat v2 element
                    cost[i-1, j-1]    # match elements
                )
                
                # Add cost between current elements
                if self.config.use_inner_product:
                    # Using negative dot product as distance
                    cost[i, j] = min_cost + (1.0 - v1[i-1].dot(v2[j-1]))
                else:
                    # Using squared Euclidean distance
                    diff = v1[i-1].data - v2[j-1].data
                    cost[i, j] = min_cost + np.sum(diff * diff)
        
        return cost[n, m]
    
    def clear_templates(self) -> None:
        """Remove all templates from the recognizer."""
        self.templates.clear()
    
    def num_templates(self) -> int:
        """Get the number of templates in the recognizer."""
        return len(self.templates)
    
    def get_templates(self) -> List[JackknifeTemplate]:
        """Get all templates in the recognizer."""
        return self.templates.copy()
