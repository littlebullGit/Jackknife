"""Feature extraction for the Jackknife gesture recognizer."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .vector import Vector
from .template import Sample

@dataclass
class JackknifeFeatures:
    """Store extracted information and features from a given sample."""
    
    # Resampled points
    pts: List[Vector] = field(default_factory=list)
    
    # Processed trajectory (points or direction vectors)
    vecs: List[Vector] = field(default_factory=list)
    
    # Path length features
    path_length: float = 0.0
    normalized_path_length: float = 0.0
    
    # Bounding box features
    min_x: float = float('inf')
    max_x: float = float('-inf')
    min_y: float = float('inf')
    max_y: float = float('-inf')
    
    # Mean position features
    mean_x: float = 0.0
    mean_y: float = 0.0
    
    def extract_features(self, sample: Sample, config) -> None:
        """
        Extract features from a sample.
        
        Args:
            sample: Input gesture sample
            config: Jackknife configuration
        """
        from .mathematics import resample, path_length
        
        # Resample the trajectory
        self.pts = resample(sample.points, config.resample_count)
        
        # Extract path length features
        self.path_length = path_length(self.pts)
        if len(self.pts) > 1:
            self.normalized_path_length = self.path_length / (len(self.pts) - 1)
        
        # Convert points to numpy array for efficient computation
        points_array = np.array([p.data for p in self.pts])
        
        # Extract bounding box features
        if len(points_array) > 0:
            self.min_x = np.min(points_array[:, 0])
            self.max_x = np.max(points_array[:, 0])
            self.min_y = np.min(points_array[:, 1])
            self.max_y = np.max(points_array[:, 1])
        
        # Extract mean position features
        if len(points_array) > 0:
            self.mean_x = np.mean(points_array[:, 0])
            self.mean_y = np.mean(points_array[:, 1])
        
        # Create direction vectors if using inner product measure
        if config.use_inner_product:
            self.vecs = []
            for i in range(len(self.pts) - 1):
                direction = Vector(self.pts[i+1].data - self.pts[i].data)
                direction.normalize()
                self.vecs.append(direction)
            # Add last vector again to match length
            if self.vecs:
                self.vecs.append(self.vecs[-1].copy())
        else:
            # For Euclidean distance, just use the points
            self.vecs = self.pts.copy()
    
    def clear(self) -> None:
        """Clear all extracted features."""
        self.pts.clear()
        self.vecs.clear()
        self.path_length = 0.0
        self.normalized_path_length = 0.0
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
        self.mean_x = 0.0
        self.mean_y = 0.0
