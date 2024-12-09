"""Template classes for the Jackknife gesture recognizer."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .vector import Vector

@dataclass
class Sample:
    """A gesture sample used for training or recognition."""
    gesture_id: int
    points: List[Vector]
    user_id: Optional[int] = None
    session_id: Optional[int] = None

@dataclass
class JackknifeTemplate:
    """All information about a single template, including cached results used by the recognizer."""
    
    # Original sample used to create this template
    sample: Sample
    
    # The enumerated gesture class id
    gesture_id: int
    
    # Resampled points for this template
    points: List[Vector] = field(default_factory=list)
    
    # Lower bound cache used by recognizer
    lb: float = float('inf')
    
    # Various feature caches
    mean_x: float = 0.0
    mean_y: float = 0.0
    path_length: float = 0.0
    
    def preprocess(self, config):
        """Preprocess the template points according to configuration."""
        from .mathematics import resample, z_normalize, path_length
        
        # Create resampled points
        self.points = resample(self.sample.points, config.resample_count)
        
        # Normalize if needed
        z_normalize(self.points)
        
        # Cache features
        points_array = np.array([p.data for p in self.points])
        self.mean_x = np.mean(points_array[:, 0])
        self.mean_y = np.mean(points_array[:, 1])
        self.path_length = path_length(self.points)
