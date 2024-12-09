"""Signal filtering utilities for the Jackknife gesture recognizer."""

import numpy as np
from typing import Optional
from .vector import Vector

class ExponentialMovingAverage:
    """
    An effective, easy to use low pass filter that is good for continuous data.
    Implements an exponential moving average filter.
    """
    
    def __init__(self, cut_off_frequency_hz: float = 1.0):
        """
        Initialize the filter.
        
        Args:
            cut_off_frequency_hz: Cut-off frequency in Hz. 1 Hz is a good starting point.
        """
        self.pt: Optional[Vector] = None
        self.cut_off_frequency_hz = cut_off_frequency_hz
        self._alpha = 0.0
        
    def set_sampling_rate(self, sampling_rate_hz: float) -> None:
        """
        Set the sampling rate to compute the correct alpha value.
        
        Args:
            sampling_rate_hz: Sampling rate in Hz
        """
        # Compute alpha based on cut-off frequency and sampling rate
        # Alpha controls how much weight is given to new samples vs old ones
        rc = 1.0 / (2.0 * np.pi * self.cut_off_frequency_hz)
        dt = 1.0 / sampling_rate_hz
        self._alpha = dt / (rc + dt)
        
    def filter(self, new_pt: Vector) -> Vector:
        """
        Apply the filter to a new point.
        
        Args:
            new_pt: New point to filter
            
        Returns:
            Filtered point
        """
        if self.pt is None:
            # First point, initialize the filter
            self.pt = new_pt.copy()
        else:
            # Apply exponential moving average
            # y[n] = α * x[n] + (1-α) * y[n-1]
            self.pt = new_pt * self._alpha + self.pt * (1.0 - self._alpha)
            
        return self.pt
    
    def reset(self) -> None:
        """Reset the filter state."""
        self.pt = None
