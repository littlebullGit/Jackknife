"""Mathematical utilities for the Jackknife gesture recognition system."""

import numpy as np
from typing import List, Union
from dataclasses import dataclass

@dataclass
class Vector:
    """A vector class to maintain compatibility with the C++ implementation."""
    data: np.ndarray

    def __add__(self, other):
        return Vector(self.data + other.data)
    
    def __truediv__(self, scalar):
        return Vector(self.data / scalar)

def z_normalize(points: List[Vector]) -> None:
    """Component-wise z-score normalize each point of a time series."""
    points_array = np.array([p.data for p in points])
    mean = np.mean(points_array, axis=0)
    std = np.std(points_array, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    for i, point in enumerate(points):
        point.data = (point.data - mean) / std

def path_length(points: List[Vector]) -> float:
    """Calculate the path length through m-dimensional space."""
    total_length = 0.0
    for i in range(len(points) - 1):
        diff = points[i+1].data - points[i].data
        total_length += np.sqrt(np.sum(diff * diff))
    return total_length

def resample(points: List[Vector], n: int, variance: float = 0.0) -> List[Vector]:
    """
    Resample a trajectory either uniformly or stochastically.
    
    Args:
        points: Input trajectory points
        n: Number of points in resampled trajectory
        variance: If 0, uniform resampling. If > 0, stochastic resampling.
    
    Returns:
        Resampled trajectory points
    """
    if len(points) < 2:
        return points.copy()
        
    # Calculate the total path length
    total_length = path_length(points)
    if total_length == 0:
        return [Vector(points[0].data.copy()) for _ in range(n)]
    
    # Calculate desired spacing
    spacing = total_length / (n - 1)
    
    # Initialize result
    result = []
    current_point = Vector(points[0].data.copy())
    result.append(current_point)
    
    point_idx = 0
    remaining_segment = 0.0
    
    for i in range(1, n-1):
        target_distance = spacing
        if variance > 0:
            # Add noise to spacing for stochastic resampling
            target_distance += np.random.normal(0, variance * spacing)
        
        while target_distance > 0 and point_idx < len(points) - 1:
            if remaining_segment == 0:
                diff = points[point_idx + 1].data - points[point_idx].data
                remaining_segment = np.sqrt(np.sum(diff * diff))
                if remaining_segment == 0:
                    point_idx += 1
                    continue
                    
            if remaining_segment >= target_distance:
                t = target_distance / remaining_segment
                current_point = Vector(
                    points[point_idx].data * (1-t) + points[point_idx + 1].data * t
                )
                remaining_segment -= target_distance
                target_distance = 0
            else:
                target_distance -= remaining_segment
                point_idx += 1
                remaining_segment = 0
                
        result.append(current_point)
    
    # Add last point
    result.append(Vector(points[-1].data.copy()))
    return result

def gpsr(points: List[Vector], n: int, variance: float, remove_cnt: int) -> List[Vector]:
    """
    Perform gesture path stochastic resampling (GPSR).
    
    Reference:
    Eugene M. Taranta II, et al. "A Rapid Prototyping Approach to Synthetic Data 
    Generation For Improved 2D Gesture Recognition" UIST 2016
    
    Args:
        points: Input trajectory points
        n: Number of points in resampled trajectory
        variance: Controls amount of noise in resampling
        remove_cnt: Number of points to randomly remove before resampling
    
    Returns:
        Synthetic variation of the input trajectory
    """
    if len(points) <= remove_cnt:
        return points.copy()
        
    # Randomly remove points
    indices = np.random.choice(len(points), len(points) - remove_cnt, replace=False)
    indices.sort()
    reduced_points = [points[i] for i in indices]
    
    # Perform stochastic resampling
    return resample(reduced_points, n, variance)
