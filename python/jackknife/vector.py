"""Vector class implementation for Jackknife."""

import numpy as np
from typing import Union, List

class Vector:
    """A wrapper around numpy arrays that provides mathematical operations for points and vectors."""
    
    def __init__(self, data_or_size: Union[np.ndarray, int, List[float]], fill_value: float = 0.0):
        """
        Initialize a Vector.
        
        Args:
            data_or_size: Either initial data as numpy array/list or size of vector
            fill_value: Value to fill vector with if size is provided
        """
        if isinstance(data_or_size, (np.ndarray, list)):
            self.data = np.array(data_or_size, dtype=np.float64)
        else:
            self.data = np.full(data_or_size, fill_value, dtype=np.float64)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> float:
        return self.data[idx]
    
    def __setitem__(self, idx: int, value: float):
        self.data[idx] = value
    
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.data + other.data)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.data - other.data)
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.data * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(self.data / scalar)
    
    def __iadd__(self, other: 'Vector') -> 'Vector':
        self.data += other.data
        return self
    
    def __isub__(self, other: 'Vector') -> 'Vector':
        self.data -= other.data
        return self
    
    def __imul__(self, scalar: float) -> 'Vector':
        self.data *= scalar
        return self
    
    def __itruediv__(self, scalar: float) -> 'Vector':
        self.data /= scalar
        return self
    
    def dot(self, other: 'Vector') -> float:
        """Compute dot product with another vector."""
        return np.dot(self.data, other.data)
    
    def magnitude(self) -> float:
        """Compute magnitude (L2 norm) of the vector."""
        return np.sqrt(self.dot(self))
    
    def normalize(self) -> None:
        """Normalize the vector in place."""
        mag = self.magnitude()
        if mag > 0:
            self.data /= mag
    
    def copy(self) -> 'Vector':
        """Create a deep copy of the vector."""
        return Vector(self.data.copy())
    
    def clear(self) -> None:
        """Set all components to zero."""
        self.data.fill(0.0)
    
    def resize(self, new_size: int, fill_value: float = 0.0) -> None:
        """Resize the vector."""
        old_size = len(self.data)
        new_data = np.full(new_size, fill_value, dtype=np.float64)
        if old_size > 0:
            new_data[:min(old_size, new_size)] = self.data[:min(old_size, new_size)]
        self.data = new_data
