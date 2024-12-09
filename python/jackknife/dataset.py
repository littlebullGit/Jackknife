"""Dataset management for the Jackknife gesture recognizer."""

import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from .template import Sample
from .vector import Vector

class Dataset:
    """A class to help manage all samples of a dataset."""
    
    def __init__(self):
        """Initialize an empty dataset."""
        self.gestures: List[str] = []
        self.subjects: List[str] = []
        self.samples: List[Sample] = []
        self.samples_by_gesture: List[List[Sample]] = []
    
    def add_gesture(self, gesture_name: str) -> int:
        """
        Add gesture name to database if it does not already exist.
        
        Args:
            gesture_name: Name of the gesture
            
        Returns:
            Enumerated gesture ID
        """
        try:
            return self.gestures.index(gesture_name)
        except ValueError:
            gesture_id = len(self.gestures)
            self.gestures.append(gesture_name)
            self.samples_by_gesture.append([])
            return gesture_id
    
    def add_subject(self, subject_name: str) -> int:
        """
        Add subject to database if it does not already exist.
        
        Args:
            subject_name: Name of the subject
            
        Returns:
            Enumerated subject ID
        """
        try:
            return self.subjects.index(subject_name)
        except ValueError:
            subject_id = len(self.subjects)
            self.subjects.append(subject_name)
            return subject_id
            
    def add_sample(self, sample: Sample, subject_id: int, gesture_id: int):
        """
        Add a new sample to the dataset.
        
        Args:
            sample: The gesture sample to add
            subject_id: ID of the subject who performed the gesture
            gesture_id: ID of the gesture class
        """
        self.samples.append(sample)
        self.samples_by_gesture[gesture_id].append(sample)
    
    @property
    def sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.samples)
    
    @property
    def gesture_count(self) -> int:
        """Get number of gesture classes."""
        return len(self.gestures)
    
    @property
    def subject_count(self) -> int:
        """Get number of subjects."""
        return len(self.subjects)
    
    def gesture_name_to_id(self, gesture_name: str) -> int:
        """
        Convert gesture name to its ID.
        
        Args:
            gesture_name: Name of the gesture
            
        Returns:
            Gesture ID
            
        Raises:
            ValueError: If gesture name not found
        """
        try:
            return self.gestures.index(gesture_name)
        except ValueError:
            raise ValueError(f"Gesture '{gesture_name}' not found in dataset")
    
    def dump_catalog(self):
        """Print information about the dataset."""
        print(f"Subject Count: {self.subject_count}")
        print(f"Sample Count: {self.sample_count}")
        print(f"Gesture Count: {self.gesture_count}")
        
        print("\nGestures:")
        for gesture_id, samples in enumerate(self.samples_by_gesture):
            print(f"  {self.gestures[gesture_id]}: {len(samples)} samples")
            
        print("\nSubjects:")
        for subject_id, subject in enumerate(self.subjects):
            subject_samples = sum(1 for s in self.samples if s.user_id == subject_id)
            print(f"  {subject}: {subject_samples} samples")

def load_dataset(path: str) -> Dataset:
    """
    Load a dataset from a directory.
    
    Expected directory structure:
    <path>/Sub_* /gesture_name/ex_*
    
    Args:
        path: Path to dataset directory
        
    Returns:
        Loaded dataset
    """
    dataset = Dataset()
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
        
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    
    # Find all subject directories
    subject_dirs = list(path.glob("Sub_*"))
    if not subject_dirs:
        raise ValueError(f"No subject directories (Sub_*) found in {path}")
    
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
            
        subject_id = dataset.add_subject(subject_dir.name)
        
        # Find all gesture directories
        gesture_dirs = list(subject_dir.glob("*"))
        if not gesture_dirs:
            print(f"Warning: No gesture directories found in {subject_dir}")
            continue
            
        for gesture_dir in gesture_dirs:
            if not gesture_dir.is_dir():
                continue
                
            gesture_id = dataset.add_gesture(gesture_dir.name)
            
            # Load all example files
            example_files = list(gesture_dir.glob("ex_*"))
            if not example_files:
                print(f"Warning: No example files (ex_*) found in {gesture_dir}")
                continue
                
            for example_file in example_files:
                if not example_file.is_file():
                    continue
                    
                try:
                    # Load points from file
                    points = _load_points_from_file(example_file)
                    
                    # Create and add sample
                    sample = Sample(
                        gesture_id=gesture_id,
                        points=points,
                        user_id=subject_id
                    )
                    dataset.add_sample(sample, subject_id, gesture_id)
                except Exception as e:
                    print(f"Warning: Failed to load {example_file}: {e}")
    
    if dataset.sample_count == 0:
        raise ValueError(f"No valid samples found in dataset directory: {path}")
    
    return dataset

def _load_points_from_file(file_path: Path) -> List[Vector]:
    """
    Load points from a gesture example file.
    
    Args:
        file_path: Path to example file
        
    Returns:
        List of points as Vectors from all sequences in the file
    """
    with open(file_path, 'r') as f:
        # Read gesture name
        gesture_name = f.readline().strip()
        
        # Read point count
        pt_cnt_str = f.readline().strip()
        pt_cnt = int(pt_cnt_str)
        
        # Read hashes
        hash_line = f.readline().strip()
        assert hash_line == "####"
        
        points = []  # Final list of vectors
        pt = []  # Current point coordinates
        
        while True:
            # Read next line
            line = f.readline()
            
            # Check for empty line or separator
            if not line or line.strip() == "####":
                # Add pt to trajectory if we have coordinates
                if len(pt) == 3:  # Only add if we have x,y,z
                    points.append(Vector(np.array(pt)))
                pt = []
                
                # End of file
                if not line:
                    break
                    
                continue
            
            # Parse x,y,z coordinates
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                # Handle comma-separated values
                if ',' in line:
                    coords = [float(x) for x in line.split(',')]
                else:
                    # Parse space-separated values
                    coords = [float(x) for x in line.split()]
                
                if len(coords) >= 3:
                    pt = coords[:3]  # Take first 3 coordinates
            except ValueError:
                # Skip invalid lines
                continue
                
        # Verify point count
        if len(points) != pt_cnt:
            raise ValueError(f"Expected {pt_cnt} points but got {len(points)}")
            
        return points
