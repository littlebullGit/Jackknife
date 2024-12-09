"""Evaluation framework for the Jackknife gesture recognizer."""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from typing import List, Dict
from .vector import Vector

class ErrorType(IntEnum):
    """Enumeration of error types for confusion matrices."""
    TRUE_POSITIVE = 0
    FALSE_NEGATIVE = 1
    FALSE_POSITIVE = 2
    TRUE_NEGATIVE = 3

@dataclass
class EvaluationResults:
    """Results from confusion matrix analysis."""
    recall: float = 0.0
    precision: float = 0.0
    fall_out: float = 0.0
    f1: float = 0.0
    
    def print(self):
        """Print the evaluation results."""
        print(f"Recall:    {self.recall * 100.0:3.2f}%")
        print(f"Precision: {self.precision * 100.0:3.2f}%")
        print(f"Fall Out:  {self.fall_out * 100.0:3.2f}%")
        print(f"F1 Score:  {self.f1 * 100.0:3.2f}%")

class ConfusionMatrices:
    """Helper class for tracking confusion matrices."""
    
    def __init__(self, num_gestures: int):
        """
        Initialize confusion matrices.
        
        Args:
            num_gestures: Number of different gesture classes
        """
        self.entries = 0
        self.confusion_matrices = [
            Vector(np.zeros(4)) for _ in range(num_gestures)
        ]
    
    def add_result(self, expected_id: int, detected_id: int) -> None:
        """
        Add a single result to confusion matrices.
        
        Args:
            expected_id: The true gesture class ID
            detected_id: The predicted gesture class ID (-2 for special case in sessions)
        """
        assert expected_id < len(self.confusion_matrices)
        assert detected_id < len(self.confusion_matrices) or detected_id == -2
        
        self.entries += 1
        
        for gesture_id in range(len(self.confusion_matrices)):
            if gesture_id == expected_id:
                # Special case for sessions logic
                if detected_id == -2:
                    self.confusion_matrices[gesture_id].data[ErrorType.FALSE_POSITIVE] += 1.0
                elif gesture_id == detected_id:
                    self.confusion_matrices[gesture_id].data[ErrorType.TRUE_POSITIVE] += 1.0
                else:
                    self.confusion_matrices[gesture_id].data[ErrorType.FALSE_NEGATIVE] += 1.0
            else:
                if gesture_id == detected_id:
                    self.confusion_matrices[gesture_id].data[ErrorType.FALSE_POSITIVE] += 1.0
                else:
                    self.confusion_matrices[gesture_id].data[ErrorType.TRUE_NEGATIVE] += 1.0
    
    def merge_results(self, other: 'ConfusionMatrices') -> None:
        """
        Merge another confusion matrix into this one.
        
        Used to average together confusion matrices over different tests.
        
        Args:
            other: Another ConfusionMatrices instance to merge
        """
        assert len(self.confusion_matrices) == len(other.confusion_matrices)
        self.entries += 1
        
        for i, other_matrix in enumerate(other.confusion_matrices):
            matrix_sum = np.sum(other_matrix.data)
            if matrix_sum > 0:
                self.confusion_matrices[i].data += other_matrix.data / matrix_sum
    
    def get_results(self) -> EvaluationResults:
        """
        Extract statistics from the confusion matrices.
        
        Returns:
            EvaluationResults containing precision, recall, etc.
        """
        # Aggregate confusion matrix
        cm = np.sum([m.data for m in self.confusion_matrices], axis=0)
        
        results = EvaluationResults()
        
        # Calculate metrics
        tp = cm[ErrorType.TRUE_POSITIVE]
        fp = cm[ErrorType.FALSE_POSITIVE]
        fn = cm[ErrorType.FALSE_NEGATIVE]
        tn = cm[ErrorType.TRUE_NEGATIVE]
        
        # Recall (True Positive Rate)
        results.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Precision
        results.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Fall-out (False Positive Rate)
        results.fall_out = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # F1 Score
        results.f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        return results
