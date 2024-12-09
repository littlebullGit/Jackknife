"""Example usage of the Jackknife gesture recognizer."""

import argparse
import os
import pickle
from pathlib import Path
import numpy as np
from jackknife import (
    JackknifeConfig,
    JackknifeRecognizer,
    Sample,
    train_recognizer,
    Vector,
    load_dataset
)

def create_example_dataset():
    """Create a simple example dataset with basic gestures."""
    # Create a circle gesture
    t = np.linspace(0, 2*np.pi, 32)
    circle = [Vector(np.array([np.cos(x), np.sin(x)])) for x in t]
    
    # Create a square gesture
    square_pts = [
        [1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]
    ]
    square = [Vector(np.array(pt)) for pt in square_pts]
    
    # Create a triangle gesture
    triangle_pts = [
        [0, 1], [-1, -1], [1, -1], [0, 1]
    ]
    triangle = [Vector(np.array(pt)) for pt in triangle_pts]
    
    # Create samples
    samples = [
        Sample(gesture_id=0, points=circle),
        Sample(gesture_id=1, points=square),
        Sample(gesture_id=2, points=triangle)
    ]
    
    # Create some variations for testing
    test_samples = []
    
    # Add noise to create variations
    for sample in samples:
        for _ in range(3):  # Create 3 variations of each
            noisy_points = []
            for pt in sample.points:
                noise = np.random.normal(0, 0.1, 2)  # Small Gaussian noise
                noisy_points.append(Vector(pt.data + noise))
            test_samples.append(Sample(
                gesture_id=sample.gesture_id,
                points=noisy_points
            ))
    
    return samples, test_samples

def main():
    """Run gesture recognition evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Jackknife gesture recognition')
    parser.add_argument('--data', type=str,
                      help='Path to dataset directory (optional)')
    parser.add_argument('--resample', type=int, default=16,
                      help='Number of points to resample to')
    parser.add_argument('--radius', type=int, default=2,
                      help='Sakoe-Chiba band radius')
    parser.add_argument('--euclidean', action='store_true', default=False,
                      help='Use Euclidean distance (default: inner product)')
    parser.add_argument('--model_path', type=str, default='trained_model.pkl',
                      help='Path to save trained model')
    args = parser.parse_args()
    
    # Create recognizer
    config = JackknifeConfig(
        resample_count=args.resample,
        radius=args.radius,
        use_euclidean_distance=args.euclidean,
        use_inner_product=not args.euclidean
    )
    recognizer = JackknifeRecognizer(config)
    
    if args.data:
        # Convert to absolute path if needed
        data_path = args.data
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)
            
        # Load dataset from directory
        print(f"Loading dataset from {data_path}...")
        dataset = load_dataset(data_path)
        dataset.dump_catalog()
        
        # Add templates from dataset
        print("\nAdding templates...")
        for gesture_id, samples in enumerate(dataset.samples_by_gesture):
            print(f"Gesture {gesture_id}: {len(samples)} samples")
            for sample in samples:
                recognizer.add_template(sample)
                
        # Use dataset samples for testing
        test_samples = dataset.samples
                
    else:
        # Use example dataset
        print("Using example dataset...")
        templates, test_samples = create_example_dataset()
        
        # Add templates
        print("\nAdding templates...")
        for template in templates:
            recognizer.add_template(template)
            
        # Print gesture counts
        gesture_counts = {}
        for sample in templates:
            gesture_counts[sample.gesture_id] = gesture_counts.get(sample.gesture_id, 0) + 1
        for gesture_id, count in gesture_counts.items():
            print(f"Gesture {gesture_id}: {count} samples")
    
    # Train recognizer
    print("\nTraining recognizer...")
    train_recognizer(recognizer)
    
    # Save trained model
    print(f"\nSaving trained model to {args.model_path}...")
    model_data = {
        'recognizer': recognizer,
        'config': config,
        'gesture_info': {
            'gesture_ids': sorted(list(set(s.gesture_id for s in test_samples))),
            'total_samples': len(test_samples)
        }
    }
    with open(args.model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved successfully!")
    
    # Evaluate
    print("\nEvaluating recognition...")
    correct = 0
    total = 0
    
    for sample in test_samples:
        gesture_id, confidence = recognizer.recognize(sample.points)
        total += 1
        if gesture_id == sample.gesture_id:
            correct += 1
            print(f" Correctly recognized gesture {gesture_id} with confidence {confidence:.2f}")
        else:
            print(f" Misclassified gesture {sample.gesture_id} as {gesture_id} with confidence {confidence:.2f}")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nResults:")
    print(f"Total samples: {total}")
    print(f"Correct recognitions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
