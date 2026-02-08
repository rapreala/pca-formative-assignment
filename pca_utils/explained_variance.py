"""
Explained Variance Module

Functions for calculating explained variance and selecting components.
"""

import numpy as np


def calculate_explained_variance(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate explained variance percentage for each component.
    
    Args:
        eigenvalues: Sorted eigenvalues in descending order
        
    Returns:
        Tuple of (explained_variance, cumulative_variance)
        - explained_variance: Percentage of variance for each component
        - cumulative_variance: Cumulative sum of explained variance
        
    Formula:
        explained_variance[i] = (eigenvalues[i] / sum(eigenvalues)) * 100
    """
    # Calculate total variance
    total_variance = np.sum(eigenvalues)
    
    # Calculate explained variance percentage for each component
    explained_variance = (eigenvalues / total_variance) * 100
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(explained_variance)
    
    return explained_variance, cumulative_variance


def select_components_by_variance(cumulative_variance: np.ndarray, 
                                  threshold: float = 0.9) -> int:
    """
    Determine number of components to retain based on variance threshold.
    
    Args:
        cumulative_variance: Cumulative explained variance array
        threshold: Variance threshold (e.g., 0.9 for 90%)
        
    Returns:
        Number of components to retain
    """
    # Convert threshold to percentage if needed
    threshold_percentage = threshold * 100
    
    # Find minimum k where cumulative_variance[k-1] >= threshold * 100
    num_components = np.argmax(cumulative_variance >= threshold_percentage) + 1
    
    return num_components
