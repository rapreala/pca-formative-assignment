"""
Standardization Module

Functions for standardizing data to zero mean and unit variance.
"""

import numpy as np


def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardize data using formula: (X - μ) / σ
    
    This function transforms the input data so that each feature has a mean of 
    approximately 0 and a standard deviation of approximately 1. This is essential 
    for PCA to ensure all features contribute equally to the analysis.
    
    Args:
        data: Input data array of shape (n_samples, n_features)
        
    Returns:
        Standardized data array of same shape with mean ≈ 0 and std ≈ 1 per feature
        
    Formula:
        X_std = (X - mean(X, axis=0)) / std(X, axis=0)
        
    Note:
        Zero-variance features (where std = 0) are handled by setting their 
        standard deviation to 1 to avoid division by zero. This preserves the 
        feature but keeps it at its constant value after mean centering.
    """
    # Calculate mean for each feature (column-wise)
    mean = np.mean(data, axis=0)
    
    # Calculate standard deviation for each feature (column-wise)
    std = np.std(data, axis=0)
    
    # Handle zero-variance features to avoid division by zero
    # Set std to 1 for features with zero variance
    zero_var_mask = std == 0
    if np.any(zero_var_mask):
        std[zero_var_mask] = 1
    
    # Apply standardization formula: (X - μ) / σ
    standardized_data = (data - mean) / std
    
    return standardized_data
