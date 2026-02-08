"""
Data Projection Module

This module provides functions for projecting data onto principal components.
"""

import numpy as np


def project_data(standardized_data: np.ndarray, 
                eigenvectors: np.ndarray, 
                num_components: int) -> np.ndarray:
    """
    Project data onto principal components.
    
    This function performs dimensionality reduction by projecting the standardized
    data onto the selected principal components (eigenvectors).
    
    Args:
        standardized_data: Standardized data of shape (n_samples, n_features)
        eigenvectors: Sorted eigenvectors of shape (n_features, n_features)
                     where each column is an eigenvector
        num_components: Number of components to retain (must be between 1 and n_features)
        
    Returns:
        Reduced data of shape (n_samples, num_components)
        
    Raises:
        ValueError: If num_components is not between 1 and n_features
        ValueError: If input dimensions are incompatible
        
    Formula:
        X_reduced = X_std @ eigenvectors[:, :num_components]
        
    Example:
        >>> standardized_data = np.array([[0.5, -0.5], [-0.5, 0.5]])
        >>> eigenvectors = np.array([[0.707, 0.707], [-0.707, 0.707]])
        >>> reduced = project_data(standardized_data, eigenvectors, 1)
        >>> reduced.shape
        (2, 1)
    """
    # Validate input shapes
    if standardized_data.ndim != 2:
        raise ValueError(f"standardized_data must be 2D, got {standardized_data.ndim}D")
    
    if eigenvectors.ndim != 2:
        raise ValueError(f"eigenvectors must be 2D, got {eigenvectors.ndim}D")
    
    n_samples, n_features = standardized_data.shape
    
    # Validate eigenvectors shape
    if eigenvectors.shape[0] != n_features:
        raise ValueError(
            f"eigenvectors first dimension ({eigenvectors.shape[0]}) must match "
            f"number of features ({n_features})"
        )
    
    # Validate num_components
    if num_components < 1:
        raise ValueError(f"num_components must be at least 1, got {num_components}")
    
    if num_components > n_features:
        raise ValueError(
            f"num_components ({num_components}) cannot exceed "
            f"number of features ({n_features})"
        )
    
    # Perform projection: X_reduced = X_std @ eigenvectors[:, :num_components]
    reduced_data = standardized_data @ eigenvectors[:, :num_components]
    
    return reduced_data
