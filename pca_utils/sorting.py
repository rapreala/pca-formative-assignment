"""
Principal Component Sorting Module

Functions for sorting principal components by eigenvalues in descending order.
"""

import numpy as np


def sort_principal_components(eigenvalues: np.ndarray, 
                              eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort eigenvalues and eigenvectors in descending order.
    
    Principal components should be ordered by the amount of variance they explain,
    which corresponds to sorting by eigenvalues in descending order. This function
    sorts both eigenvalues and their corresponding eigenvectors while preserving
    the eigenvalue-eigenvector correspondence.
    
    Args:
        eigenvalues: 1D array of eigenvalues of shape (n_features,)
        eigenvectors: 2D array of shape (n_features, n_features) where each 
                     column is an eigenvector corresponding to the eigenvalue 
                     at the same index
        
    Returns:
        Tuple of (sorted_eigenvalues, sorted_eigenvectors)
        - sorted_eigenvalues: 1D array sorted in descending order
        - sorted_eigenvectors: 2D array with columns reordered to match sorted eigenvalues
    
    Raises:
        ValueError: If eigenvalues is not 1D
        ValueError: If eigenvectors is not 2D
        ValueError: If the number of eigenvalues doesn't match the number of eigenvectors
    
    Example:
        >>> eigenvalues = np.array([1.5, 3.2, 0.8])
        >>> eigenvectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> sorted_vals, sorted_vecs = sort_principal_components(eigenvalues, eigenvectors)
        >>> sorted_vals
        array([3.2, 1.5, 0.8])
    """
    # Validate inputs
    if eigenvalues.ndim != 1:
        raise ValueError(f"Eigenvalues must be 1D array, got {eigenvalues.ndim}D")
    
    if eigenvectors.ndim != 2:
        raise ValueError(f"Eigenvectors must be 2D array, got {eigenvectors.ndim}D")
    
    n_eigenvalues = eigenvalues.shape[0]
    n_eigenvectors = eigenvectors.shape[1]
    
    if n_eigenvalues != n_eigenvectors:
        raise ValueError(
            f"Number of eigenvalues ({n_eigenvalues}) must match number of "
            f"eigenvectors ({n_eigenvectors})"
        )
    
    # Get indices that would sort eigenvalues in descending order
    # np.argsort sorts in ascending order, so we reverse with [::-1]
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # Sort eigenvalues using the indices
    sorted_eigenvalues = eigenvalues[sorted_indices]
    
    # Sort eigenvectors by reordering columns using the same indices
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvectors
