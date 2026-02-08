"""
Covariance Matrix Module

Functions for calculating the covariance matrix of standardized data.
"""

import numpy as np


def calculate_covariance_matrix(standardized_data: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix from standardized data.
    
    The covariance matrix represents the pairwise covariances between features.
    It is a symmetric matrix where element (i,j) represents the covariance 
    between feature i and feature j. For standardized data, this is equivalent 
    to the correlation matrix.
    
    Args:
        standardized_data: Standardized data array of shape (n_samples, n_features)
        
    Returns:
        Covariance matrix of shape (n_features, n_features)
        
    Formula:
        Cov = (1 / (n-1)) * X^T @ X
        where X is the standardized data (n_samples x n_features)
        
    Note:
        We use np.cov with rowvar=False, which treats each column as a variable
        and each row as an observation. This is the standard convention for 
        datasets where rows are samples and columns are features.
    """
    # Use NumPy's covariance function
    # rowvar=False means columns are variables (features), rows are observations (samples)
    cov_matrix = np.cov(standardized_data, rowvar=False)
    
    return cov_matrix
