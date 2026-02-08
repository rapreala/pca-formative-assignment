"""
Visualization module for PCA implementation.

This module provides functions to visualize original and PCA-reduced data.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_original_data(data: np.ndarray, feature_names: list = None):
    """
    Plot scatter plot of first two features of original data.
    
    Args:
        data: Original data array of shape (n_samples, n_features)
        feature_names: Optional list of feature names for axis labels
        
    Raises:
        ValueError: If data has fewer than 2 features or is empty
    """
    # Validate input
    if data.shape[0] == 0:
        raise ValueError("Cannot plot empty dataset")
    
    if data.shape[1] < 2:
        raise ValueError(f"Need at least 2 dimensions for plotting, found {data.shape[1]}")
    
    # Handle infinite or NaN values
    if np.any(~np.isfinite(data[:, :2])):
        print("Warning: Data contains infinite or NaN values, filtering for plot")
        mask = np.all(np.isfinite(data[:, :2]), axis=1)
        data = data[mask]
    
    # Create scatter plot of first 2 features
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add labels and title
    if feature_names and len(feature_names) >= 2:
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
    else:
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
    
    plt.title("Original Data (First 2 Features)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_reduced_data(reduced_data: np.ndarray, explained_variance: np.ndarray):
    """
    Plot scatter plot of PC1 vs PC2.
    
    Args:
        reduced_data: Reduced data array with at least 2 components, shape (n_samples, n_components)
        explained_variance: Explained variance percentages for labeling
        
    Raises:
        ValueError: If reduced_data has fewer than 2 components or is empty
    """
    # Validate input
    if reduced_data.shape[0] == 0:
        raise ValueError("Cannot plot empty dataset")
    
    if reduced_data.shape[1] < 2:
        raise ValueError(f"Need at least 2 dimensions for plotting, found {reduced_data.shape[1]}")
    
    # Handle infinite or NaN values
    if np.any(~np.isfinite(reduced_data[:, :2])):
        print("Warning: Data contains infinite or NaN values, filtering for plot")
        mask = np.all(np.isfinite(reduced_data[:, :2]), axis=1)
        reduced_data = reduced_data[mask]
    
    # Create scatter plot of PC1 vs PC2
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add labels with variance percentages
    pc1_var = explained_variance[0] if len(explained_variance) > 0 else 0
    pc2_var = explained_variance[1] if len(explained_variance) > 1 else 0
    
    plt.xlabel(f"PC1 ({pc1_var:.2f}% variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pc2_var:.2f}% variance)", fontsize=12)
    plt.title("PCA Reduced Data (PC1 vs PC2)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def create_comparison_plots(original_data: np.ndarray, 
                           reduced_data: np.ndarray,
                           explained_variance: np.ndarray,
                           feature_names: list = None):
    """
    Create side-by-side comparison of original and reduced data.
    
    Args:
        original_data: Original data array of shape (n_samples, n_features)
        reduced_data: PCA-reduced data array of shape (n_samples, n_components)
        explained_variance: Explained variance percentages for labeling
        feature_names: Optional list of feature names for original data axis labels
        
    Raises:
        ValueError: If either dataset has fewer than 2 dimensions or is empty
    """
    # Validate inputs
    if original_data.shape[0] == 0 or reduced_data.shape[0] == 0:
        raise ValueError("Cannot plot empty dataset")
    
    if original_data.shape[1] < 2:
        raise ValueError(f"Original data needs at least 2 dimensions for plotting, found {original_data.shape[1]}")
    
    if reduced_data.shape[1] < 2:
        raise ValueError(f"Reduced data needs at least 2 dimensions for plotting, found {reduced_data.shape[1]}")
    
    # Handle infinite or NaN values in original data
    original_plot_data = original_data.copy()
    if np.any(~np.isfinite(original_plot_data[:, :2])):
        print("Warning: Original data contains infinite or NaN values, filtering for plot")
        mask = np.all(np.isfinite(original_plot_data[:, :2]), axis=1)
        original_plot_data = original_plot_data[mask]
    
    # Handle infinite or NaN values in reduced data
    reduced_plot_data = reduced_data.copy()
    if np.any(~np.isfinite(reduced_plot_data[:, :2])):
        print("Warning: Reduced data contains infinite or NaN values, filtering for plot")
        mask = np.all(np.isfinite(reduced_plot_data[:, :2]), axis=1)
        reduced_plot_data = reduced_plot_data[mask]
    
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot original data (first 2 features)
    axes[0].scatter(original_plot_data[:, 0], original_plot_data[:, 1], 
                   alpha=0.6, edgecolors='k', linewidth=0.5)
    
    if feature_names and len(feature_names) >= 2:
        axes[0].set_xlabel(feature_names[0], fontsize=12)
        axes[0].set_ylabel(feature_names[1], fontsize=12)
    else:
        axes[0].set_xlabel("Feature 1", fontsize=12)
        axes[0].set_ylabel("Feature 2", fontsize=12)
    
    axes[0].set_title("Original Data (First 2 Features)", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot reduced data (PC1 vs PC2)
    axes[1].scatter(reduced_plot_data[:, 0], reduced_plot_data[:, 1], 
                   alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    
    pc1_var = explained_variance[0] if len(explained_variance) > 0 else 0
    pc2_var = explained_variance[1] if len(explained_variance) > 1 else 0
    
    axes[1].set_xlabel(f"PC1 ({pc1_var:.2f}% variance)", fontsize=12)
    axes[1].set_ylabel(f"PC2 ({pc2_var:.2f}% variance)", fontsize=12)
    axes[1].set_title("PCA Reduced Data (PC1 vs PC2)", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()


def create_regional_comparison_plot(african_reduced: np.ndarray,
                                   non_african_reduced: np.ndarray,
                                   explained_variance: np.ndarray):
    """
    Create comparison plot showing African vs non-African countries in PCA space.
    
    Args:
        african_reduced: PCA-reduced data for African countries
        non_african_reduced: PCA-reduced data for non-African countries
        explained_variance: Explained variance percentages
    """
    plt.figure(figsize=(12, 8))
    
    # Plot non-African countries (background)
    plt.scatter(non_african_reduced[:, 0], non_african_reduced[:, 1], 
                alpha=0.3, s=40, color='gray', label='Rest of World')
    
    # Plot African countries (foreground)
    plt.scatter(african_reduced[:, 0], african_reduced[:, 1], 
                alpha=0.7, s=80, color='#E74C3C', edgecolors='black', 
                linewidths=0.5, label='African Countries')
    
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}% variance)', fontsize=13, fontweight='bold')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}% variance)', fontsize=13, fontweight='bold')
    plt.title('African Countries vs Rest of World in PCA Space', 
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print statistical comparison
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: AFRICA VS REST OF WORLD")
    print("="*60)
    
    # PC1 comparison
    african_pc1_mean = np.mean(african_reduced[:, 0])
    world_pc1_mean = np.mean(non_african_reduced[:, 0])
    african_pc1_std = np.std(african_reduced[:, 0])
    world_pc1_std = np.std(non_african_reduced[:, 0])
    
    print(f"\nPC1 (Primary Development Axis - {explained_variance[0]:.2f}% variance):")
    print(f"  African Countries:    Mean = {african_pc1_mean:>7.2f}, Std = {african_pc1_std:.2f}")
    print(f"  Rest of World:        Mean = {world_pc1_mean:>7.2f}, Std = {world_pc1_std:.2f}")
    print(f"  Difference:           {african_pc1_mean - world_pc1_mean:>7.2f}")
    
    # PC2 comparison
    african_pc2_mean = np.mean(african_reduced[:, 1])
    world_pc2_mean = np.mean(non_african_reduced[:, 1])
    african_pc2_std = np.std(african_reduced[:, 1])
    world_pc2_std = np.std(non_african_reduced[:, 1])
    
    print(f"\nPC2 (Secondary Development Axis - {explained_variance[1]:.2f}% variance):")
    print(f"  African Countries:    Mean = {african_pc2_mean:>7.2f}, Std = {african_pc2_std:.2f}")
    print(f"  Rest of World:        Mean = {world_pc2_mean:>7.2f}, Std = {world_pc2_std:.2f}")
    print(f"  Difference:           {african_pc2_mean - world_pc2_mean:>7.2f}")
    
    # Variance comparison
    print(f"\nVariance (Spread) Comparison:")
    print(f"  African PC1 variance: {african_pc1_std**2:.2f}")
    print(f"  World PC1 variance:   {world_pc1_std**2:.2f}")
    print(f"  African PC2 variance: {african_pc2_std**2:.2f}")
    print(f"  World PC2 variance:   {world_pc2_std**2:.2f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    
    if african_pc1_mean < world_pc1_mean:
        print(f"• African countries tend to score LOWER on PC1 (by {abs(african_pc1_mean - world_pc1_mean):.2f} units)")
        print("  This suggests different development patterns compared to global average")
    else:
        print(f"• African countries tend to score HIGHER on PC1 (by {abs(african_pc1_mean - world_pc1_mean):.2f} units)")
        print("  This suggests advanced development in PC1 dimensions")
    
    if african_pc1_std > world_pc1_std:
        print(f"• African countries show GREATER diversity in PC1 ({african_pc1_std:.2f} vs {world_pc1_std:.2f})")
        print("  Indicating more varied development levels within Africa")
    else:
        print(f"• African countries show LESS diversity in PC1 ({african_pc1_std:.2f} vs {world_pc1_std:.2f})")
        print("  Indicating more uniform development patterns within Africa")
    
    print("\n" + "="*60 + "\n")
