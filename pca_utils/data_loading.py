"""
Data Loading and Validation Module for PCA Assignment

This module provides functions to load and validate datasets for PCA analysis.
It handles CSV file loading with error handling and validates that datasets
meet the assignment requirements.
"""

import pandas as pd
import numpy as np
from typing import Optional


# List of African countries for filtering
AFRICAN_COUNTRIES = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
    'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 'Comoros',
    'Republic of the Congo', 'Democratic Republic of the Congo', 'Djibouti', 
    'Egypt', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 
    'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 
    'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 
    'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 
    'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone',
    'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Eswatini',
    'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
]


def load_dataset(filepath: str, filter_african: bool = True) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        filter_african: If True, filter to only African countries (default: True)
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is empty or cannot be parsed
    """
    try:
        df = pd.read_csv(filepath)
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("Dataset file is empty")
        
        # Filter to African countries if requested
        if filter_african and 'Country' in df.columns:
            df = df[df['Country'].isin(AFRICAN_COUNTRIES)].copy()
            print(f"Filtered to {len(df)} African countries")
            
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Unable to parse dataset file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


def load_dataset_with_regions(filepath: str) -> tuple:
    """
    Load dataset and return both African and non-African countries separately.
    Useful for comparative analysis.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (african_df, non_african_df, full_df)
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is empty or cannot be parsed
    """
    try:
        full_df = pd.read_csv(filepath)
        
        if full_df.empty:
            raise ValueError("Dataset file is empty")
        
        if 'Country' not in full_df.columns:
            raise ValueError("Dataset must have a 'Country' column")
        
        # Split into African and non-African
        african_df = full_df[full_df['Country'].isin(AFRICAN_COUNTRIES)].copy()
        non_african_df = full_df[~full_df['Country'].isin(AFRICAN_COUNTRIES)].copy()
        
        print(f"Loaded {len(african_df)} African countries and {len(non_african_df)} non-African countries")
        
        return african_df, non_african_df, full_df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Unable to parse dataset file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that dataset meets assignment requirements.
    
    Requirements:
    - At least 10 columns
    - Contains missing values (NaN)
    - Contains at least 1 non-numeric column
    - At least 10 rows
    
    Args:
        df: Input DataFrame to validate
        
    Raises:
        ValueError: If any validation requirement is not met
    """
    # Check minimum columns
    if df.shape[1] < 10:
        raise ValueError(
            f"Dataset must have at least 10 columns, found {df.shape[1]}"
        )
    
    # Check for missing values
    if not df.isnull().any().any():
        raise ValueError(
            "Dataset must contain missing values for this assignment"
        )
    
    # Check for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) == 0:
        raise ValueError(
            "Dataset must contain at least 1 non-numeric column"
        )
    
    # Check minimum rows
    if df.shape[0] < 10:
        raise ValueError(
            f"Dataset must have at least 10 rows, found {df.shape[0]}"
        )


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values using specified imputation strategy.
    
    Args:
        df: Input DataFrame with missing values
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
            - 'mean': Replace missing values with column mean (numeric only)
            - 'median': Replace missing values with column median (numeric only)
            - 'mode': Replace missing values with column mode (all types)
            - 'drop': Drop rows with any missing values
        
    Returns:
        DataFrame with missing values handled
        
    Raises:
        ValueError: If strategy is not recognized
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        return df_copy.dropna()
    
    elif strategy == 'mean':
        # Apply mean imputation to numeric columns only
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        return df_copy
    
    elif strategy == 'median':
        # Apply median imputation to numeric columns only
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        return df_copy
    
    elif strategy == 'mode':
        # Apply mode imputation to all columns
        for col in df_copy.columns:
            if df_copy[col].isnull().any():
                mode_value = df_copy[col].mode()
                if len(mode_value) > 0:
                    df_copy[col] = df_copy[col].fillna(mode_value[0])
        return df_copy
    
    else:
        raise ValueError(
            f"Unknown imputation strategy: {strategy}. "
            f"Valid options are: 'mean', 'median', 'mode', 'drop'"
        )


def encode_categorical_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encode categorical columns using label encoding or one-hot encoding.
    
    Uses one-hot encoding for low cardinality features (<=50 unique values)
    and label encoding for high cardinality features (>50 unique values).
    
    Args:
        df: Input DataFrame
        columns: List of categorical column names to encode
        
    Returns:
        DataFrame with encoded categorical features
        
    Raises:
        ValueError: If a specified column does not exist in the DataFrame
    """
    from sklearn.preprocessing import LabelEncoder
    
    df_copy = df.copy()
    
    # Validate that all columns exist
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    for col in columns:
        n_unique = df_copy[col].nunique()
        
        if n_unique > 50:
            # Use label encoding for high cardinality
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        else:
            # Use one-hot encoding for low cardinality
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=False)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop(col, axis=1, inplace=True)
    
    return df_copy


def select_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract numeric features suitable for PCA.
    
    Args:
        df: Input DataFrame
        
    Returns:
        NumPy array of numeric features with shape (n_samples, n_features)
        
    Raises:
        ValueError: If no numeric columns are found
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame")
    
    return numeric_df.values
