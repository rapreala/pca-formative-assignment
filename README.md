# PCA Assignment: Principal Component Analysis on Global Country Data

## 📋 Project Overview

This project implements **Principal Component Analysis (PCA) from scratch** using NumPy for my Math of Machine Learning class. I analyzed socio-economic and tourism indicators across 51 African countries to demonstrate dimensionality reduction while preserving variance.

**Key Features:**
- ✅ PCA implemented from scratch (no sklearn.decomposition.PCA)
- ✅ Analysis of global country data (tourism, economy, demographics)
- ✅ Complete 8-step implementation in Jupyter notebook
- ✅ Visualization of data before and after PCA

---

## 🗂️ Dataset

**Dataset Name:** Global Country Information Dataset 2023  
**Source:** Kaggle - [Countries of the World 2023](https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023)

**Dataset Characteristics:**
- **52 African countries** (filtered from global dataset of 195 countries)
- **35 columns** including:
  - Country name (categorical)
  - Population, Density, Land Area
  - GDP, GDP per capita
  - Education enrollment rates (Primary, Secondary, Tertiary)
  - CO2 emissions
  - Healthcare metrics (Life expectancy, Infant mortality)
  - Economic indicators (Unemployment rate, Tax revenue)
  - Tourism-related factors (GDP contribution, infrastructure)
- **Contains missing values** (NaN in various columns)
- **Contains non-numeric data** (country names)

**Why I Chose This Dataset:**
I selected this dataset and filtered it to African countries because:
1. It focuses on African nations, which is relevant to the hospitality and tourism context
2. With 52 African countries, it provides sufficient data points for meaningful PCA analysis
3. It has multiple correlated features (economic indicators often correlate)
4. Features have different scales (population in millions, GDP in billions, rates in percentages)
5. It demonstrates how PCA can reveal patterns in African country development and tourism potential

---

## 🚀 Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download This Repository
```bash
# If using git
git clone <repository-url>
cd pca-assignment

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - For mathematical computations
- `pandas` - For data loading and preprocessing
- `matplotlib` & `seaborn` - For visualization
- `jupyter` - For running the notebook
- `scikit-learn` - For preprocessing utilities only (NOT for PCA)

### Step 3: Download the Dataset

The dataset is already included in the `data/` folder. If you need to re-download:

**Option A: Using Kaggle API**
```bash
kaggle datasets download -d nelgiriyewithana/countries-of-the-world-2023
unzip countries-of-the-world-2023.zip -d data/
```

**Option B: Manual Download**
1. Visit [the dataset page](https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023)
2. Click "Download" button
3. Extract the CSV file to the `data/` directory

---

## 📓 Running the Notebook

### Start Jupyter Notebook
```bash
jupyter notebook
```

This will open your browser. Navigate to and open:
- `PCA_Formative_1_RyanKWApreala.ipynb`

### Run All Cells
1. In Jupyter, click **Cell → Run All** to execute all steps
2. Or run cells individually by pressing **Shift + Enter**

### Expected Outputs
Each cell will display:
- Data loading confirmation and sample rows
- Standardized data statistics
- Covariance matrix
- Eigenvalues and eigenvectors
- Explained variance percentages
- Reduced data shape and samples
- Before/After PCA visualizations

---

## 🔧 Data Preprocessing Steps

Before applying PCA, I preprocessed the data through several steps:

### **1. Missing Value Imputation**
**Strategy Used:** Mean imputation for numeric columns
- Missing values in numeric columns are replaced with the column mean
- This preserves the overall distribution while handling NaN values
- I considered alternatives like median (for skewed data) and mode (for categorical), but mean imputation works well for this dataset

### **2. Categorical Encoding**
**Approach Used:** Label encoding for country names
- Country names (categorical) are converted to numeric labels (0, 1, 2, ...)
- Since there are 195 unique countries (high cardinality), label encoding is more efficient than one-hot encoding
- This allows the country column to be included in the numeric feature matrix

### **3. Data Cleaning Decisions**
- **Feature Selection:** All numeric columns are retained after encoding
- **Outlier Handling:** No outlier removal (PCA is sensitive to outliers, which helps identify interesting patterns)
- **Scaling:** Handled by standardization in Step 1 (not during preprocessing)
- **Missing Data:** All missing values are imputed before PCA to ensure complete data matrix

---

## 📊 PCA Implementation Steps

I implemented PCA in 8 steps following the classical algorithm:

### **Step 0: Data Loading & Preprocessing**
- Load the Global Country Information dataset
- Filter to African countries only (52 countries)
- Handle missing values using mean imputation
- Encode categorical features (country names)
- Select numeric features for PCA

### **Step 1: Data Standardization**
- Calculate mean and standard deviation for each feature
- Apply formula: `(X - μ) / σ`
- Ensures all features have mean=0 and std=1
- **Why?** Features with larger scales would dominate PCA without standardization

### **Step 3: Covariance Matrix Calculation**
- Compute covariance between all feature pairs
- Results in symmetric matrix showing feature relationships
- **Formula:** `Cov = (1 / (n-1)) * X^T @ X`

### **Step 4: Eigendecomposition**
- Extract eigenvalues and eigenvectors from covariance matrix
- Eigenvalues = variance in each direction
- Eigenvectors = directions of maximum variance
- **Key insight:** Eigenvectors are orthogonal (perpendicular) to each other

### **Step 5: Sort Principal Components**
- Sort eigenvalues in descending order
- Reorder eigenvectors to match
- Calculate explained variance percentages
- **Purpose:** Identify which components capture the most information

### **Step 6: Project Data onto Principal Components**
- Select number of components (based on variance threshold, e.g., 85-90%)
- Project standardized data onto principal components
- Reduce dimensionality while preserving information
- **Formula:** `X_reduced = X_std @ eigenvectors[:, :k]`

### **Step 7: Output Reduced Data**
- Display shape of reduced data
- Show sample rows
- Compare original vs reduced dimensions
- **Result:** Fewer dimensions with most variance preserved

### **Step 8: Visualize Before and After PCA**
- Plot original data (first 2 features)
- Plot reduced data (PC1 vs PC2)
- Compare to see how PCA rotates and compresses data
- **Observation:** PC1 shows highest variance, PC2 shows second highest

### **Step 9: Comparative Analysis (Bonus)**
- Compare African countries to the rest of the world in PCA space
- Visualize how Africa positions relative to global patterns
- Statistical comparison of PC1 and PC2 distributions
- **Insight:** Reveals unique characteristics of African development patterns



---

## 🧮 Mathematical Concepts Explained

### What is PCA?
**Real-life analogy:** Imagine you're taking a photo of a 3D object. You want to capture it from the best angle that shows the most detail. PCA finds that "best angle" for your data.

### Key Formulas

**1. Standardization:**
```
X_std = (X - mean(X)) / std(X)
```

**2. Covariance Matrix:**
```
Cov = (1 / (n-1)) * X^T @ X
```

**3. Eigendecomposition:**
```
Cov @ v = λ * v
```
Where:
- `v` = eigenvector (principal component direction)
- `λ` = eigenvalue (variance in that direction)

**4. Explained Variance:**
```
explained_variance[i] = (λ[i] / sum(λ)) * 100
```

**5. Data Projection:**
```
X_reduced = X_std @ eigenvectors[:, :k]
```

---

## 📁 Project Structure

```
pca-assignment/
├── PCA_Formative_1_RyanKWApreala.ipynb  # Main submission notebook
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── data/
│   └── world-data-2023.csv              # Dataset (filtered to 51 African countries)
└── pca_utils/                           # Python library for PCA
    ├── __init__.py
    ├── data_loading.py                  # Data loading and preprocessing
    ├── standardization.py               # Data standardization
    ├── covariance.py                    # Covariance matrix calculation
    ├── eigendecomposition.py            # Eigenvalue/eigenvector computation
    ├── sorting.py                       # Principal component sorting
    ├── explained_variance.py            # Variance calculations
    ├── projection.py                    # Data projection
    └── visualization.py                 # Plotting functions
```

---

## 🎯 Assignment Requirements Met

✅ **Data Requirements:**
- African-focused dataset (51 African countries)
- 35 columns (exceeds 10 minimum requirement)
- Contains missing values (various columns have NaN)
- Contains non-numeric data (country names)

---

## 🧪 Testing (Optional)

If you want to run the test suite:

```bash
pytest tests/
```

This runs property-based tests to verify mathematical correctness of the PCA implementation.

---

## 📚 Dependencies & Versions

All dependencies are listed in `requirements.txt` and can be installed with:
```bash
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.21.0 | Mathematical computations (matrix operations, eigendecomposition) |
| **pandas** | ≥1.3.0 | Data loading, preprocessing, and manipulation |
| **matplotlib** | ≥3.5.0 | Plotting and visualization |
| **seaborn** | ≥0.11.0 | Enhanced statistical visualizations |
| **jupyter** | ≥1.0.0 | Notebook environment |
| **notebook** | ≥6.4.0 | Jupyter notebook server |

### Optional Dependencies (for testing)

| Package | Version | Purpose |
|---------|---------|---------|
| **hypothesis** | ≥6.50.0 | Property-based testing framework |
| **pytest** | ≥7.0.0 | Unit testing framework |

### Additional Tools

| Package | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | ≥1.0.0 | Used only for preprocessing (LabelEncoder), NOT for PCA |
| **kaggle** | ≥1.5.12 | Dataset download helper (optional) |

**Note:** While scikit-learn is installed, `sklearn.decomposition.PCA` is **NOT used** in this implementation. All PCA calculations are done from scratch using NumPy.

---

## �‍🎓 Author

**Student:** Ryan Apreala  
**Course:** Math of Machine Learning (MMLJ2026)  
**Assignment:** Formative 1 - Principal Component Analysis  
**Date:** February 2026

---

## 📖 References

1. **Dataset Source:** Nidula Elgiriyewithana. (2023). *Global Country Information Dataset 2023*. Kaggle. https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023

2. **PCA Theory:** Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.

3. **Explained Variance Video:** [How Is Explained Variance Used In PCA?](https://www.youtube.com/watch?v=vaF-1xUEXsA&t=17s)

---

## 📝 License

This project is for educational purposes as part of a university assignment.

---

## 🤝 Acknowledgments

- Kaggle for providing the dataset
- Course instructors for the assignment template
- The open-source Python community for the amazing libraries
