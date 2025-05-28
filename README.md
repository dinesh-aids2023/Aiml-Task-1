# Data Cleaning & Preprocessing for Machine Learning

This project demonstrates how to clean and preprocess raw data to make it suitable for machine learning models. The workflow includes importing data, handling missing values, encoding categorical variables, scaling numerical features, and visualizing/removing outliers.

## 🧰 Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (for scaling)

---

## 🧭 Workflow Overview

### 1. Importing the Dataset
- Load data using `pandas.read_csv()`.
- Display initial rows and structure using `.head()` and `.info()`.

### 2. Handling Missing Values
- For numerical columns: fill with **mean**.
- For categorical columns: fill with **mode**.

### 3. Encoding Categorical Variables
- Apply **one-hot encoding** using `pd.get_dummies()`.

### 4. Scaling Features
- Normalize or standardize numerical columns using:
  - `StandardScaler` (standardization)
  - `MinMaxScaler` (normalization)

### 5. Outlier Detection and Removal
- Visualize using **boxplots**.
- Remove outliers using **Z-score** (threshold > 3).



