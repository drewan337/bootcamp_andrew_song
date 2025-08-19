## Data Cleaning Strategy

### Cleaning Pipeline
1. **Missing Value Handling**:
   - Drop columns with >50% missing values
   - Fill numeric missing values with median (robust to outliers)
   - Preserve categorical missing values for specific handling

2. **Normalization**:
   - StandardScaler applied to numeric features (mean=0, std=1)
   - Prepares data for machine learning algorithms
   - Maintains data distribution while scaling

3. **Data Validation**:
   - Shape consistency checks
   - Missing value verification
   - Distribution comparison pre/post cleaning

### Key Functions
- `fill_missing_median()`: Robust missing value imputation
- `drop_missing()`: Removes high-missing columns
- `normalize_data()`: Standardizes numeric features
- `basic_clean()`: Complete cleaning pipeline

### Assumptions
- Numeric data benefits from median imputation
- Columns with >50% missing are not recoverable
- Normalization improves ML model performance

### File Structure
- Functions in src/cleaning.py
- Raw dataset in /data/raw
- Saved cleaned dataset in /data/processed/