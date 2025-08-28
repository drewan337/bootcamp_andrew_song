## Stage 14: Orchestration Plan for TSLA Stock Analysis Project

### **1. Key Tasks in the Project**
The pipeline consists of the following 7 key tasks:
1.  **Fetch Data:** Download historical TSLA stock data from Yahoo Finance.
2.  **Validate Data:** Check the downloaded data for consistency, missing values, and correct data types.
3.  **Calculate Metrics:** Compute key financial metrics (e.g., daily returns, volatility, moving averages).
4.  **Feature Engineering:** Create model-ready features from the cleaned data (e.g., lagged prices, volume changes).
5.  **Train Model:** Train a machine learning model (e.g., Random Forest) to predict future price movements.
6.  **Evaluate Model:** Assess the model's performance on a test set and generate evaluation figures.
7.  **Generate Report:** Compile a final HTML report with key metrics, charts, and model performance results.

### **2. Dependencies Diagram (DAG)**
[Fetch Data] -> [Validate Data] -> [Calculate Metrics] -
| |
-> [Feature Engineering] -> [Train Model] -> [Evaluate Model]
|
-> [Generate Report] <-/
**Parallelizable Tasks:** `Calculate Metrics` and `Feature Engineering` can run in parallel once `Validate Data` is complete.

### **3. Task Specification**

| Task Name | Input(s) | Output(s) | Idempotent? (Y/N) & Why | Logging | Checkpoint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Fetch Data** | Yahoo Finance API Parameters | `data/raw/tsla_raw_data.csv` | **Y** - Will overwrite the output file. Same API call always produces the same data file. | `logs/fetch_data.log` | `data/raw/tsla_raw_data.csv` |
| **2. Validate Data** | `data/raw/tsla_raw_data.csv` | `data/processed/tsla_clean.csv` | **Y** - Validation and cleaning rules are deterministic. Running it again produces the same clean output. | `logs/validate_data.log` | `data/processed/tsla_clean.csv` |
| **3. Calculate Metrics** | `data/processed/tsla_clean.csv` | `data/processed/tsla_metrics.json` | **Y** - Calculations (returns, moving averages) are purely deterministic. | `logs/calculate_metrics.log` | `data/processed/tsla_metrics.json` |
| **4. Feature Engineering** | `data/processed/tsla_clean.csv` | `data/features/tsla_features.csv` | **Y** - Feature creation logic (lags, rolling stats) is fixed and deterministic. | `logs/feature_engineering.log` | `data/features/tsla_features.csv` |
| **5. Train Model** | `data/features/tsla_features.csv` | `models/random_forest_tsla.pkl` | **N** - Training involves randomness (e.g., bootstrapping). Each run produces a slightly different model. | `logs/train_model.log` | `models/random_forest_tsla.pkl` |
| **6. Evaluate Model** | `models/random_forest_tsla.pkl`, `data/features/tsla_features.csv` | `reports/figures/confusion_matrix.png`, `reports/model_metrics.txt` | **Y** - Evaluating a fixed model on a fixed test set produces the same results every time. | `logs/evaluate_model.log` | `reports/model_metrics.txt` |
| **7. Generate Report** | `data/processed/tsla_metrics.json`, `reports/model_metrics.txt` | `reports/final_analysis_report.html` | **Y** - Compiling a report from static input files is a repeatable process. | `logs/generate_report.log` | `reports/final_analysis_report.html` |

### **4. Failure Points & Retry Policy**

*   **Fetch Data:** **Most likely to fail.** Failure points: network timeout, Yahoo Finance API changes/limits. **Retry Policy:** Implement a retry decorator (e.g., `@retry(tries=3, delay=2)`) with exponential backoff. If all retries fail, the pipeline should halt.
*   **Validate Data:** Could fail if raw data is malformed or completely unexpected. **Retry Policy:** No retry. This is a data quality issue that requires manual intervention. The error should be logged clearly.
*   **Train Model:** Could fail on a machine with insufficient memory. **Retry Policy:** No retry. This is a hardware/resource issue that must be solved manually.
*   **General Policy:** For other steps, a single retry is sufficient for transient errors (e.g., a momentary file read/write lock). The pipeline should be designed to fail clearly and log the error for any other issue.

### **5. Automation Strategy: Now vs. Later**

*   **Automate Now (Fully Scripted):**
    *   **Tasks 1-4 and 7 (Fetch, Validate, Calculate Metrics, Feature Engineering, Generate Report).**
    *   **Rationale:** These tasks are **idempotent**, deterministic, and form the core data processing pipeline. They are run frequently (e.g., daily or weekly to get new data) and are stable. Automating them saves significant time, ensures consistency, and eliminates manual error. The final report automation provides a always-up-to-date view of the project.

*   **Keep Manual (For Now):**
    *   **Task 5-6 (Train Model, Evaluate Model).**
    *   **Rationale:** Model training is **not idempotent** and is part of active experimentation. I am still refining hyperparameters, trying different models, and evaluating new features. Automating this now would be premature and could lead to wasted compute resources. The current workflow is to run the data pipeline, then manually execute and tweak the training notebook. Once the modeling approach is finalized, this step can be automated and scheduled to run less frequently (e.g., weekly).