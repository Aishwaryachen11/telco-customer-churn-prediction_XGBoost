# Telco Customer Churn Prediction

## Project Overview
This project predicts **customer churn** for a telecom company using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
We compare **Logistic Regression**, **Random Forest**, and **XGBoost** (both tuned and regularized) models, evaluate their performance, and use **SHAP** values for model explainability.

The primary business goal:  
> Identify customers most likely to leave, so targeted retention strategies can be applied.

Use this link to access the notebook in Google Colab:  
[Open Colab Notebook](https://github.com/Aishwaryachen11/telco-customer-churn-prediction_XGBoost/blob/main/Customer_churn_XGBoost%2C_Random_Forest.ipynb)

##  Steps & Workflow
### 1. **Data Loading & Initial Checks**
- Loaded dataset into Pandas.
- Checked data shape, column names, data types, and target distribution.
- Found:
  - 7,043 rows × 21 columns.
  - Target (`Churn`) is imbalanced: ~27% churners, ~73% non-churners.
  - `TotalCharges` had 11 non-numeric entries.

### 2. **Data Cleaning**
- Converted `TotalCharges` to numeric; imputed blanks:
  - If `tenure = 0` → set to 0.
  - Else → impute as `MonthlyCharges × tenure`.
- Dropped `customerID` (no predictive value).
- Converted `Churn` to binary: **Yes → 1, No → 0**.
- Verified: No missing values remained.

### 3. **Feature Engineering**
- Separated **categorical** (15) and **numeric** (4) columns.
- Created preprocessing pipeline:
  - **OneHotEncoder** for categoricals.
  - **StandardScaler** for numerics.

### 4. **Train/Test Split**
- 80/20 split, stratified by churn rate to preserve class distribution.

### 5. **Model Training — Initial Baseline**
Trained three models using the same preprocessing pipeline:
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost** (default settings)

**Results (Test Set)**:
| Model              | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC AUC |
|--------------------|----------|-------------------|----------------|------------|---------|
| Logistic Regression| 0.806    | 0.657              | 0.559          | 0.604      | 0.842   |
| Random Forest      | 0.788    | 0.628              | 0.492          | 0.552      | 0.816   |
| XGBoost (default)  | 0.778    | 0.610              | 0.509          | 0.556      | 0.819   |

### 6. **Tuned XGBoost**
**Why tune:**  
The default XGBoost model underperformed compared to Logistic Regression due to:
- Small dataset size (~7k rows) and many low-cardinality categorical features.
- Class imbalance (~27% churners) pushing predictions toward the majority class.
- No regularization or imbalance handling in the defaults.

**Parameter changes made:**
- `n_estimators=500` → more boosting rounds for finer learning.
- `learning_rate=0.05` → slower learning to capture complex patterns gradually.
- `max_depth=4` → moderate tree depth for balanced bias-variance.
- `subsample=0.8` & `colsample_bytree=0.8` → inject randomness for robustness.
- `reg_lambda=2.0` (L2) and `reg_alpha=0.5` (L1) → prevent overfitting.
- `scale_pos_weight≈2.77` → give more weight to the minority churn class.

**Key improvement (compared to default XGBoost):**
- **Recall (Churn)** jumped from **0.509 → 0.754** (+24.5 pp).
- **F1 (Churn)** improved from **0.556 → 0.629**.
- ROC AUC slightly decreased from 0.819 → 0.836, but the recall gain is more important for churn use cases.

### 7. **Overfitting Check (Tuned XGBoost)**
- **Train Accuracy:** 0.835  
- **Test Accuracy:** 0.764  
- **Train ROC AUC:** 0.929  
- **Test ROC AUC:** 0.836  
- **Gap:** ~0.093 in ROC AUC → indicates mild to moderate overfitting.
Interpretation:
- The model learns patterns in training extremely well (near-perfect separation) but generalizes less effectively to unseen data.
- This gap, while not extreme, suggests the model is too complex for the dataset without stronger regularization.

### 8. **Regularized XGBoost**
To reduce overfitting and improve generalization, we applied stronger regularization and early stopping:

**Parameter changes made:**
- `max_depth=3` → shallower trees to reduce complexity.
- `min_child_weight=5` → require more samples per leaf node.
- `subsample=0.7` & `colsample_bytree=0.7` → increase randomness in training.
- `reg_lambda=5.0` (L2) and `reg_alpha=1.0` (L1) → stronger regularization.
- `gamma=0.5` → require minimum loss reduction for splits.
- `n_estimators=1000` with `learning_rate=0.03` → more trees but slower learning.
- `early_stopping_rounds=30` → stop when validation performance plateaus.

**Results (Test Set):**
- Accuracy: 0.739  
- Precision (Churn): 0.505  
- **Recall (Churn): 0.791** — the highest among all models.  
- F1 (Churn): 0.617  
- ROC AUC: **0.847** — slightly higher than Logistic Regression (0.842).

**Overfitting gap:**
- Train ROC AUC: 0.868  
- Test ROC AUC: 0.847  
- Gap: **0.021** → a significant improvement over the tuned model’s 0.093 gap.
Interpretation:
- The regularized model is far less overfit and maintains excellent recall, making it ideal for business contexts where catching as many churners as possible is the priority.

### 10. **Explainability with SHAP**
- Used **SHAP TreeExplainer** for feature importance.
- Aggregated one-hot encoded features into their original groups for a business-friendly view.

<img src="https://github.com/Aishwaryachen11/telco-customer-churn-prediction_XGBoost/blob/main/SHAP-Barplot.png" width="550"/>
<img src="https://github.com/Aishwaryachen11/telco-customer-churn-prediction_XGBoost/blob/main/SHAP-Barplot2.png" width="550"/>

**Top Factors Influencing Churn**:
1. **Contract type** — Month-to-month strongly increases churn risk.
2. **Tenure** — Lower tenure → higher churn.
3. **MonthlyCharges** — Higher charges increase churn risk.
4. **OnlineSecurity** — Lack of it correlates with churn.
5. **InternetService type** — Fiber optic customers churn more in this dataset.
6. **TechSupport** — Lack of support increases churn.


## Key Insights
- Short-term contracts, high monthly charges, and lack of service add-ons (security/support) are major churn drivers.
- Payment by electronic check is associated with higher churn.
- Retention campaigns should target **new, month-to-month, high-cost, low-service customers**.

## Tech Stack
- **Python**: pandas, numpy, scikit-learn, xgboost, shap, matplotlib
- **Machine Learning**: Logistic Regression, Random Forest, XGBoost
- **Explainability**: SHAP (TreeExplainer)

