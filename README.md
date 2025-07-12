# Insurance ML Model Comparison

This repository presents a comparative study of machine learning models applied to both classificatioM and regression tasks using a structured insurance dataset. The project emphasizes reproducible ML pipelines with strong feature engineering, dimensionality reduction and hyperparameter tuning using GridSearchCV.


## Project Structure

| File | Task | Description |
|------|------|-------------|
| `Insurance_Classification_ModelComparison.ipynb` | **Start Here** | Includes full Exploratory Data Analysis (EDA), feature preprocessing, encoding, and classification pipelines with grid search tuning for all steps. |
| `Insurance_Regression_ModelComparison.ipynb` | Regression | Builds on the prepared dataset and applies similar pipeline optimization to predict continuous targets (insurance charges). |


## Key Highlights

-  **EDA & Data Preparation**: Performed in the classification notebook with categorical encoding, missing value checks, and statistical summaries.
-  **Pipeline Architecture**: Built using `Pipeline` from `scikit-learn`, integrating scalers, feature selectors (RFE, SelectKBest, PCA), and models.
-  **Model Comparison**:
  - **Classification Models**: SVM, RandomForest, kNN
  - **Regression Models**: SVR, RandomForestRegressor, Linear Regression
-  **Hyperparameter Tuning**:
  - Performed via `GridSearchCV` over full pipelines
  - Parameters optimized include: 
    - Dimensionality reduction components (PCA, RFE, SelectKBest)
    - Model-specific hyperparameters
-  **Model Evaluation**:
  - Classification: Accuracy, Precision, Recall, F1-score, ROC curves
  - Regression: MAE, MAE, R²

## 🗃 Dataset

- **`insurance.csv`**
  - Features: age, sex, BMI, children, smoker, region
  - Target:
    - Classification: binary labels (Mmedian split)
    - Regression: insurance charges


## Tools

- Python, pandas, NumPy
- scikit-learn
- matplotlib, seaborn
