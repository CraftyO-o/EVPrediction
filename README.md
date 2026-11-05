# Predicting Electric Vehicle Purchase Intention
### A Stacked Gradient Boosting Ensemble with Adaptive Feature Engineering

A machine learning project developed for the **Kaggle Playground Series – Season 6, Episode 9**, focused on predicting consumer electric vehicle (EV) purchase intention using a heterogeneous gradient boosting ensemble.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/LightGBM-4.3-green" />
  <img src="https://img.shields.io/badge-XGBoost-2.0-orange" />
  <img src="https://img.shields.io/badge/CatBoost-1.2-yellow" />
  <img src="https://img.shields.io/badge/Optuna-3.6-blueviolet" />
</p>

## Overview

Predicting electric vehicle adoption is a challenging binary classification problem due to the complex interactions between socioeconomic factors, environmental attitudes, financial incentives, and charging infrastructure.

This project implements an end-to-end machine learning pipeline combining adaptive feature engineering, hyperparameter optimization, and heterogeneous gradient boosting models.

The final solution uses **LightGBM, XGBoost, and CatBoost**, independently optimized using Optuna and combined through rank-based ensemble averaging.

### Results

| Metric | Score |
|---|---:|
| Best Individual Model (LightGBM) | 0.94151 |
| Final Ensemble – Public AUC | 0.94519 |
| Final Ensemble – Private AUC | **0.94541** |
| Competition Placement | Approximately Top 15% |

The final ensemble outperformed all individual model configurations evaluated in the project.

## Dataset

The dataset was provided through Kaggle's Playground Series S6E9 competition.

| Dataset | Samples | Features |
|---|---:|---:|
| Training | 668,665 | 13 + target |
| Testing | 286,571 | 13 |

**Target variable:** `Will_Buy_EV`

The task is to estimate the probability that a consumer intends to purchase an electric vehicle.

The dataset contains demographic, financial, and behavioural features, including:

- Age and Gender
- Annual Income
- Daily Commute Distance
- Number of Cars Owned
- Charging Stations Near Home and Work
- Environmental Concern Level
- Home Charging Availability
- Government Subsidy Availability
- Range Anxiety Level
- City Type and Current Car Type

The positive class represents approximately 17.5% of the training data, making this an imbalanced binary classification problem.

## Methodology

The proposed pipeline consists of four major feature engineering stages, followed by model training and rank-based ensembling.

### 1. Feature Engineering

**Base Numeric and Ordinal Encoding**

- Converted numerical features to `float32`.
- Applied ordinal encoding to `Range_Anxiety_Level`.
- Applied binary encoding to subsidy availability and home charging availability.
- Normalized annual income to units of $100,000.
- Constructed infrastructure-related features:
  - `total_charging`
  - `commute_per_charge`

**Multiplicative Interaction Features**

Constructed pairwise interaction features to capture relationships between financial and environmental factors:

- `income_x_concern`
- `income_x_subsidy`
- `concern_x_subsidy`

The most predictive engineered feature was:

`concern_x_subsidy = Environmental_Concern_Level × Subsidy_Available`

It achieved a univariate AUC of **0.915**, highlighting the importance of the interaction between environmental awareness and financial incentives.

**Frequency and Digit Pattern Features**

- Added frequency encoding for selected numerical features.
- Extracted modulo-10 and modulo-100 digit patterns from five numerical columns.
- Used these features to capture value repetition and patterns present in the synthetic dataset.

**Adaptive Exact-Value Target Encoding**

Implemented smoothed target encoding with independently selected smoothing parameters for each feature.

The smoothing parameter was selected through inner 4-fold cross-validation from:

`S = {0.5, 1, 2, 5, 10, 20, 50, 100}`

Out-of-fold encoding was used to prevent target leakage during training.

The resulting feature matrix contained 58 features after feature construction and categorical encoding.

### 2. Model Training

Three gradient boosting implementations were used as base learners.

| Model | Implementation |
|---|---|
| LightGBM | Histogram-based gradient boosting |
| XGBoost | Histogram tree method with second-order optimization |
| CatBoost | Ordered boosting with symmetric trees |

Each model was independently optimized using **Optuna with Tree-structured Parzen Estimator (TPE) sampling**.

- 100 Optuna trials per model family.
- Stratified 3-fold outer cross-validation.
- Fixed random seed: 42.
- Early stopping where applicable.

### 3. Ensemble Strategy

Rather than directly averaging predicted probabilities, the final ensemble uses rank-based averaging.

For each model:

1. Generate out-of-fold predictions on the training data.
2. Generate predictions on the test data.
3. Average predictions across outer folds.
4. Convert predictions into normalized ranks.
5. Average the ranks across constituent models.

The final ensemble is:

`Final Prediction = Rank Average(LightGBM, XGBoost, CatBoost)`

This approach reduces sensitivity to differences in probability calibration between individual model families.

## Experimental Results

### Model Comparison

| Model | Public AUC | Private AUC |
|---|---:|---:|
| Logistic Regression | 0.93715 | 0.93754 |
| Random Forest | 0.93840 | 0.93880 |
| Artificial Neural Network | 0.93733 | 0.93764 |
| XGBoost | 0.94060 | 0.94104 |
| CatBoost | 0.94008 | 0.94069 |
| LightGBM | 0.94098 | 0.94151 |
| XGBoost + CatBoost + RF | 0.94031 | 0.94079 |
| XGBoost + CatBoost + LightGBM | 0.94070 | 0.94123 |
| Optuna-Tuned LightGBM Ensemble | 0.94091 | 0.94161 |
| **Final Rank-Blend Ensemble** | **0.94519** | **0.94541** |

### Key Findings

- Multiplicative feature interactions provided the largest individual feature engineering improvement.
- `concern_x_subsidy` was the strongest individual predictive feature.
- Adaptive per-column target encoding outperformed fixed-smoothing target encoding.
- Heterogeneous gradient boosting models captured complementary patterns.
- Rank-based blending improved the final ensemble over individual model predictions.

## Project Architecture

The overall workflow follows:

```text
             Raw Dataset
                  |
                  v
          Data Preprocessing
                  |
                  v
          Feature Engineering
          /       |        \
     Interaction Frequency  Adaptive
       Features   & Digit   Target Encoding
                  Features
                  |
                  v
           Feature Matrix
                  |
          Stratified 3-Fold CV
                  |
        +---------+---------+
        |         |         |
        v         v         v
     LightGBM  XGBoost  CatBoost
        |         |         |
        +---------+---------+
                  |
                  v
          Optuna Optimization
                  |
                  v
          Fold Predictions
                  |
                  v
          Rank Transformation
                  |
                  v
          Rank-Based Averaging
                  |
                  v
          Final EV Predictions
```

## Tech Stack

- **Language:** Python 3.12
- **Data Processing:** NumPy, Pandas, SciPy
- **Machine Learning:** Scikit-learn
- **Gradient Boosting:** LightGBM, XGBoost, CatBoost
- **Hyperparameter Optimization:** Optuna
- **Environment:** Kaggle Notebooks

### Library Versions

| Library | Version |
|---|---|
| NumPy | 1.26 |
| Pandas | 2.2 |
| SciPy | 1.13 |
| Scikit-learn | 1.5 |
| LightGBM | 4.3 |
| XGBoost | 2.0 |
| CatBoost | 1.2 |
| Optuna | 3.6 |

## Installation

Clone the repository:

```bash
git clone https://github.com/CraftyO-o/EVPrediction.git
cd EVPrediction
```

Install the dependencies:

```bash
pip install numpy==1.26 pandas==2.2 scipy==1.13 \
scikit-learn==1.5 lightgbm==4.3 \
xgboost==2.0 catboost==1.2 optuna==3.6
```

Alternatively, install dependencies from the project's `requirements.txt` if available.

## Running the Project

1. Download the competition dataset from Kaggle.
2. Place the training and testing CSV files in the expected data directory.
3. Open the project notebooks or Python scripts.
4. Execute the preprocessing and feature engineering pipeline.
5. Train and optimize the individual gradient boosting models.
6. Generate fold-based predictions.
7. Apply rank-based ensemble averaging.
8. Export the final predictions in Kaggle submission format.

**Note:** Optuna hyperparameter optimization is computationally intensive and may require several hours when executed on CPU.

## Limitations and Future Work

### Limitations

- Digit-pattern and frequency features exploit characteristics of the synthetic dataset and may not generalize to real-world EV adoption data.
- Optuna optimization is computationally expensive.
- The dataset is based on synthetic survey data, which may not fully represent real consumer behaviour.

### Future Work

- Incorporate tabular deep learning architectures such as FT-Transformer and TabNet.
- Explore SHAP interaction values for feature selection and interpretability.
- Integrate real-world features such as charging station density, electricity prices, and vehicle preferences.
- Investigate additional ensemble architectures for improved prediction diversity.

## Contributors

Developed as an academic machine learning project at **Shiv Nadar University Chennai**.

- R. Yadunandan
- Rahul Balakrishnan Adhi
- Rohit Srinivasan
- S. Sharath

## References

1. Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*
2. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.*
3. Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.*
4. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.*

## Competition

[Kaggle Playground Series – Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9)

---

**Final Private Leaderboard AUC: 0.94541**  
