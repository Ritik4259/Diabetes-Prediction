# Diabetes Prediction using Machine Learning

## Imbalanced Classification with Recall Optimization

A complete end-to-end Machine Learning classification project focused on predicting diabetes while handling severe class imbalance and optimizing recall to reduce false negatives.

Implementation file: Dataset/diabetes_prediction_dataset.csv

---

# Project Objective

## Business Problem

Early detection of diabetes is critical. In medical diagnosis:

* False negatives are more dangerous than false positives
* Accuracy can be misleading in imbalanced datasets
* Recall must be prioritized

Binary classification setup:

$$
\hat{y} = f(X)
$$

Where:

* $X$ → Patient features
* $y$ → Diabetes label (0 = No, 1 = Yes)
* $\hat{y}$ → Predicted outcome

---

# Dataset Overview

## Features

The dataset includes:

* Demographic features

  * age
  * gender

* Lifestyle indicators

  * smoking_history

* Clinical measurements

  * bmi
  * HbA1c_level
  * blood_glucose_level

Target variable:

* diabetes

---

# Data Preprocessing

## Smoking History Mapping

Categorical inconsistencies were standardized:

* never
* former
* current
* unknown

Then:

* Applied One-Hot Encoding
* Converted encoded columns to integer type
* Dropped redundant dummy variables

---

## Outlier Filtering

BMI values were filtered using:

$$
12 \leq \text{BMI} \leq 70
$$

This removes unrealistic physiological values.

---

## Feature Scaling

Used RobustScaler to reduce outlier influence:

$$
X_{scaled} = \frac{X - \text{median}}{\text{IQR}}
$$

Applied only to numerical features:

* age
* bmi
* HbA1c_level
* blood_glucose_level

---

# Exploratory Data Analysis

Performed:

* Class distribution analysis
* Histogram plots for numeric features
* Boxplots grouped by diabetes class
* Correlation heatmap

Key Observations:

* Severe class imbalance
* HbA1c and blood glucose show strong class separation
* BMI and age moderately correlated with diabetes

---

# Model Training

## Train-Test Split

* Stratified split
* 70 percent training
* 30 percent testing

---

# Models Implemented

## Support Vector Machine

* Kernel: RBF
* Class weight: balanced
* Probability enabled
* Tuned C and gamma

Purpose:

* Improve minority detection
* Control overfitting

---

## Logistic Regression

* Solver: liblinear
* Supports L1 and L2 regularization
* Hyperparameter tuning using GridSearchCV

Optimized for:

* F1 score
* Balanced accuracy

### Threshold Tuning

Custom probability threshold applied:

$$
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1) \geq 0.68 \
0 & \text{otherwise}
\end{cases}
$$

Improves recall for diabetic cases.

---

## Decision Tree

Tuned parameters:

* max_depth
* min_samples_split
* min_samples_leaf
* criterion

Optimized using GridSearchCV.

---

## Random Forest

Initial model:

* Class weight adjustment

Advanced optimization using RandomizedSearchCV:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf
* max_features
* criterion
* bootstrap

### Custom Threshold

$$
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1) \geq 0.38 \
0 & \text{otherwise}
\end{cases}
$$

Significantly improves recall performance.

---

# Evaluation Metrics

Because of class imbalance, the following metrics were emphasized:

## Recall

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

* Measures ability to detect diabetic patients
* Primary optimization objective

## Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

* Measures correctness of positive predictions

## F1 Score

$$
F1 = \frac{2 \cdot (\text{Precision} \cdot \text{Recall})}{\text{Precision} + \text{Recall}}
$$

* Balances precision and recall

## Additional Tools

* Confusion Matrix
* ROC Curve
* AUC Score
* Classification Report

---

# Key Insights

* Accuracy is unreliable for imbalanced medical datasets
* Class weighting significantly improves minority detection
* Threshold tuning is critical for healthcare classification
* Random Forest achieved strong balanced performance
* HbA1c and blood glucose are the most predictive features

---

# Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

# Project Structure

* diabetes_prediction.py
* Dataset folder
* README.md

---

# How to Run

## Step 1
```
Clone the repository
```
## Step 2

Install dependencies
```
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Step 3

Run the script
```
python diabetes_prediction.py
```
---

# Professional Highlights

* Built a recall-focused healthcare classification system
* Handled severe class imbalance using class weights
* Applied systematic hyperparameter optimization
* Implemented probability threshold tuning
* Compared multiple classification algorithms
* Used medical-relevant evaluation metrics

