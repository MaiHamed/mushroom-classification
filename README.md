
# Mushroom Classification – Data Mining & Business Intelligence

## Project Information

**Course:** CSE387 – Data Mining and Business Intelligence
**Project Title:** Mushroom Classification

**Submitted To:**

* Mahmoud Mounir
* Esraa Karam

**Team Members:**

* Rawan Essam Hussein – 23P0309
* Mai Hamed Hussein Ahmed – 23P0261
* Maryam Hamdy Hassan – 23P0260
* Mariam Riyad Freeg Aboelyazed – 23P0147
* Rawan Hany Shaker Hussein – 23P0393
* Basmala Hany Mohamed Taha – 23P0008

**Submission Date:** 6 December 2025

---

# README

## Project Overview

This project applies Data Mining and Machine Learning techniques to the Mushroom Dataset in order to classify mushrooms as either edible or poisonous based on their categorical characteristics.

The project is divided into two phases:

* **Phase 1:** Data understanding, preprocessing, outlier detection, and categorical correlation analysis.
* **Phase 2:** Building and evaluating multiple classification models to determine the most accurate and safest classifier.

The dataset contains mushroom attributes such as cap shape, odor, gill size, stalk characteristics, habitat, and more.

---

# Dataset Information

## Dataset

The project uses the Mushroom Dataset (`mushrooms.csv`).

### Dataset Characteristics

* Total Rows: 8124
* Total Columns: 23
* Feature Type: Categorical/Nominal
* Target Variable:

  * `e` → Edible
  * `p` → Poisonous

### Main Features

The dataset includes:

* Cap properties
* Gill structure and color
* Stalk features
* Veil and ring information
* Spore print color
* Population and habitat
* Odor characteristics

---

# Phase 1 – Data Understanding & Preprocessing

## Objectives

* Explore and understand the dataset
* Handle missing values
* Detect categorical outliers
* Analyze relationships between features
* Prepare the dataset for classification

---

## Preprocessing Steps

### 1. Missing Value Handling

Missing values represented by `?` were replaced using:

```python
SimpleImputer(strategy='most_frequent')
```

### 2. Data Validation

All features were verified to be categorical (`object` type).

### 3. Encoding

* Label Encoding was used for Decision Tree.
* One-Hot Encoding was used for KNN, Random Forest, and Naïve Bayes.

### 4. Scaling

Normalization and scaling were not applied because the dataset contains only nominal categorical attributes.

---

# Outlier Detection

Since the dataset is categorical, traditional numerical outlier methods were not suitable. Three categorical approaches were used.

## 1. Frequency-Based Outliers

* Categories appearing in less than 5% of the dataset were marked as rare.
* Useful for detecting uncommon mushroom characteristics.

## 2. Mode-Based Outliers

* Categories occurring below 5% of the feature mode frequency were identified as outliers.
* Focuses on deviation from dominant categories.

## 3. K-Modes Clustering Outliers

The K-Modes clustering algorithm was applied using:

```python
KModes(
    n_clusters=3,
    init='Huang',
    n_init=10
)
```

Small clusters containing fewer than 5% of samples were treated as cluster-based outliers.

### Why Outliers Were Kept

The detected outliers represent valid but rare mushroom patterns rather than data errors. Removing them would reduce useful predictive information.

---

# Correlation Analysis

Because all features are categorical, Pearson correlation was unsuitable.

Two categorical association measures were used instead:

## 1. Cramér’s V

Measures symmetric association strength between categorical variables.

### Strong Relationships Found

* odor ↔ class
* gill-size ↔ class
* stalk-color-above-ring ↔ gill-attachment

## 2. Theil’s U

Measures directional dependency and predictive influence.

### Strong Predictive Features

* odor → class
* gill-size → class

These features strongly influence mushroom edibility prediction.

---

# Phase 2 – Classification Modeling

## Objective

Train machine learning models to classify mushrooms as edible or poisonous.

---

# Models Implemented

## 1. Decision Tree Classifier

### Features

* Uses Gini impurity
* Hyperparameter tuning using `GridSearchCV`

### Performance

* Accuracy: 1.000
* Zero false positives
* Zero false negatives

### Important Features

* odor
* gill-size
* spore-print-color
* stalk-surface-below-ring

### Advantages

* Highly interpretable
* Perfect classification
* Best overall model

---

## 2. K-Nearest Neighbors (KNN)

### Configuration

```python
KNeighborsClassifier(n_neighbors=5)
```

### Performance

* Accuracy: 1.000
* Perfect classification on test data

### Notes

KNN performed extremely well because poisonous mushrooms form distinct categorical patterns after one-hot encoding.

---

## 3. Random Forest Classifier

### Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None
)
```

### Performance

* Accuracy: 1.000
* Strong robustness with cross-validation

### Important Features

* odor
* spore-print-color
* stalk-surface-below-ring

---

## 4. Naïve Bayes Classifier

### Model Used

```python
BernoulliNB()
```

### Performance

* Accuracy: 0.9397

### Limitations

* Assumes feature independence
* Produced dangerous false positives

### Errors

* 14 poisonous mushrooms predicted as edible
* 84 edible mushrooms predicted as poisonous

---

# Model Comparison Summary

| Model         | Accuracy | Error Rate | False Positives | Notes                 |
| ------------- | -------- | ---------- | --------------- | --------------------- |
| Decision Tree | 1.000    | 0%         | 0               | Best Overall          |
| KNN           | 1.000    | 0%         | 0               | Excellent performance |
| Random Forest | 1.000    | 0%         | 0               | Very robust           |
| Naïve Bayes   | 0.9397   | 6.03%      | 14              | Unsafe predictions    |

---

# Final Recommendation

## Best Model: Decision Tree Classifier

The Decision Tree model was selected as the best classifier because it:

* Achieved perfect accuracy
* Produced no dangerous predictions
* Had the highest interpretability
* Achieved the best AUC score
* Clearly showed feature importance

---

# Tools & Libraries Used

| Purpose               | Libraries                                                              |
| --------------------- | ---------------------------------------------------------------------- |
| Data Handling         | pandas, numpy                                                          |
| Visualization         | matplotlib, seaborn                                                    |
| Preprocessing         | sklearn.preprocessing, sklearn.impute                                  |
| Clustering            | kmodes                                                                 |
| Statistical Analysis  | scipy                                                                  |
| Classification Models | sklearn.tree, sklearn.neighbors, sklearn.ensemble, sklearn.naive_bayes |
| Model Evaluation      | sklearn.metrics, GridSearchCV                                          |
| Model Saving          | joblib                                                                 |

---

# Project Outcomes

By the end of this project:

* The dataset was fully cleaned and analyzed.
* Rare categorical outliers were successfully detected.
* Strong feature relationships were identified.
* Multiple machine learning models were evaluated.
* Decision Tree, KNN, and Random Forest achieved perfect classification.
* Decision Tree was selected as the safest and most interpretable solution.

---

# Future Improvements

Possible future enhancements include:

* Testing additional ensemble models
* Applying feature selection techniques
* Deploying the classifier as a web application
* Using explainable AI tools for deeper interpretability
* Comparing with deep learning approaches

---

# Conclusion

This project demonstrates the effectiveness of categorical data mining techniques for mushroom toxicity classification. Through preprocessing, outlier analysis, correlation analysis, and machine learning classification, highly accurate predictive models were achieved.

The results confirm that categorical features such as odor and gill-size are highly informative for toxicity prediction, enabling near-perfect classification performance.
