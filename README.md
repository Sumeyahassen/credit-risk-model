# Credit Risk Probability Model for Bati Bank

**End-to-End Credit Risk Scoring using Alternative Data (RFM + Machine Learning)**

---

## Project Overview

This project develops a complete **credit risk probability model** for Bati Bank's Buy-Now-Pay-Later (BNPL) service in partnership with an eCommerce platform (Xente). 

Since the dataset does not contain an explicit default label, we engineered a **proxy target** using customer behavioral data and deployed the solution as a production-ready REST API.

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord influence modeling choices?

The Basel II Capital Accord sets international standards for banking regulation with strong emphasis on:

- Accurate **Probability of Default (PD)** estimation
- Model **interpretability** and transparency
- Thorough **documentation** and validation
- Ongoing performance monitoring

**Our Approach**: We prioritized interpretability, reproducibility, and documentation throughout the project to align with regulatory expectations, especially in the Ethiopian banking context under the National Bank of Ethiopia.

### 2. Why do we need a Proxy Variable? What are the business risks?

The Xente dataset has no direct "default" label. Therefore, a **proxy target** (`is_high_risk`) was created using **RFM Analysis + K-Means Clustering**.

**Business Risks**:
- Misclassification of customers (lost revenue or increased bad debt)
- Concept drift over time
- Regulatory scrutiny on proxy validity
- Potential bias in customer segmentation

### 3. Trade-offs: Interpretable vs High-Performance Models

| Aspect                    | Interpretable (Logistic Regression + WoE) | Complex (Random Forest / XGBoost) |
|---------------------------|-------------------------------------------|-----------------------------------|
| Interpretability          | Very High                                 | Low (needs SHAP)                  |
| Regulatory Acceptance     | High                                      | Medium                            |
| Predictive Performance    | Moderate                                  | Higher                            |
| Training Speed            | Fast                                      | Slower                            |
| Business Use Case         | Regulatory scorecard                      | Internal decisioning              |
| Overfitting Risk          | Lower                                     | Higher                            |

**Strategy**: We trained both types of models and selected based on performance while maintaining documentation.

---

## Key EDA Insights (Task 2)

1. **High Class Imbalance**: `FraudResult` is extremely imbalanced (~0.2% fraud) and cannot be used as the target variable.
2. **Transaction Nature**: `Amount` contains negative values (refunds/credits). We used `Value` (absolute amount) for monetary calculations.
3. **Customer Behavior**: 95,662 transactions from **3,742 unique customers** (average ~25.56 transactions per customer).
4. **High Variation**: Some customers made over 4,000 transactions, while many are low-activity — justifying customer-level aggregation.
5. **Constant Features**: `CountryCode` and `CurrencyCode` have no variation and were dropped.
6. **Time Feature**: `TransactionStartTime` was converted to datetime for RFM calculations.

---

## Methodology

### Proxy Target Creation (RFM + Clustering)

We calculated **RFM** features and applied K-Means clustering (4 clusters):

| Cluster | Recency | Frequency | Monetary   | Risk Label    |
|---------|---------|-----------|------------|---------------|
| 0       | 61.84   | 7.73      | 90,694     | **High Risk** |
| 1       | 29.00   | 4091      | 104.9M     | Very Low Risk |
| 2       | 12.69   | 34.72     | 224k       | Low Risk      |
| 3       | 21.33   | 109       | 64.87M     | Medium Risk   |

**High-risk customers** = Cluster 0 (disengaged / low activity).

### Feature Engineering

- Aggregated transaction data to **CustomerId** level
- Created statistical features (`AvgValue`, `StdValue`, `MaxValue`, etc.)
- Handled missing values with median imputation
- Standardized numerical features

### Model Training & Results

Models were trained with proper NaN handling and tracked using **MLflow**.

**Final Model Performance:**

| Model                  | ROC-AUC | F1 Score | Precision | Recall | Accuracy |
|------------------------|---------|----------|-----------|--------|----------|
| Logistic Regression    | 0.9998  | 0.9972   | 0.9944    | 1.0000 | 0.9979   |
| **Random Forest**      | **1.0000** | **0.9958** | **0.9917** | **1.0000** | **0.9968** |

**Best Model**: Random Forest

---

## Project Structure
