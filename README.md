# Credit Risk Probability Model for Bati Bank

**End-to-End Implementation using Alternative Data (RFM + Machine Learning)**

## Project Overview
This project builds a credit scoring model for Bati Bank's Buy-Now-Pay-Later (BNPL) service in partnership with an eCommerce platform. The model uses transaction behavioral data to predict customer credit risk.

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord influence modeling choices?

The Basel II Capital Accord is the international standard for banking regulation. It requires banks to maintain sufficient capital to cover credit risk.

**Key Influences on Our Model:**
- **Pillar 1 (Minimum Capital)**: Requires accurate estimation of **Probability of Default (PD)**. Our model must output a reliable risk probability.
- **Model Validation & Documentation**: Every step (feature selection, proxy target definition, model choice) must be well documented and justifiable to regulators.
- **Interpretability**: Regulators prefer transparent models. This is why we will consider Logistic Regression with Weight of Evidence (WoE) alongside more complex models like Random Forest or XGBoost.
- **Ongoing Monitoring**: The model must be monitored for performance degradation over time.

In the Ethiopian banking context, compliance with Basel II (or similar local regulations from National Bank of Ethiopia) is critical for risk management.

### 2. Why do we need a Proxy Variable? What are the business risks?

The Xente dataset does **not** contain an explicit "default" label. Therefore, we cannot directly train a supervised model.

**Solution**: We create a **proxy target** (`is_high_risk`) using **RFM Analysis + K-Means Clustering**.

**Business Risks of Using Proxy:**
- **Misclassification Risk**: Some low-engagement customers might be good payers → lost business opportunities.
- **Concept Drift**: Customer behavior can change over time, making the proxy less accurate.
- **Regulatory Challenge**: Auditors may question the validity of the proxy. We must provide strong justification and backtesting.
- **Bias Risk**: If RFM features correlate with demographic factors, it could introduce unintended bias.

### 3. Trade-offs Between Interpretable vs High-Performance Models

| Aspect                    | Interpretable Model (Logistic Regression + WoE) | Complex Model (XGBoost / Random Forest) |
|--------------------------|------------------------------------------------|---------------------------------------|
| Interpretability         | Very High (easy to explain to risk team)      | Low (needs SHAP values)              |
| Regulatory Acceptance    | High                                           | Medium (requires extra documentation)|
| Predictive Performance   | Moderate                                       | Usually Higher                       |
| Implementation Speed     | Fast                                           | Slower                               |
| Business Use Case        | Scorecard for loan approval & audit           | Internal ranking + ensemble          |
| Risk of Overfitting      | Lower                                          | Higher                               |

**Our Strategy**: Start with interpretable models for regulatory comfort, then compare with powerful models. Use SHAP for explainability on complex models.

---

## Project Goals
- Define proxy target using RFM
- Build reproducible feature engineering pipeline
- Train and compare multiple models using MLflow
- Deploy model as FastAPI service
- Set up CI/CD pipeline

**Team**: Sumeya (Analytics Engineer at Bati Bank)
## Task-2
## Key EDA Insights

1. **High Class Imbalance**: `FraudResult` cannot be used as target variable (only ~0.2% fraud cases).

2. **Transaction Nature**: `Amount` has negative values (refunds/credits). We should use `Value` for monetary calculations.

3. **Multiple Transactions per Customer**: We must aggregate data at `CustomerId` level.

4. **Constant Columns**: `CountryCode` and `CurrencyCode` have no variation → can be dropped.

5. **Time Feature**: `TransactionStartTime` needs to be converted to datetime for RFM analysis.