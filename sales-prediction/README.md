# Sales Prediction with TensorFlow and Real-Time Kafka Streaming

A machine learning pipeline for predicting aggregated sales performance using TensorFlow regression, with real-time streaming to Apache Kafka and visualization through Kafdrop.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Architecture Overview](#architecture-overview)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis Findings](#exploratory-data-analysis-findings)
6. [Machine Learning Models](#machine-learning-models)
7. [Kafka Streaming Results](#kafka-streaming-results)
8. [Key Findings Summary](#key-findings-summary)
9. [Docker Setup](#docker-setup)
10. [Quick Start](#quick-start)
11. [Requirements](#requirements)

---

## Project Goal

The goal of this project is to **predict aggregated sales performance** using a regression model, focusing on **total monthly or customer-level sales** rather than individual transactions.

After generating predictions in TensorFlow, the model output is integrated into a **real-time streaming pipeline** using Apache Kafka and visualized through Kafdrop.

This project demonstrates how machine learning models can be **deployed and monitored in a scalable cloud environment** through data streaming technologies, bridging predictive analytics with real-time operational insight.

---

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  TensorFlow      |---->|  Apache Kafka    |---->|    Kafdrop       |
|  Model Training  |     |  Message Broker  |     |  Visualization   |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
   predictions_tf         sales_predictions         Web UI @ :9000
                              topic
```

**Data Flow:**
1. TensorFlow model trains on historical sales data
2. Model generates predictions on test set
3. Kafka producer streams each prediction record to the `sales_predictions` topic
4. Kafdrop provides real-time visualization of message offsets and payloads
5. Each offset confirms successful send/receive, forming an end-to-end data flow

---

## Dataset Description

The dataset (`raw_dataset_week4.csv`) contains customer and transaction data with the following features:

| Feature | Description |
|---------|-------------|
| `Sales` | Target variable - aggregated sales amount |
| `Marketing_Spend` | Marketing expenditure |
| `Seasonality` | Seasonal period indicator |
| `Spending_Score` | Customer spending behavior score |
| `Gender` | Customer gender |
| `Customer_Churn` | Churn status (binary) |
| `Credit_Score` | Customer credit rating |
| `Defaulted` | Loan default status |

---

## Data Preprocessing

### Missing Value Treatment

- **Numeric columns**: Imputed with column mean values
- **Categorical columns**: Imputed with mode (most frequent value)

### Outlier Removal

Outliers removed using the **IQR method**:

```
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

**Results:** 500 rows loaded → 405 rows after cleaning

---

## Exploratory Data Analysis Findings

### 1. Sales Distribution
- Analyzed using histogram with KDE overlay
- Reveals central tendency, spread, and potential multimodal patterns
- **Business Insight**: Helps set realistic sales targets and identify market segments

### 2. Seasonality Impact
- Box plots show significant variation across seasonal periods
- Some seasons more predictable than others
- **Business Insight**: Dynamic pricing and inventory allocation needed

### 3. Feature Correlations
- **Sales vs Marketing_Spend**: Positive correlation confirms marketing effectiveness
- **Credit_Score vs Defaulted**: Negative correlation validates credit risk assessment
- **Business Insight**: Guides feature selection for predictive models

### 4. Customer Churn by Gender
- Identifies if churn rates differ by demographic
- **Business Insight**: Informs targeted retention strategies

### 5. Credit Score and Loan Default
- Defaulted customers show lower median credit scores
- **Business Insight**: Establishes thresholds for lending decisions

---

## Machine Learning Models

### Model 1: Linear Regression
- **Target**: Spending_Score
- **Feature**: Sales
- **R² Score**: -0.00
- **Use Case**: Interpretable baseline model

### Model 2: TensorFlow Neural Network
- **Target**: Sales
- **Features**: Marketing_Spend, Seasonality
- **Architecture**:
  ```
  Input(2) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Output(1)
  ```
- **Training**: Adam optimizer, MSE loss, 50 epochs, 80/20 split
- **Use Case**: Captures non-linear relationships for production deployment

---

## Kafka Streaming Results

### Pipeline Overview
```
┌────────────────────────┬───────────────────┐
│         Metric         │       Value       │
├────────────────────────┼───────────────────┤
│ Total Records Streamed │ 81                │
├────────────────────────┼───────────────────┤
│ Kafka Topic            │ sales_predictions │
├────────────────────────┼───────────────────┤
│ Broker                 │ localhost:9092    │
└────────────────────────┴───────────────────┘
```

### Prediction vs Actual Statistics
```
┌─────────┬──────────────────┬──────────────────┐
│ Metric  │ Predicted Sales  │   Actual Sales   │
├─────────┼──────────────────┼──────────────────┤
│ Range   │ $5,218 - $84,287 │ $6,051 - $99,220 │
├─────────┼──────────────────┼──────────────────┤
│ Mean    │ $44,211          │ $50,764          │
├─────────┼──────────────────┼──────────────────┤
│ Median  │ $40,937          │ $44,104          │
├─────────┼──────────────────┼──────────────────┤
│ Std Dev │ $23,151          │ $27,275          │
└─────────┴──────────────────┴──────────────────┘
```

### Error Analysis
```
┌────────────────────────────────┬────────────┐
│             Metric             │   Value    │
├────────────────────────────────┼────────────┤
│ Mean Error (Bias)              │ $6,552.95  │
├────────────────────────────────┼────────────┤
│ Mean Absolute Error (MAE)      │ $30,752.86 │
├────────────────────────────────┼────────────┤
│ Root Mean Square Error (RMSE)  │ $36,567.27 │
├────────────────────────────────┼────────────┤
│ Mean Absolute Percentage Error │ 88.34%     │
└────────────────────────────────┴────────────┘
```

### Prediction Accuracy
```
┌──────────────────────┬─────────┬────────────┐
│  Accuracy Threshold  │ Records │ Percentage │
├──────────────────────┼─────────┼────────────┤
│ Within 10% of actual │ 4/81    │ 4.9%       │
├──────────────────────┼─────────┼────────────┤
│ Within 25% of actual │ 15/81   │ 18.5%      │
├──────────────────────┼─────────┼────────────┤
│ Within 50% of actual │ 32/81   │ 39.5%      │
└──────────────────────┴─────────┴────────────┘
```

**Prediction Direction:**
- Over-predicted: 38 records (46.9%)
- Under-predicted: 43 records (53.1%)

### Best & Worst Predictions

**Best (Lowest Error):**
1. Predicted: $27,797 | Actual: $28,569 | Error: $771
2. Predicted: $69,296 | Actual: $68,344 | Error: -$952
3. Predicted: $9,025 | Actual: $7,775 | Error: -$1,250

**Needs Improvement (Highest Error):**
1. Predicted: $5,218 | Actual: $95,022 | Error: $89,803
2. Predicted: $7,950 | Actual: $96,066 | Error: $88,115
3. Predicted: $15,427 | Actual: $98,459 | Error: $83,031

### Kafka Message Schema

```json
{
    "prediction_index": int,
    "predicted_sales": float,
    "actual_sales": float,
    "error": float
}
```

---

## Key Findings Summary

| Category | Finding | Action |
|----------|---------|--------|
| Seasonality | Sales vary significantly by season | Implement dynamic pricing |
| Marketing | Positive ROI captured by neural network | Optimize budget allocation |
| Churn | Patterns identified by gender | Develop targeted retention |
| Credit Risk | Credit score predicts defaults | Set approval thresholds |
| Model Bias | Tends to underestimate high sales | Add more features |

### Key Insights

1. **Model Performance**: The current model has high error rates, suggesting the features (Marketing_Spend, Seasonality) alone don't fully explain Sales variance.

2. **Bias**: The positive mean error ($6,552) indicates the model tends to underestimate actual sales.

3. **High Variance Cases**: The worst predictions occur when actual sales are high ($95k+) but the model predicts low values - likely missing important features that drive high sales.

4. **Improvement Opportunities**: Consider adding more features (Customer demographics, Credit Score, historical trends) to improve accuracy.

---

## Docker Setup

The project uses Docker to containerize the full streaming environment.

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Zookeeper | confluentinc/cp-zookeeper:7.5.0 | 2181 | Kafka coordination |
| Kafka | confluentinc/cp-kafka:7.5.0 | 9092 | Message broker |
| Kafdrop | obsidiandynamics/kafdrop:latest | 9000 | Web UI for Kafka |

---

## Quick Start

### Prerequisites

- Docker Desktop installed and running
- Python 3.11+
- Required Python packages installed

### Step 1: Start the Streaming Infrastructure

```bash
docker-compose up -d
```

Wait ~30 seconds for Kafka to initialize.

### Step 2: Verify Services

```bash
docker-compose ps
```

All three containers should show "Up" status.

### Step 3: Open Kafdrop

Navigate to http://localhost:9000 in your browser.

### Step 4: Run the Pipeline

**Option A - Run the Python script:**
```bash
python run_notebook.py
```

**Option B - Run the Jupyter notebook:**
1. Open `EDA_Sales_Prediction_Regression_TF (2).ipynb`
2. Run all cells sequentially
3. The Kafka producer cell will stream predictions to the `sales_predictions` topic

### Step 5: View Predictions in Kafdrop

1. Refresh Kafdrop at http://localhost:9000
2. Click on the `sales_predictions` topic
3. View messages to see prediction records with offsets

### Stopping the Environment

```bash
docker-compose down
```

---

## Requirements

### Python Packages

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
kafka-python
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow kafka-python
```

### Docker

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- docker-compose

---

## Project Structure

```
Sales_Prediction_Kafka/
│
├── EDA_Sales_Prediction_Regression_TF (2).ipynb   # Main Jupyter notebook
├── run_notebook.py                                 # Standalone Python script
├── docker-compose.yml                              # Kafka infrastructure
├── README.md                                       # This file
└── raw_dataset_week4.csv                           # Dataset (in Documents)
```

---

## Model Persistence

The trained model is saved for deployment:

```python
model.save("sales_model.h5")
```

Model saved to: `C:\Users\angel\sales_model.h5`

---

## Conclusion

This project demonstrates a complete **end-to-end machine learning pipeline**:

1. **Data Preprocessing**: Cleaned raw data with missing value imputation and outlier removal
2. **Exploratory Analysis**: Identified key business drivers (seasonality, marketing spend, customer behavior)
3. **Predictive Modeling**: Built interpretable (Linear Regression) and powerful (TensorFlow) models
4. **Real-Time Streaming**: Deployed predictions to Kafka for live monitoring
5. **Containerization**: Packaged infrastructure with Docker for reproducibility

The architecture bridges **predictive analytics with real-time operational insight**, demonstrating how ML models can be deployed and monitored in a **scalable cloud environment** through data streaming technologies.

---

## Access Points

| Service | URL |
|---------|-----|
| Kafdrop UI | http://localhost:9000 |
| Kafka Broker | localhost:9092 |
| Zookeeper | localhost:2181 |
