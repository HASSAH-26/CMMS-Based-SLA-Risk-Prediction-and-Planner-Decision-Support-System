# CMMS SLA Breach Prediction

## Machine Learning Based SLA Breach Prediction for CMMS Work Requests

### Project Overview

This project develops a machine learning based decision support system
for predicting the risk of SLA breach for maintenance work requests
created in a Computerized Maintenance Management System (CMMS).

The system uses historical maintenance data and predicts whether a new
maintenance request is likely to meet or breach its SLA.

## Business Problem

Facility management teams receive a large number of maintenance work
requests. Some requests may take longer than the allowed SLA response
time.

The objective of this project is to provide planners with an additional
data-driven indication of SLA breach risk.

## Machine Learning Problem

This is a supervised binary classification problem.

Target variable:

- 0 = SLA Met
- 1 = SLA Breach

## Features

The model uses information including:

- Building
- Department
- Asset Category
- Asset Type
- Equipment Age
- Criticality
- Complaint Type
- Maintenance Type
- Priority
- Emergency
- Technician Trade
- Technician Count
- Spare Requirement
- PTW Requirement
- Outside Temperature
- Response Target
- Estimated Cost
- Request Date/Time features

## Machine Learning Models

Two models were evaluated:

1. Logistic Regression
2. Random Forest

### Final Model

Logistic Regression was selected as the final model based on the
evaluation results.

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 50.48% | 34.77% | 49.33% | 40.79% | 0.5027 |
| Random Forest | 52.34% | 34.45% | 41.94% | 37.83% | 0.4994 |

The results indicate that the current historical dataset provides
limited predictive power. Additional operational features may be
required to improve the model.

## Streamlit Application

The trained model is deployed using Streamlit.

The planner enters information about a new maintenance work request.

The application then displays:

- SLA prediction
- SLA breach probability
- Risk level
- Planner recommendation

## Project Structure

```text
CMMS-SLA-Breach-Prediction/
│
├── app.py
├── cmms_sla_risk_model.pkl
├── cmms_model_columns.pkl
├── requirements.txt
└── README.md
