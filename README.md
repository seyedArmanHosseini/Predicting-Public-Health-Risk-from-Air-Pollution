# Predicting-Public-Health-Risk-from-Air-Pollution
# Air Pollution Health Impact Analysis

This project presents a comprehensive machine learning analysis of the health impacts of air pollution using real-world data. The dataset was obtained from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/seyedarmanhossaini)) and includes a range of air quality indicators alongside health-related outcomes.

## Dataset Description

The dataset contains air pollution metrics and associated public health records. Key features include:

- **Pollution Indicators**: AQI, PM10, PM2.5, NO2, SO2, O3  
- **Weather Conditions**: Temperature, Humidity, WindSpeed  
- **Health Data**: RespiratoryCases, CardiovascularCases, HospitalAdmissions  
- **Target Variables**:
  - `HealthImpactScore`: A numeric score indicating total health impact.
  - `HealthImpactClass`: A categorical label (Very High, High, Moderate, Low, Very Low) derived from the score.

## Objective

The goal is to **predict the severity class of health impacts** caused by air pollution, using classification models trained on air quality and weather data. This allows for proactive public health responses and better environmental policy planning.

## Models Used

Several machine learning models were tested. The best performing models included:

- **XGBoost Classifier**  
- **LightGBM Classifier**  

These models outperformed traditional decision trees and random forests, based on classification accuracy and confusion matrix evaluations.

## Methods

- Data Cleaning & Preprocessing
- Feature Engineering
- Class Mapping Based on `HealthImpactScore`
- Model Training & Tuning (using Scikit-learn, XGBoost, LightGBM)
- Evaluation Metrics: Accuracy, F1-score, Confusion Matrix
- SHAP Analysis for model interpretability

## Insights

- PM2.5, NO2, and AQI were the most influential features in predicting severe health impacts.
- High levels of particulate matter (PM) strongly correlated with increased respiratory and cardiovascular admissions.

## Folder Structure

air_quality_health_impact/
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── exploratory_data_analysis.ipynb
│ └── model_training.ipynb
├── src/
│ ├── data_processing.py
│ ├── model.py
│ └── utils.py
├── requirements.txt
└── README.md


- `data/`: Contains the datasets used for training and testing the models.
- `notebooks/`: Jupyter notebooks for data preprocessing, exploratory analysis, and model training.
- `src/`: Python scripts for data processing, modeling, and utility functions.
- `requirements.txt`: List of dependencies required to run the project.

Links
[Kaggle Dataset Link](https://www.kaggle.com/datasets/rabieelkharoua/air-quality-and-health-impact-dataset)

XGBoost Documentation

LightGBM Documentation
