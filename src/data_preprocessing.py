# Import necessary libraries
!pip install imbalanced-learn
!pip install tabulate pandas_profiling
!pip install shap
!pip install xgboost lightgbm
!pip install lightgbm --timeout=100
from tabulate import tabulate
import pandas as pd
from scipy.stats import pearsonr
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

# Load the dataset
try:
    df = pd.read_csv('/kaggle/input/air-quality-health-impact-data/air_quality_health_impact_data.csv')
    display(df.head())
except FileNotFoundError:
    print("Error: 'air_quality_health_impact_data.csv' not found.")
    df = None
except Exception as e:
    print(f"An error occurred: {e}")
    df = None

# Initial Data Inspection

# Dataset Shape
print("="*60)
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Data Types
print("="*60)
print("Data types:\n")
print(tabulate(pd.DataFrame(df.dtypes, columns=["Type"]), headers="keys", tablefmt="psql"))

# Descriptive Statistics
print("="*60)
print("Descriptive statistics:\n")
print(tabulate(df.describe().T, headers="keys", tablefmt="psql", floatfmt=".2f"))

# Missing Values
print("="*60)
print("Missing values:\n")
missing_df = pd.DataFrame({
    'Missing Values': df.isnull().sum(),
    'Missing Percentage (%)': (df.isnull().sum() / len(df)) * 100
})
print(tabulate(missing_df[missing_df['Missing Values'] > 0], headers="keys", tablefmt="psql", floatfmt=".2f"))

# Unique Values in Categorical Columns
print("="*60)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(tabulate(pd.DataFrame(df[col].value_counts()), headers=["Value", "Count"], tablefmt="pretty"))

# Check for Negative Values
print("="*60)
print("Negative value checks:")
for col in ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore']:
    if df[col].dtype in ['int64', 'float64']:
        if (df[col] < 0).any():
            print(f"Inconsistency detected in '{col}': Negative values present.")

# Check for Unrealistic Large Values
print("="*60)
print("Large value checks:")
for col in ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore']:
    if df[col].dtype in ['int64', 'float64']:
        if df[col].max() > 10000:  # Arbitrary threshold
            print(f"Possible inconsistency in '{col}': Max value = {df[col].max():,.2f}.")

# =================================
# Visualization Settings
# =================================
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =================================
# Visualizing the Distribution of Air Quality Indicators & Health Outcomes
# =================================

air_quality_indicators = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3']
health_outcomes = ['RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore']
environmental_vars = ['Temperature', 'Humidity', 'WindSpeed']

# Remove RecordID if present
if 'RecordID' in df.columns:
    df.drop(columns=['RecordID'], inplace=True)

# Plot distributions
plt.figure(figsize=(18, 12))
all_vars = air_quality_indicators + health_outcomes + environmental_vars
for i, col in enumerate(all_vars):
    plt.subplot(4, 4, i + 1)  # 4x4 grid
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=10)
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# =================================
# Correlation Matrix and Heatmap
# =================================

# Compute correlation matrix
correlation_matrix = df.corr()

# Create the heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 10},
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.8}
)

plt.title('Correlation Matrix of Numerical Variables', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add white grid lines for better readability
for i in range(len(correlation_matrix) + 1):
    plt.axhline(i, color='white', lw=1)
    plt.axvline(i, color='white', lw=1)

plt.tight_layout()
plt.savefig('improved_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# =================================
# Calculate P-values for Correlations
# =================================

def calculate_pvalues(df):
    cols = df.columns
    p_values = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j:
                _, p_val = pearsonr(df.iloc[:, i], df.iloc[:, j])
                p_values.iloc[i, j] = p_val
    return p_values

p_values = calculate_pvalues(df)

print("\n📜 Matrix of p-values for statistical significance:")
print(p_values)

# Save correlation results to Excel
with pd.ExcelWriter('correlation_analysis_results.xlsx') as writer:
    correlation_matrix.to_excel(writer, sheet_name='Correlation Coefficients')
    p_values.to_excel(writer, sheet_name='P-values')
    (p_values < 0.05).to_excel(writer, sheet_name='Significant at 0.05')

# =================================
# Analyze Relationship Between Numerical Variables and HealthImpactClass
# =================================

ordered_classes = [0, 1, 2, 3, 4]
numerical_variables = air_quality_indicators + health_outcomes + environmental_vars

for var in numerical_variables:
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x='HealthImpactClass',
        y=var,
        data=df,
        order=ordered_classes,
        palette='Set2'
    )
    plt.title(f'Boxplot of {var} by Health Impact Class', fontsize=14)
    plt.xlabel("Health Impact Class")
    plt.ylabel(var)
    plt.show()
    
    # Display descriptive statistics
    print(f"📊 Descriptive Statistics for {var} by Health Impact Class:\n")
    print(df.groupby('HealthImpactClass')[var].agg(['mean', 'median', 'std']).round(2))
    print("="*60)

# Visualization settings
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# List of numerical columns
cols = [
    'AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3',
    'Temperature', 'Humidity', 'WindSpeed',
    'RespiratoryCases', 'CardiovascularCases',
    'HospitalAdmissions', 'HealthImpactScore'
]

for col in cols:
    # Compute statistics
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_val = df[col].std()
    skew_val = df[col].skew()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df[col],
        bins=30,
        color='skyblue',
        edgecolor='black',
        kde=True  # Enable KDE without additional settings
    )
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median = {median_val:.2f}')
    
    plt.title(f'Distribution of {col} (with Mean & Median)', pad=15)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Display descriptive statistics
    print(f"\n📊 Descriptive Statistics for {col}:")
    print(f"   ➡️ Mean: {mean_val:.2f}")
    print(f"   ➡️ Median: {median_val:.2f}")
    print(f"   ➡️ Standard Deviation: {std_val:.2f}")
    print(f"   ➡️ Skewness: {skew_val:.2f}")
    print("—" * 60)

# List of columns with outliers
outlier_columns = ['HealthImpactClass', 'CardiovascularCases', 'RespiratoryCases', 'AQI']

# Function to remove outliers only on specified columns
def remove_outliers_iqr_selected(dataframe, columns):
    df_clean = dataframe.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Apply outlier removal
df_cleaned = remove_outliers_iqr_selected(df, outlier_columns)

# Compare the number of rows before and after outlier removal
print(f"Number of rows before removing outliers: {df.shape[0]}")
print(f"Number of rows after removing outliers: {df_cleaned.shape[0]}")


# Compare Boxplot before and after removing outliers for the specified columns
for col in outlier_columns:
    plt.figure(figsize=(12, 4))

    # Before removing outliers
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[col])
    plt.title(f'{col} - Before Removing Outliers')

    # After removing outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_cleaned[col])
    plt.title(f'{col} - After Removing Outliers')

    plt.tight_layout()
    plt.show()


