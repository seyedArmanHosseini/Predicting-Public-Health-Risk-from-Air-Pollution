# 1. Take a sample of 500 rows from X_train while keeping all columns
X_sample = X_train.sample(500, random_state=42).copy()

# 2. Create a SHAP Explainer for the XGBoost model
explainer = shap.Explainer(xgb_model, X_sample)

# 3. Compute SHAP values
shap_values = explainer(X_sample)

# 4. Convert SHAP values to NumPy array
raw_shap = shap_values.values  # shape: (n_samples, n_features, n_classes)

print("SHAP values shape:", raw_shap.shape)
print("Number of features in X_sample:", X_sample.shape[1])

# 5. Check consistency
assert raw_shap.shape[1] == X_sample.shape[1], "Mismatch in number of features!"

# 6. Compute mean absolute SHAP values per class
mean_shaps = np.mean(np.abs(raw_shap), axis=0)  # shape: (n_features, n_classes)

# 7. Create a DataFrame for easier visualization
shap_df = pd.DataFrame(mean_shaps, index=X_sample.columns,
                       columns=[f"Class {i}" for i in range(raw_shap.shape[2])])

# 8. Select only important target classes (1 to 4)
selected_classes = [1, 2, 3, 4]
shap_df = shap_df[[f"Class {i}" for i in selected_classes]]

# 9. Plot SHAP importance per class
import seaborn as sns

sns.set(style="whitegrid")
for cls in selected_classes:
    plt.figure(figsize=(10, 5))
    sorted_vals = shap_df[f"Class {cls}"].sort_values(ascending=True)
    sns.barplot(x=sorted_vals.values, y=sorted_vals.index, palette="viridis")
    plt.title(f"Top Feature Importances - Class {cls}", fontsize=14, weight='bold')
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import is_classifier
import xgboost as xgb
import lightgbm as lgb

# Load your dataset (assuming df is already loaded)
features = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2']
target = 'HealthImpactClass'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models with optimized parameters
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}

# Class name mapping (should match your actual classes)
class_names = {
    0: 'Very High',
    1: 'High',
    2: 'Moderate',
    3: 'Low',
    4: 'Very Low'
}

# Create output folder
output_dir = "pdp_plots"
os.makedirs(output_dir, exist_ok=True)

def plot_pdp_xgboost(model, X, feature, class_id):
    """Custom PDP implementation for XGBoost"""
    X_temp = X.copy()
    unique_vals = np.linspace(X_temp[feature].min(), X_temp[feature].max(), 50)
    
    preds = []
    for val in unique_vals:
        X_temp[feature] = val
        pred = model.predict_proba(X_temp)[:, class_id]
        preds.append(pred.mean())
    
    plt.plot(unique_vals, preds)
    return unique_vals, preds

def plot_pdp_compatible(model, model_name, X, features, class_id, class_names):
    """Universal PDP plotting function that handles all model types"""
    for feature in features:
        try:
            plt.figure(figsize=(10, 6))
            
            if model_name == 'XGBoost':
                # Use custom implementation for XGBoost
                x_vals, y_vals = plot_pdp_xgboost(model, X, feature, class_id)
                plt.plot(x_vals, y_vals)
            else:
                # Use standard PartialDependenceDisplay for other models
                PartialDependenceDisplay.from_estimator(
                    estimator=model,
                    X=X,
                    features=[feature],
                    target=class_id,
                    kind='average',
                    grid_resolution=50,
                    random_state=42
                )
            
            plt.title(f"{model_name} - {feature}\nClass: {class_names[class_id]}", pad=20)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel("Partial Dependence", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            filename = f"{model_name}_{feature}_Class{class_id}_PDP.png"
            plt.savefig(
                os.path.join(output_dir, filename),
                dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            plt.close()
            
        except Exception as e:
            print(f"Error plotting PDP for {model_name}, feature {feature}, class {class_id}: {str(e)}")

# Train models and generate PDP plots
results = []

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training model: {name}")
    print(f"{'='*50}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy of {name}: {acc:.4f}")
    results.append({'Model': name, 'Accuracy': acc})
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        print(importance.to_string(index=False))
    
    # Check available classes
    available_classes = np.unique(y_train)
    print(f"\nAvailable classes in training data: {available_classes}")
    
    # Generate PDP plots for each available class
    for class_id in class_names:
        if class_id in available_classes:
            print(f"\nGenerating PDP plots for Class {class_id} ({class_names[class_id]})...")
            plot_pdp_compatible(model, name, X_train, features, class_id, class_names)
        else:
            print(f"\nClass {class_id} not found in training data. Skipping.")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\n" + "="*50)
print("Processing complete!")
print(f"Results saved to: model_results.csv")
print(f"PDP plots saved to: {output_dir}/")
print("="*50)

import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv('/kaggle/input/air-quality-health-impact-data/air_quality_health_impact_data.csv')

# 2. Select features and target
feature_cols = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2']
target_col = 'HealthImpactClass'
X = df[feature_cols]
y = df[target_col]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train XGBoost model
model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 5. Sampling for SHAP analysis
sample_size = min(500, len(X_test))  # Use 500 samples or fewer if data is smaller
X_sample = X_test.iloc[:sample_size].copy()
y_sample = y_test.iloc[:sample_size].copy()

# 6. Compute SHAP values using TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# 7. Class-wise SHAP analysis
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
shap_values_array = np.array(shap_values.values)
print(f"SHAP values shape: {shap_values_array.shape}")

# Loop through each class and generate visualizations
for class_idx in range(5):
    try:
        class_shap = shap_values_array[:, :, class_idx]

        # Validate shape
        assert X_sample.shape == class_shap.shape, "Shape mismatch between features and SHAP values"

        print(f"\nSHAP Analysis for {class_names[class_idx]}:")

        # Summary plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(class_shap, X_sample, show=False)
        plt.title(f"SHAP Value Distribution - {class_names[class_idx]}", fontsize=14)
        plt.tight_layout()
        plt.show()

        # Feature importance bar chart
        mean_shap = np.abs(class_shap).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), mean_shap[sorted_idx], 
                 color=plt.cm.viridis(mean_shap[sorted_idx]/mean_shap.max()))
        plt.yticks(range(len(sorted_idx)), X_sample.columns[sorted_idx])
        plt.title(f"Feature Importance - {class_names[class_idx]}", fontsize=14)
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error analyzing {class_names[class_idx]}: {str(e)}")
