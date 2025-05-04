# -*- coding: utf-8 -*-
"""
SurgePricing_XgBoost

Surge Pricing Classification for Sigma Cabs Using Machine Learning

Import Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("/content/sigma_cabs.csv")

# Exploration
print(df.head())
print(df.info())
print(df.isnull().sum())

# Data Pre-processing
# Drop columns with excessive missing values
df.drop(columns=["Var1"], inplace=True)
df.dropna(subset=["Type_of_Cab", "Life_Style_Index", "Confidence_Life_Style_Index", "Customer_Since_Months"], inplace=True)

# Encode Categorical Data
le = LabelEncoder()
for col in ["Type_of_Cab", "Life_Style_Index", "Confidence_Life_Style_Index", "Destination_Type", "Gender"]:
    df[col] = le.fit_transform(df[col])

# EDA - Exploration and Visualization
sns.set(style="whitegrid")

# Plot the distribution of Surge Pricing Type
plt.figure(figsize=(10, 7))
ax = sns.countplot(data=df, x="Surge_Pricing_Type", palette="viridis")

# Adding title and labels
plt.title("Distribution of Surge Pricing Types", fontsize=16)
plt.xlabel("Surge Pricing Type", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Annotate the bars with the count and percentage
total = len(df)
for p in ax.patches:
    height = p.get_height()
    percentage = (height / total) * 100
    ax.annotate(f'{height}\n({percentage:.1f}%)',
                (p.get_x() + p.get_width() / 2., height),
                ha="center", va="center", fontsize=10, color="black", xytext=(0, 5), textcoords='offset points')

# Show the plot
plt.tight_layout()
plt.show()

# Calculate Surge Pricing Type Distribution
distribution = df['Surge_Pricing_Type'].value_counts()

# Create a table with counts and percentages
distribution_table = pd.DataFrame({
    'Surge_Pricing_Type': distribution.index,
    'Count': distribution.values,
    'Percentage': (distribution.values / len(df)) * 100
})

# Show the distribution table
print(distribution_table)

# Plotting Distribution of Surge Pricing Type
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Surge_Pricing_Type")
plt.title("Surge Pricing Type Distribution")
plt.show()

# Correlation between selected features
corr_df = df[['Trip_Distance', 'Type_of_Cab', 'Surge_Pricing_Type']].copy()
corr_df.rename(columns={
    'Trip_Distance': 'Trip Distance',
    'Type_of_Cab': 'Cab Type (Encoded)',
    'Surge_Pricing_Type': 'Surge Pricing Type'
}, inplace=True)

# Correlation matrix
corr_matrix = corr_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f", square=True, linewidths=0.6, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

# Visualizing Key Numerical Features
key_features = ['Trip_Distance', 'Life_Style_Index', 'Confidence_Life_Style_Index']

plt.figure(figsize=(12, 5))
for i, feature in enumerate(key_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[feature], color='skyblue')
    plt.title(f'Distribution of {feature}', fontsize=12)
    plt.ylabel('')
    plt.grid(True, axis='y')

plt.suptitle("Distribution of Key Numerical Features", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Feature and Target Selection
X = df.drop(columns=["Trip_ID", "Surge_Pricing_Type"])
y = df["Surge_Pricing_Type"]

# Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjust y labels for XGBoost
y_train_adj = y_train - 1
y_test_adj = y_test - 1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression Pipeline
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

logistic_param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__solver": ["lbfgs", "liblinear"]
}

logistic_search = GridSearchCV(logistic_pipeline, logistic_param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
logistic_search.fit(X_train, y_train)
logistic_pred = logistic_search.predict(X_test)

print("\nBest Logistic Regression Params:", logistic_search.best_params_)
print("Classification Report:\n", classification_report(y_test, logistic_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_pred))

# Random Forest Grid
rf_model = RandomForestClassifier(random_state=42)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_search = GridSearchCV(rf_model, rf_param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_pred = rf_search.predict(X_test)

print("\nBest Random Forest Params:", rf_search.best_params_)
print("Classification Report:\n", classification_report(y_test, rf_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# XGBoost Model
xgb_model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)
xgb_param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1, 0.3],
    "max_depth": [3, 5, 7]
}

xgb_search = GridSearchCV(xgb_model, xgb_param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
xgb_search.fit(X_train, y_train_adj)
xgb_pred = xgb_search.predict(X_test)
xgb_pred_labels = xgb_pred + 1  # revert to original classes

print("\nBest XGBoost Params:", xgb_search.best_params_)
print("Classification Report:\n", classification_report(y_test, xgb_pred_labels, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred_labels))

# Feature Importance Visualization
import matplotlib.ticker as mtick
from xgboost import plot_importance

best_xgb_model = xgb_search.best_estimator_

plt.figure(figsize=(12, 8))
plot_importance(best_xgb_model, importance_type='gain', max_num_features=10, height=0.5)
plt.title("Top 10 Feature Importances (by Gain) - XGBoost")
plt.tight_layout()
plt.show()

# Custom Feature Importance Plot
plt.figure(figsize=(16, 10))
ax = plot_importance(
    best_xgb_model,
    importance_type='gain',
    max_num_features=10,
    height=0.7,
    grid=False,
    show_values=True
)

# Format X-axis numbers to 2 decimals
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

plt.title("Top 10 Feature Importances (by Gain) - XGBoost", fontsize=18, fontweight='bold')
plt.xlabel("Average Gain", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Conclusion and Key Observations
"""
1. **Key Dominant Feature**:
   - Type_of_Cab: Has the highest score (174.07), most important feature.
   
2. **Medium Impact Features**:
   - Confidence_Life_Style_Index: Score of 21.81
   - Cancellation_Last_1Month: Score of 13.39
   
3. **Low Impact Features**:
   - Destination_Type: Score of 6.28
   - Trip_Distance: Score of 5.53
   - Life_Style_Index: Score of 4.26
   - Customer_Rating: Score of 4.06
   
4. **Minimal Impact Features**:
   - Var3: Score of 3.33
   - Var2: Score of 2.07
   - Customer_Since_Months: Score of 2.06

Key Observations:
- **Type_of_Cab** dominates the model's prediction.
- Features are grouped into three categories: dominant, moderate, and minor.
- Some features have very little impact on the model (<5).
"""
