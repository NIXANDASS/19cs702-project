# train_income_model.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("Starting model training...")

# --- 1. Load the synthetic dataset you created ---
try:
    df = pd.read_csv('synthetic_household_data_2000_no_land.csv')
    print("Dataset 'synthetic_household_data_2000_no_land.csv' loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'synthetic_household_data_2000_no_land.csv' not found.")
    print("Please run your data generation script first before running this.")
    exit()

# --- 2. Select the features for clustering ---
# These features will be used to group beneficiaries into income/stability proxies.
features_to_cluster = [
    'avg_monthly_utility_spend',
    'household_deprivation_score',
    'asset_ownership_count',
    'household_size_to_earners_ratio',
    'utility_payment_regularity',
    'housing_type',
    'primary_income_source',
    'mobile_recharge_pattern'
]

X = df[features_to_cluster]
print("Features for clustering selected.")

# --- 3. Create a preprocessing pipeline ---
# This is crucial for handling mixed data types (numbers and categories).
numeric_features = [
    'avg_monthly_utility_spend',
    'household_deprivation_score',
    'asset_ownership_count',
    'household_size_to_earners_ratio'
]
categorical_features = [
    'utility_payment_regularity',
    'housing_type',
    'primary_income_source',
    'mobile_recharge_pattern'
]

# Create a preprocessor object that scales numeric data and one-hot encodes categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
print("Preprocessing pipeline created.")

# --- 4. Define the final model pipeline ---
# This pipeline chains the preprocessing steps with the clustering algorithm.
# This ensures that any new data you predict on later will be treated the same way.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clusterer', KMeans(n_clusters=3, random_state=42, n_init=10)) # We'll create 3 clusters
])
print("K-Means model pipeline defined.")

# --- 5. Train the model ---
print("Training the clustering model on the data...")
model_pipeline.fit(X)
print("Model training complete.")

# --- 6. Save the entire trained pipeline ---
joblib.dump(model_pipeline, 'income_model.joblib')

print("\n✅ Success! Your second model has been trained and saved as 'income_model.joblib'.")