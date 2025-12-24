import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

print("Starting model training process...")


df = pd.read_csv('beneficiary_credit_scoring_dataset.csv')

features = [
    'age', 'loan_amount', 'loan_tenure_months', 'number_of_loans',
    'on_time_repayments_percentage', 'missed_payments_total',
    'avg_monthly_electricity_units', 'avg_monthly_mobile_recharge',
    'business_activity'
]
target = 'loan_default_status'
X = df[features]
y = df[target]


categorical_features = ['business_activity']
numerical_features = [col for col in features if col not in categorical_features]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split. Before SMOTE, training set has {sum(y_train==1)} defaults and {sum(y_train==0)} non-defaults.")



print("Preprocessing the training data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test) 

print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
print(f"After SMOTE, training set has {sum(y_train_resampled==1)} defaults and {sum(y_train_resampled==0)} non-defaults.")

print("Training the XGBoost model on the balanced data...")
classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)
print("Model training complete.")

print("\n--- Model Evaluation ---")
y_pred = classifier.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test_processed)[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])


model_filename = 'cibil_model_A.joblib'
preprocessor_filename = 'preprocessor.joblib'

joblib.dump(classifier, model_filename)
joblib.dump(preprocessor, preprocessor_filename)

print(f"\nModel saved successfully as '{model_filename}'")
print(f"Preprocessor saved successfully as '{preprocessor_filename}'")