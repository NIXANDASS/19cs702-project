import joblib
import pandas as pd
import streamlit as st
import numpy as np
import os 

# --- MODEL FILEPATHS ---
CIBIL_MODEL_FILE = 'cibil_model_A.joblib'
PREPROCESSOR_FILE = 'preprocessor.joblib'
INCOME_MODEL_FILE = 'income_model.joblib'

# Initialize global variables for the models
preprocessor = None
credit_model = None
income_model = None
MODEL_SOURCE = "ML Model Prediction"

# --- MODEL LOADING ---
@st.cache_resource
def load_ml_components():
    """
    Loads all necessary joblib files for prediction.
    Uses try/except to catch common errors (like FileNotFoundError or dependency mismatch)
    and falls back to a static score of 300.
    """
    global preprocessor, credit_model, income_model, MODEL_SOURCE 
    
    current_dir = os.getcwd()
    
    # Print directory path to console for easy debugging
    print("-" * 50)
    print(f"DEBUG: Attempting to load models from current directory: {current_dir}")
    print("-" * 50)

    try:
        # Attempt to load all three components
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        credit_model = joblib.load(CIBIL_MODEL_FILE)
        income_model = joblib.load(INCOME_MODEL_FILE)
        
        st.sidebar.success("✅ Loaded all 3 ML components.")
        MODEL_SOURCE = "ML Model Prediction (CIBIL A)"
        return True
        
    except FileNotFoundError as e:
        # Handles the case where the file is missing from the directory
        missing_file = e.filename
        error_msg = (
            f"❌ FATAL ERROR: Model file '{missing_file}' not found. "
            f"Please place all three files into the directory: `{current_dir}`"
        )
        st.sidebar.markdown(error_msg, unsafe_allow_html=True) 
        
        # --- FALLBACK DEFINITION ---
        def static_score_fallback(features=None): return [300] 
        class MockPreprocessor:
            def transform(self, df): return np.array([[300]]) 
                
        credit_model = static_score_fallback
        preprocessor = MockPreprocessor()
        income_model = static_score_fallback
        MODEL_SOURCE = "Static Fallback Score (Model Missing)"
        return False
        
    except Exception as e:
        # Handles the case where files are found but cannot be loaded 
        # (i.e., dependency mismatch or file corruption)
        error_type = type(e).__name__
        
        st.sidebar.error(
            f"❌ Model Loading Failed: **{error_type}**.\n\n"
            f"The model files were found but could not be loaded. "
            f"This is often a dependency mismatch (scikit-learn/xgboost version)."
        )
        
        # Print the detailed error to the console for debugging
        print("\n" + "="*80)
        print(f"MODEL LOAD EXCEPTION (Check Dependencies): {error_type} - {e}")
        print("="*80 + "\n")
        
        # --- FALLBACK DEFINITION ---
        def static_score_fallback(features=None): return [300] 
        class MockPreprocessor:
            def transform(self, df): return np.array([[300]]) 
        
        credit_model = static_score_fallback
        preprocessor = MockPreprocessor()
        income_model = static_score_fallback
        MODEL_SOURCE = "Static Fallback Score (Load Error)"
        return False

# Execute the loading function to initialize models
load_ml_components()


def predict_score(age, income, loan_amount, loan_tenure, repayment_history, credit_limit, debt, consumption_data):
    """
    Predicts the credit score using the loaded ML model components (or fallback).
    """
    
    # --- 1. Map Streamlit Inputs to Model Features ---
    total_repayments = loan_tenure 
    on_time_count = total_repayments * (0.9 if repayment_history == "Good" else 0.7 if repayment_history == "Average" else 0.5)
    on_time_pct = (on_time_count / total_repayments) if total_repayments > 0 else 0.0

    simulated_features = {
        'business_activity': 'Retail', 
        'number_of_loans': 1,          
        'missed_payments_total': int(total_repayments - on_time_count),
    }

    consumption_defaults = {
        'avg_monthly_mobile_recharge': income * 0.02,
        'avg_monthly_electricity_units': income / 500,
    }
    
    # --- 2. Construct the FINAL Feature DataFrame ---
    feature_data = {
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'loan_tenure_months': loan_tenure,
        'on_time_repayments_percentage': on_time_pct,
        'missed_payments_total': simulated_features['missed_payments_total'],
        'number_of_loans': simulated_features['number_of_loans'],
        'business_activity': simulated_features['business_activity'], 
        'avg_monthly_mobile_recharge': consumption_defaults['avg_monthly_mobile_recharge'],
        'avg_monthly_electricity_units': consumption_defaults['avg_monthly_electricity_units'],
    }
    final_features_df = pd.DataFrame([feature_data])
    
    # --- 3. Preprocessing and Prediction ---
    
    credit_score = 300 # Default score
    
    if not MODEL_SOURCE.startswith("Static Fallback Score"):
        try:
            # Note: We skip the income_model for simplicity in the prediction function, 
            # focusing on the main credit model pipeline
            processed_features = preprocessor.transform(final_features_df) 
            
            # --- CRITICAL CHANGE: Use predict_proba for continuous score ---
            # XGBoost predicts the probability of class 1 (default)
            # We want the probability of class 0 (non-default), which is column [:, 0]
            probability_non_default = credit_model.predict_proba(processed_features)[:, 0][0]
            
            # Map probability (0.0 to 1.0) to score range (300 to 850)
            # Score = Min Score + (Probability * Score Range)
            MIN_SCORE = 300
            MAX_SCORE = 850
            SCORE_RANGE = MAX_SCORE - MIN_SCORE
            
            # Scale the probability linearly to the score range
            credit_score = MIN_SCORE + (probability_non_default * SCORE_RANGE)
            credit_score = int(round(credit_score))
            
        except Exception as e:
            st.error(f"ML Prediction Failure (Post-Feature Construction): {e}. Returning default score.")
            credit_score = 300
        
    # --- 4. Final Score and Breakdown ---
    
    final_score = min(max(int(credit_score), 300), 850)
    
    breakdown = {
        # Explicitly convert all dynamic values to strings for safe serialization
        'Score Source': MODEL_SOURCE,
        'Raw Income (RS)': str(income),
        'On-Time Repayment %': f"{on_time_pct * 100:.1f}%",
        'Simulated Missed Payments': str(simulated_features['missed_payments_total']),
        'Credit Utilization Ratio (CUR)': f"{(debt / credit_limit) * 100:.2f}%" if credit_limit > 0 else "0.00%",
    }
    
    return final_score, breakdown