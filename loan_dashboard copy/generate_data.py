import pandas as pd
import numpy as np
import random

print("Starting synthetic dataset generation...")

# --- Configuration ---
NUM_RECORDS = 2000
FILENAME = 'beneficiary_credit_scoring_dataset.csv'

# --- Data Generation Logic ---
data = []
business_activities = ["Retail Shop", "Handicrafts", "Tailoring", "Farming", "Artisan", "Food Stall"]

for _ in range(NUM_RECORDS):
    # --- Basic Features ---
    age = random.randint(22, 65)
    number_of_loans = random.randint(1, 8)
    loan_amount = random.randint(25000, 200000)
    loan_tenure_months = random.choice([12, 24, 36, 48, 60])
    business_activity = random.choice(business_activities)
    
    # --- Repayment Behavior & Default Logic (Crucial for a realistic model) ---
    # We'll create a 'profile_score' to determine the likelihood of default
    profile_score = 0
    
    # Good profiles have lower missed payments
    if random.random() < 0.7: # 70% of profiles are generally good
        missed_payments_total = random.randint(0, 3)
        on_time_repayments_percentage = round(random.uniform(90.0, 100.0), 2)
        profile_score += 2
    else: # 30% are riskier
        missed_payments_total = random.randint(2, 10)
        on_time_repayments_percentage = round(random.uniform(60.0, 95.0), 2)
        profile_score -= 2
        
    # Older, more experienced borrowers are slightly less risky
    if 35 < age < 55:
        profile_score += 1
    
    # Higher number of previous loans with good history is a positive sign
    if number_of_loans > 4:
        profile_score += 1

    # --- Consumption Metrics (Proxy for income/stability) ---
    avg_monthly_electricity_units = random.randint(50, 400)
    avg_monthly_mobile_recharge = random.randint(150, 800)
    
    # Higher consumption can indicate more stability
    if avg_monthly_electricity_units > 200 and avg_monthly_mobile_recharge > 400:
        profile_score += 1
    
    # --- Determine Final Default Status ---
    # A higher profile_score means a lower chance of default
    default_probability = 0.1 - (profile_score * 0.02) # Base 10% default rate, adjusted by profile
    
    # Ensure probability is within a sensible range [0.01, 0.5]
    default_probability = max(0.01, min(default_probability, 0.5)) 
    
    loan_default_status = 1 if random.random() < default_probability else 0

    data.append([
        age,
        loan_amount,
        loan_tenure_months,
        number_of_loans,
        on_time_repayments_percentage,
        missed_payments_total,
        avg_monthly_electricity_units,
        avg_monthly_mobile_recharge,
        business_activity,
        loan_default_status
    ])

# --- Create DataFrame and Save ---
columns = [
    'age', 'loan_amount', 'loan_tenure_months', 'number_of_loans',
    'on_time_repayments_percentage', 'missed_payments_total',
    'avg_monthly_electricity_units', 'avg_monthly_mobile_recharge',
    'business_activity', 'loan_default_status'
]
df = pd.DataFrame(data, columns=columns)
df.to_csv(FILENAME, index=False)

print(f"Successfully generated '{FILENAME}' with {NUM_RECORDS} records.")