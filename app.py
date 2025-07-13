import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "radt_prediction_model.pkl"
OPTIMAL_THRESHOLD = 0.3474 # From our analysis in Step 5

# --- LOAD MODEL ---
# Use a try-except block to handle missing model file
try:
    pipeline = joblib.load(MODEL_PATH)
    # Get the feature names from the pipeline's preprocessor step
    feature_names = pipeline.named_steps['preprocessor'].transformers_[0][2]
except FileNotFoundError:
    st.error(f"Model file not found. Please ensure '{MODEL_PATH}' is in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.stop()


# --- APP LAYOUT ---
st.set_page_config(page_title="Pediatric RADT Predictor", layout="wide")
st.title("Pediatric Pharyngitis: RADT Positivity Prediction Tool")
st.markdown("""
This tool uses an **explainable XGBoost model** to predict the probability of a positive Rapid Antigen Detection Test (RADT) for Group A Streptococcus based on clinical signs and symptoms.
**Disclaimer:** This is a clinical decision support tool and does not replace professional medical judgment.
""")

st.header("Enter Patient's Clinical Signs & Symptoms")

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

# Dictionary to hold user inputs
user_inputs = {}

with col1:
    user_inputs['age_y'] = st.number_input("Age (Years)", min_value=1, max_value=18, value=8, step=1)
    user_inputs['pain'] = 1 if st.selectbox("Pain (e.g., throat pain)", ["No", "Yes"]) == "Yes" else 0
    user_inputs['swollenadp'] = 1 if st.selectbox("Swollen Adenopathy", ["No", "Yes"]) == "Yes" else 0
    user_inputs['tender'] = 1 if st.selectbox("Tender Adenopathy", ["No", "Yes"]) == "Yes" else 0
    user_inputs['tonsillarswelling'] = 1 if st.selectbox("Tonsillar Swelling", ["No", "Yes"]) == "Yes" else 0
    user_inputs['exudate'] = 1 if st.selectbox("Tonsillar Exudate", ["No", "Yes"]) == "Yes" else 0

with col2:
    user_inputs['temperature'] = st.number_input("Temperature (°C)", min_value=36.0, max_value=41.0, value=38.0, step=0.1)
    user_inputs['sudden'] = 1 if st.selectbox("Sudden Onset", ["No", "Yes"]) == "Yes" else 0
    user_inputs['cough'] = 1 if st.selectbox("Cough Present", ["Yes", "No"]) == "Yes" else 0
    user_inputs['rhinorrhea'] = 1 if st.selectbox("Rhinorrhea (Runny Nose)", ["No", "Yes"]) == "Yes" else 0
    user_inputs['conjunctivitis'] = 1 if st.selectbox("Conjunctivitis", ["No", "Yes"]) == "Yes" else 0
    user_inputs['headache'] = 1 if st.selectbox("Headache", ["No", "Yes"]) == "Yes" else 0

with col3:
    user_inputs['erythema'] = 1 if st.selectbox("Pharyngeal Erythema (Redness)", ["No", "Yes"]) == "Yes" else 0
    user_inputs['petechiae'] = 1 if st.selectbox("Palatal Petechiae", ["No", "Yes"]) == "Yes" else 0
    user_inputs['abdopain'] = 1 if st.selectbox("Abdominal Pain", ["No", "Yes"]) == "Yes" else 0
    user_inputs['diarrhea'] = 1 if st.selectbox("Diarrhea", ["No", "Yes"]) == "Yes" else 0
    user_inputs['nauseavomit'] = 1 if st.selectbox("Nausea or Vomiting", ["No", "Yes"]) == "Yes" else 0
    user_inputs['scarlet'] = 1 if st.selectbox("Scarlatiniform Rash", ["No", "Yes"]) == "Yes" else 0

# --- PREDICTION LOGIC ---
if st.button("Predict RADT Result", type="primary"):
    # Create a DataFrame from the user inputs with columns in the correct order
    input_df = pd.DataFrame([user_inputs], columns=feature_names)

    # Predict the probability using the loaded pipeline
    probability = pipeline.predict_proba(input_df)[0, 1]
    
    # Classify based on the optimal threshold
    prediction = "Positive" if probability >= OPTIMAL_THRESHOLD else "Negative"
    
    st.subheader("Prediction Result")
    
    if prediction == "Positive":
        st.metric("RADT Positivity Probability", f"{probability:.1%}", "High Risk")
        st.success(f"**Recommendation: POSITIVE (Consider RADT)**")
    else:
        st.metric("RADT Positivity Probability", f"{probability:.1%}", "Low Risk")
        st.warning(f"**Recommendation: NEGATIVE (RADT May Not Be Necessary)**")
    
    st.info(f"The decision is based on a probability of {probability:.1%} against a clinical threshold of {OPTIMAL_THRESHOLD:.1%}.")
