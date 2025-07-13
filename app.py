import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie

# --------------------
# PAGE CONFIGURATION
# --------------------
st.set_page_config(
    page_title="Pediatric RADT Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------
# LOTTIE ANIMATION
# --------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation_url = "https://lottie.host/b0422998-8959-4676-9657-68b25184293f/c2k45Cvg4s.json"
lottie_anim = load_lottieurl(lottie_animation_url)

# --------------------
# MODEL LOADING
# --------------------
MODEL_PATH = "radt_prediction_model.pkl"
OPTIMAL_THRESHOLD = 0.3474

@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except FileNotFoundError:
        st.error(f"Model file not found. Ensure '{MODEL_PATH}' is in the repository.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

pipeline = load_model()
if pipeline:
    feature_names = pipeline.named_steps['preprocessor'].transformers_[0][2]

# --------------------
# HEADER & ATTRIBUTION
# --------------------
st.image("header_image.jpeg")
st.title("🩺 AI-Powered Pediatric Pharyngitis (RADT) Predictor")

st.markdown("""
    <div style="text-align: center; font-size: 1.1em; margin-bottom: 20px;">
        <p>A sophisticated clinical decision support tool developed to aid in the diagnosis of Group A Streptococcus.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Creator attribution using columns for a clean layout
col_creator1, col_creator2 = st.columns(2)
with col_creator1:
    st.subheader("Lead Project Architect & ML Developer")
    st.markdown("""
        **Dr. Shashank Neupane**  
        [View on LinkedIn ](https://www.linkedin.com/in/shashankneupane131?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
    """, unsafe_allow_html=True)

with col_creator2:
    st.subheader("Clinical Data & Model Fine-Tuning Lead")
    st.markdown("""
        **Dr. Prasamsa Pudasaini**  
        [View on LinkedIn ](https://www.linkedin.com/in/prasamsapudasaini77?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
    """, unsafe_allow_html=True)

st.markdown("---")

# --------------------
# DISCLAIMER
# --------------------
st.warning(
    """
    **Disclaimer:** This is an AI-powered predictive model, not a diagnostic test. Its purpose is to *aid* clinical decision-making by estimating the probability of a positive RADT result. It should **never** be used as the sole basis for diagnosis or treatment. All final medical decisions must be made by a qualified healthcare professional.
    """,
    icon="⚠️"
)


# --------------------
# TABS FOR ORGANIZATION
# --------------------
tab1, tab2, tab3 = st.tabs(["**Prediction Tool**", "**How This Model Works**", "**Data & References**"])

# =================================================================================================
# TAB 1: PREDICTION TOOL
# =================================================================================================
with tab1:
    col_form, col_anim = st.columns([2, 1])

    with col_form:
        st.header("Enter Patient's Clinical Signs & Symptoms")
        with st.form("prediction_form"):
            user_inputs = {}
            c1, c2, c3 = st.columns(3)
            with c1:
                user_inputs['age_y'] = st.number_input("Age (Years)", 1, 18, 8, 1)
                user_inputs['pain'] = 1 if st.selectbox("Pain", ["No", "Yes"]) == "Yes" else 0
                user_inputs['swollenadp'] = 1 if st.selectbox("Swollen Adenopathy", ["No", "Yes"]) == "Yes" else 0
                user_inputs['tender'] = 1 if st.selectbox("Tender Adenopathy", ["No", "Yes"]) == "Yes" else 0
            with c2:
                user_inputs['tonsillarswelling'] = 1 if st.selectbox("Tonsillar Swelling", ["No", "Yes"]) == "Yes" else 0
                user_inputs['exudate'] = 1 if st.selectbox("Tonsillar Exudate", ["No", "Yes"]) == "Yes" else 0
                user_inputs['temperature'] = st.number_input("Temperature (°C)", 36.0, 41.0, 38.0, 0.1)
                user_inputs['sudden'] = 1 if st.selectbox("Sudden Onset", ["No", "Yes"]) == "Yes" else 0
            with c3:
                user_inputs['cough'] = 1 if st.selectbox("Cough Present", ["Yes", "No"]) == "Yes" else 0
                user_inputs['rhinorrhea'] = 1 if st.selectbox("Rhinorrhea", ["No", "Yes"]) == "Yes" else 0
                user_inputs['conjunctivitis'] = 1 if st.selectbox("Conjunctivitis", ["No", "Yes"]) == "Yes" else 0
                user_inputs['headache'] = 1 if st.selectbox("Headache", ["No", "Yes"]) == "Yes" else 0

            # Additional features for a complete form
            c4, c5, c6 = st.columns(3)
            with c4:
                user_inputs['erythema'] = 1 if st.selectbox("Pharyngeal Erythema", ["No", "Yes"]) == "Yes" else 0
                user_inputs['petechiae'] = 1 if st.selectbox("Palatal Petechiae", ["No", "Yes"]) == "Yes" else 0
            with c5:
                user_inputs['abdopain'] = 1 if st.selectbox("Abdominal Pain", ["No", "Yes"]) == "Yes" else 0
                user_inputs['diarrhea'] = 1 if st.selectbox("Diarrhea", ["No", "Yes"]) == "Yes" else 0
            with c6:
                user_inputs['nauseavomit'] = 1 if st.selectbox("Nausea/Vomiting", ["No", "Yes"]) == "Yes" else 0
                user_inputs['scarlet'] = 1 if st.selectbox("Scarlatiniform Rash", ["No", "Yes"]) == "Yes" else 0

            submitted = st.form_submit_button("Analyze Patient Data & Predict RADT Result", type="primary", use_container_width=True)

    with col_anim:
        if lottie_anim:
            st_lottie(lottie_anim, speed=1, height=400, key="initial")

    if submitted:
        if pipeline:
            input_df = pd.DataFrame([user_inputs], columns=feature_names)
            probability = pipeline.predict_proba(input_df)[0, 1]
            prediction = "Positive" if probability >= OPTIMAL_THRESHOLD else "Negative"

            st.subheader("Prediction Result & Clinical Interpretation")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                if prediction == "Positive":
                    st.metric("RADT Positivity Probability", f"{probability:.1%}", "High Risk")
                    st.success("**Recommendation: POSITIVE (Consider RADT)**", icon="✅")
                else:
                    st.metric("RADT Positivity Probability", f"{probability:.1%}", "Low Risk")
                    st.error("**Recommendation: NEGATIVE (RADT May Not Be Necessary)**", icon="❌")
            with res_col2:
                with st.expander("**What does this result mean?**", expanded=True):
                    st.markdown(f"""
                        The model calculated a **{probability:.1%}** probability of this patient having a positive RADT.
                        - Our clinical threshold is set at **{OPTIMAL_THRESHOLD:.1%}**.
                        - This model is intentionally tuned for **high sensitivity (93%)**. This means it is designed to miss as few true positive cases as possible.
                        - A "Positive" recommendation means the patient's symptoms strongly align with patterns seen in confirmed RADT-positive cases.
                        - A "Negative" recommendation means the symptom profile is more consistent with non-GAS pharyngitis (e.g., viral causes).
                    """)
        else:
            st.error("Model is not loaded. Cannot perform prediction.")

# =================================================================================================
# TAB 2: HOW THIS MODEL WORKS
# =================================================================================================
with tab2:
    st.header("Our Methodology: From Data to Deployment")
    st.markdown("This tool is the result of a rigorous, end-to-end machine learning workflow designed for clinical relevance and trustworthiness. Here's a step-by-step breakdown of our process:")

    # Using columns for a more structured look
    meth_col1, meth_col2 = st.columns(2)
    with meth_col1:
        st.subheader("1. Data Foundation & Analysis")
        st.markdown("""
        - **Dataset:** The project is founded on a dataset of 676 pediatric pharyngitis cases, originally published by Cohen JF, et al. (2017) and made available on Kaggle.
        - **Objective:** To predict the binary outcome of the Rapid Antigen Detection Test (`radt`) using only recorded clinical symptoms.
        """)

        st.subheader("2. Preprocessing & Preparation")
        st.markdown("""
        - **Data Cleaning:** Missing values were imputed using the median.
        - **Feature Scaling:** Features were standardized using `StandardScaler` to ensure equitable model learning.
        - **Class Balancing:** `SMOTE` (Synthetic Minority Over-sampling Technique) was used to create a balanced training set, preventing bias.
        """)
        
        st.subheader("3. Model Benchmarking & Selection")
        st.markdown("""
        - **Candidate Models:** We benchmarked multiple models including Logistic Regression, Random Forest, and Gradient Boosting variants.
        - **Champion Model:** **XGBoost** was selected for its superior performance in cross-validation (ROC-AUC: 0.746).
        """)

    with meth_col2:
        st.subheader("4. Clinical Threshold Optimization")
        st.markdown("""
        - **The Challenge:** A default 50% threshold is clinically naive. The priority is to maximize sensitivity to avoid missing true cases.
        - **Our Solution:** An optimal decision threshold of **0.3474** was chosen. This achieves **93% sensitivity** on the test set, fulfilling the primary clinical objective.
        """)

        st.subheader("5. Explainability & Trust (SHAP)")
        st.markdown("""
        - **"Black Box" Problem:** To ensure transparency, we employed **SHAP (SHapley Additive exPlanations)**.
        - **Insights:** SHAP allows us to see which symptoms most influence any given prediction, making the model's reasoning clear and trustworthy.
        """)
        
        st.subheader("6. Tools & Technologies Used")
        st.markdown("""
        - **Language:** Python
        - **Core Libraries:** Pandas, Scikit-learn, XGBoost, Imblearn, SHAP
        - **Web Application:** Streamlit
        """)

# =================================================================================================
# TAB 3: DATA & REFERENCES
# =================================================================================================
with tab3:
    st.header("Data Provenance and Academic References")
    
    st.subheader("Primary Data Source")
    st.markdown("""
    The dataset used in this project originates from the following peer-reviewed study. We extend our gratitude to the authors for making their data publicly available for research.
    - **Cohen JF, Cohen R, Bidet P, Elbez A, Levy C, Bossuyt PM, Chalumeau M. (2017).** Efficiency of a clinical prediction model for selective rapid testing in children with pharyngitis. *PLoS One*, 12(2), e0172871. 
    - **DOI:** [https://doi.org/10.1371/journal.pone.0172871](https://doi.org/10.1371/journal.pone.0172871)
    """)

    st.subheader("Dataset Publication on Kaggle")
    st.markdown("""
    The data was compiled and made accessible on the Kaggle platform by the following user, to whom we are also thankful.
    - **Author:** yoshifumimiya (2022)
    - **Title:** *676 cases of pharyngitis in children*
    - **Publisher:** Kaggle
    - **DOI:** [https://doi.org/10.34740/kaggle/ds/2401925](https://doi.org/10.34740/kaggle/ds/2401925)
    """)
    
    st.subheader("Methodological References")
    st.markdown("""
    Our technical approach was guided by foundational work in machine learning and model interpretability.
    - **Lundberg, S. M., & Lee, S. I. (2017).** A unified approach to interpreting model predictions. In *Advances in neural information processing systems* (Vol. 30). (The foundational paper on the SHAP methodology).
    """)
