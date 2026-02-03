import streamlit as st
import numpy as np
import pickle
import os

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Blood Donation Prediction",
    page_icon=" ",
    layout="wide"
)

# ------------------- LOAD MODEL -------------------
def load_model():
   MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "random_forest_blood_donation.pkl")

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------- SIDEBAR -------------------
st.sidebar.title("Blood Donation Predictor")
st.sidebar.markdown(
    """
    **Machine Learning Web App**  
    Predict whether a person is likely to donate blood.
    
    **Final Model:** Random Forest  
    **Type:** Classification  
    """
)

st.sidebar.info("Enter donor details carefully")

# ------------------- MAIN UI -------------------
st.title("Blood Donation Prediction System By Jevin Kanani")
st.markdown(
    """
    This advanced web application uses a **Random Forest Classifier**
    to predict whether a person is likely to donate blood.
    """
)

st.divider()

# ------------------- INPUT SECTION -------------------
col1, col2 = st.columns(2)

with col1:
    msld = st.number_input(
        "Months since Last Donation",
        min_value=0,
        max_value=200,
        value=2,
        help="How many months ago the last donation was made"
    )

    nod = st.number_input(
        "Number of Donations",
        min_value=0,
        max_value=100,
        value=10,
        help="Total number of donations made by the person"
    )

with col2:
    tvd = st.number_input(
        "Total Volume Donated (c.c.)",
        min_value=0,
        max_value=50000,
        value=2500,
        step=250,
        help="Total blood donated in cubic centimeters"
    )

    msfd = st.number_input(
        "Months since First Donation",
        min_value=0,
        max_value=300,
        value=24,
        help="How long the person has been donating blood"
    )

st.divider()

# ------------------- PREDICTION -------------------
if st.button("Predict Donation", use_container_width=True):

    input_data = np.array([[msld, nod, tvd, msfd]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("This person is **LIKELY to donate blood**")
    else:
        st.error("This person is **NOT likely to donate blood**")

    st.metric(
        label="Confidence Level",
        value=f"{probability*100:.2f}%"
    )

    st.progress(int(probability * 100))

# ------------------- INFO SECTION -------------------
st.divider()

with st.expander("Model & Project Details"):
    st.markdown(
        """
        **Model Used:** Random Forest Classifier  
        **Why Random Forest?**
        - Stable performance
        - Reduced overfitting
        - Works well with tabular data
        
        **Features Used:**
        - Months since last donation
        - Number of donations
        - Total volume donated
        - Months since first donation
        
        **Output:**
        - 1 → Will donate
        - 0 → Will not donate
        """
    )

st.divider()

st.caption("Developed by Jevin Kanani | Capstone ML Project ")

