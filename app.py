import streamlit as st
import joblib
import pandas as pd

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Heart Rate Prediction",
    page_icon="❤️",
    layout="centered"
)

# ==========================
# Load Model
# ==========================
rf_model = joblib.load("rf_model.pkl")
Gender_encoder = joblib.load("Gender_encoder.pkl")

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
.main{
    background-color:#f8f9fa;
}
h1{
    color:#d62828;
    text-align:center;
}
.stButton>button{
    width:100%;
    background-color:#d62828;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:3em;
}
.stButton>button:hover{
    background-color:#a4133c;
}
.prediction{
    background:#e9ecef;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
    color:#d62828;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================
st.sidebar.title("About")
st.sidebar.info(
    """
    **Heart Rate Prediction**

    This application predicts the heart rate using a trained
    Random Forest Regression model.

    **Input Features**
    - Gender
    - Age
    """
)

# ==========================
# Main Title
# ==========================
st.title("❤️ Heart Rate Prediction")

st.write(
    "Predict your estimated **Heart Rate** using a Machine Learning model."
)

st.divider()

# ==========================
# Input Section
# ==========================
st.subheader("Enter Details")

gender = st.selectbox(
    "Gender",
    Gender_encoder.classes_
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25,
    step=1
)

st.divider()

# ==========================
# Prediction
# ==========================
if st.button("Predict Heart Rate ❤️"):

    gender_encoded = Gender_encoder.transform([gender])[0]

    input_data = pd.DataFrame({
        "Gender": [gender_encoded],
        "Age": [age]
    })

    prediction = rf_model.predict(input_data)

    st.markdown(
        f"""
        <div class="prediction">
            ❤️ Predicted Heart Rate <br><br>
            {prediction[0]:.2f} BPM
        </div>
        """,
        unsafe_allow_html=True
    )

    if prediction[0] < 60:
        st.info("Heart rate is lower than the typical resting range.")
    elif prediction[0] <= 100:
        st.success("Heart rate is within the typical resting range.")
    else:
        st.warning("Heart rate is higher than the typical resting range.")

st.divider()

st.caption("Developed using Streamlit, Scikit-learn, Pandas, and Joblib.")