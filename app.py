import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Streamlit app title with updated size and color
st.markdown("<h1 style='text-align: center; color: #FF5733; font-size: 42px;'>Heart+ (Smart Heart Care)</h1>", unsafe_allow_html=True)

# Helper function to show details in a modal
def show_details(attribute_name, details_text):
    with st.expander(f"Details about {attribute_name}"):
        st.write(details_text)

# Collect user inputs
age = st.number_input('Age', min_value=1, max_value=120, value=25)
if st.button("Details", key='age'):
    show_details(
        "Age",
        "Select your current age. Age can affect heart health, with older adults generally having higher risks."
    )

sex = st.selectbox('Sex', options=['M', 'F'], format_func=lambda x: 'Male' if x == 'M' else 'Female')
if st.button("Details", key='sex'):
    show_details(
        "Sex",
        "Choose your sex. Men and women can experience different risk factors for heart disease."
    )

chest_pain = st.selectbox('Chest Pain Type', options=['TA', 'ATA', 'NAP', 'ASY'], format_func=lambda x: {
    'TA': 'Typical Angina', 'ATA': 'Atypical Angina', 'NAP': 'Non-Anginal Pain', 'ASY': 'Asymptomatic'
}[x])
if st.button("Details", key='chest_pain'):
    show_details(
        "Chest Pain Type",
        "Choose the description that matches your experience: \n"
        "- **Typical Angina (TA)**: Pain triggered by physical activity or stress.\n"
        "- **Atypical Angina (ATA)**: Unusual chest pain not always related to exertion.\n"
        "- **Non-Anginal Pain (NAP)**: Chest pain not associated with the heart.\n"
        "- **Asymptomatic (ASY)**: No chest pain or discomfort."
    )

resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
if st.button("Details", key='resting_bp'):
    show_details(
        "Resting Blood Pressure",
        "Select your usual resting blood pressure reading:\n"
        "- **80-120 mm Hg**: Generally healthy blood pressure.\n"
        "- **121-139 mm Hg**: Slightly elevated; may need to watch salt intake.\n"
        "- **140+ mm Hg**: High blood pressure; consider talking to a doctor."
    )

cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400, value=200)
if st.button("Details", key='cholesterol'):
    show_details(
        "Cholesterol",
        "Select your cholesterol level:\n"
        "- **Less than 200 mg/dL**: Desirable cholesterol level.\n"
        "- **200-239 mg/dL**: Borderline high; try a heart-healthy diet.\n"
        "- **240+ mg/dL**: High; consider making dietary changes and consulting a healthcare provider."
    )

fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
if st.button("Details", key='fasting_bs'):
    show_details(
        "Fasting Blood Sugar",
        "Indicate if your fasting blood sugar level is above 120 mg/dL:\n"
        "- **Yes**: Blood sugar is higher than normal; monitor carbohydrate intake.\n"
        "- **No**: Blood sugar is within a normal range."
    )

resting_ecg = st.selectbox('Resting ECG', options=['Normal', 'ST', 'LVH'], format_func=lambda x: {
    'Normal': 'Normal', 'ST': 'ST-T wave abnormality', 'LVH': 'Left ventricular hypertrophy'
}[x])
if st.button("Details", key='resting_ecg'):
    show_details(
        "Resting ECG",
        "Select your resting ECG status:\n"
        "- **Normal**: No issues detected in heart's electrical activity.\n"
        "- **ST-T wave abnormality (ST)**: Possible changes in heart rhythm; consult with a healthcare professional if experiencing symptoms.\n"
        "- **Left Ventricular Hypertrophy (LVH)**: Thickening of heart walls; could be linked to high blood pressure."
    )

max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
if st.button("Details", key='max_hr'):
    show_details(
        "Maximum Heart Rate",
        "Enter the highest heart rate you reach during physical activity:\n"
        "- **60-100 bpm**: Lower than average; consider increasing activity level.\n"
        "- **101-170 bpm**: Within normal range for most people during exercise.\n"
        "- **171-220 bpm**: Higher intensity; ensure you're not overexerting."
    )

exercise_angina = st.selectbox('Exercise Induced Angina', options=['Y', 'N'], format_func=lambda x: 'Yes' if x == 'Y' else 'No')
if st.button("Details", key='exercise_angina'):
    show_details(
        "Exercise Induced Angina",
        "Select whether you experience chest discomfort during exercise:\n"
        "- **Yes**: Chest discomfort during physical activity; you may need to limit strenuous exercises.\n"
        "- **No**: No chest discomfort during exercise."
    )

oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, value=1.0)
if st.button("Details", key='oldpeak'):
    show_details(
        "Oldpeak",
        "Oldpeak indicates changes in the ST segment during exercise:\n"
        "- **0-1 mm**: Minimal change; generally low risk.\n"
        "- **1-3 mm**: Moderate change; pay attention to symptoms during exertion.\n"
        "- **3+ mm**: Higher change; consult a healthcare provider."
    )

st_slope = st.selectbox('ST Slope', options=['Up', 'Flat', 'Down'])
if st.button("Details", key='st_slope'):
    show_details(
        "ST Slope",
        "Choose the description that matches your exercise experience:\n"
        "- **Upsloping**: You feel okay even during intense exercise.\n"
        "- **Flat**: You feel fine with light exercises, but may experience discomfort during high intensity.\n"
        "- **Downsloping**: You feel discomfort even with light exercise or at rest."
    )

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})

# Predict using Random Forest model
prediction = model.predict(input_data)[0]
prediction_prob = model.predict_proba(input_data)[0][1]

# Display the result with updated size and color
if prediction == 1:
    st.markdown("<h2 style='text-align: center; color: red; font-size: 36px;'>Prediction: Positive for Heart Disease</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='text-align: center; color: green; font-size: 36px;'>Prediction: Negative for Heart Disease</h2>", unsafe_allow_html=True)

# Display the probability with increased size
st.markdown(f"<h3 style='text-align: center; font-size: 22px;'>Prediction Probability of having heart disease: {prediction_prob:.2%}</h3>", unsafe_allow_html=True)
