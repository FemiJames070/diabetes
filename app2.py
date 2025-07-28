from pycaret.classification import *
import streamlit as st
import pandas as pd

# Load the model once when the script starts
try:
    # Ensure the model file 'tuned_rf_diabetes.pkl' is in the same directory as your script
    model = load_model('tuned_rf_diabetes')
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'tuned_rf_diabetes.pkl' is in the same directory.")
    st.stop() # Stop the app if the model can't be loaded

def predict(model, input_df):
    """
    Makes predictions using the loaded model.
    """
    try:
        predictions_df = predict_model(estimator=model, data=input_df)
        # Assuming 'prediction_label' is the column with your final prediction
        prediction = predictions_df['prediction_label'].iloc[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction Error"

def run():
    """
    Defines the Streamlit application layout and logic.
    """
    st.title('Diabetes Prediction App')
    st.write('Enter the patient information to predict diabetes status.')

    # Use st.number_input with appropriate format and initial value for each field

    # Integer inputs
    pregnancies = st.number_input('Pregnancies', min_value=0, value=0, format="%d")
    age = st.number_input('Age', min_value=0, value=0, format="%d")

    # Float inputs
    glucose = st.number_input('Glucose', min_value=0.0, value=0.0, format="%f")
    blood_pressure = st.number_input('Blood Pressure', min_value=0.0, value=0.0, format="%f")
    skin_thickness = st.number_input('Skin Thickness', min_value=0.0, value=0.0, format="%f")
    insulin = st.number_input('Insulin', min_value=0.0, value=0.0, format="%f")
    bmi = st.number_input('BMI', min_value=0.0, value=0.0, format="%f")
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.0, format="%f")


    # Initialize output
    output = ""

    # Create a DataFrame from the inputs
    input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    # Ensure DataFrame is created with a single row
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        st.success(f'The prediction is: {output}') # Use f-string for cleaner formatting

# Call the run function directly for Streamlit to execute
run()