import streamlit as st
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")


# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to find the model file in the same directory
        model_path = Path(__file__).parent / "random_forest_model.pkl"
        
        # Alternative path for Streamlit sharing
        if not model_path.exists():
            model_path = Path("random_forest_model.pkl")
            
        if not model_path.exists():
            st.error("Model file not found! Please ensure 'random_forest_model.pkl' is in the same directory.")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()


def predict_heart_disease(features):
    """
    Predicts heart disease based on input features.
    :param features: A list or array of features for the prediction.
    :return: The prediction (0 or 1).
    """
    if model is None:
        raise Exception("Model not loaded properly")
    features = np.array(features).reshape(1, -1)  # Reshape for a single sample
    prediction = model.predict(features)
    return prediction[0]


# Initialize session state for form clearing
if 'clear_form' not in st.session_state:
    st.session_state.clear_form = False

# Streamlit app
st.title("Heart Disease Prediction App")

# Only show the form if model loaded successfully
if model is not None:
    with st.form("prediction_form"):
        st.header("Patient Information")

        # Clear form button
        if st.form_submit_button("Clear Form", help="Click to reset all fields"):
            st.session_state.clear_form = True
            st.rerun()  # Changed from experimental_rerun() to rerun()

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120,
                                  value=None if st.session_state.clear_form else None)
            sex = st.selectbox("Sex", options=[("Select", None), ("Male", 1), ("Female", 0)],
                               format_func=lambda x: x[0], index=0)
            chest_pain_type = st.selectbox("Chest Pain Type", options=[
                ("Select", None), ("Typical angina", 1), ("Atypical angina", 2),
                ("Non-anginal pain", 3), ("Asymptomatic", 4)], format_func=lambda x: x[0], index=0)
            resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200,
                                                     value=None)
            cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=None)
            fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                               options=[("Select", None), ("No", 0), ("Yes", 1)],
                                               format_func=lambda x: x[0], index=0)

        with col2:
            resting_electrocardiographic_results = st.selectbox("Resting Electrocardiographic Results",
                                                                options=[("Select", None), ("Normal", 0),
                                                                         ("ST-T wave abnormality", 1),
                                                                         ("Probable or definite left ventricular hypertrophy",
                                                                          2)], format_func=lambda x: x[0], index=0)
            max_heart_rate_achieved = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220,
                                                      value=None)
            exercise_induced_angina = st.selectbox("Exercise Induced Angina",
                                                   options=[("Select", None), ("No", 0), ("Yes", 1)],
                                                   format_func=lambda x: x[0], index=0)
            st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0,
                                            value=None, step=0.1)
            st_slope = st.selectbox("Slope of Peak Exercise ST Segment",
                                    options=[("Select", None), ("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)],
                                    format_func=lambda x: x[0], index=0)
            number_of_major_vessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy",
                                                   options=[None, 0, 1, 2, 3], index=0)
            thalassemia = st.selectbox("Thalassemia",
                                       options=[("Select", None), ("Normal", 3), ("Fixed defect", 6),
                                                ("Reversible defect", 7)],
                                       format_func=lambda x: x[0], index=0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                # Reset clear form state
                st.session_state.clear_form = False

                # Prepare features
                features = [
                    age if age is not None else 0,
                    sex[1] if sex and sex[1] is not None else 0,
                    chest_pain_type[1] if chest_pain_type and chest_pain_type[1] is not None else 0,
                    resting_blood_pressure if resting_blood_pressure is not None else 0,
                    cholesterol if cholesterol is not None else 0,
                    fasting_blood_sugar[1] if fasting_blood_sugar and fasting_blood_sugar[1] is not None else 0,
                    resting_electrocardiographic_results[1] if resting_electrocardiographic_results and
                                                               resting_electrocardiographic_results[
                                                                   1] is not None else 0,
                    max_heart_rate_achieved if max_heart_rate_achieved is not None else 0,
                    exercise_induced_angina[1] if exercise_induced_angina and exercise_induced_angina[
                        1] is not None else 0,
                    st_depression if st_depression is not None else 0.0,
                    st_slope[1] if st_slope and st_slope[1] is not None else 0,
                    number_of_major_vessels if number_of_major_vessels is not None else 0,
                    thalassemia[1] if thalassemia and thalassemia[1] is not None else 0
                ]

                if None in features:
                    st.warning("Please fill in all fields before prediction")
                else:
                    prediction = predict_heart_disease(features)
                    if prediction == 1:
                        st.error("⚠️ The model predicts heart disease. Please consult a healthcare professional.")
                        st.markdown("""
                        **Next Steps:**
                        - Schedule an appointment with your doctor
                        - Consider lifestyle changes (diet, exercise)
                        - Monitor your symptoms
                        """)
                    else:
                        st.success("✅ The model predicts no heart disease.")
                        st.markdown("""
                        **Prevention Tips:**
                        - Maintain a healthy diet
                        - Exercise regularly
                        - Get regular check-ups
                        """)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
else:
    st.warning("The app cannot make predictions without the model. Please ensure the model file is available.")
