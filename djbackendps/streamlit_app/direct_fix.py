import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from streamlit_option_menu import option_menu

# Set page config
st.set_page_config(page_title="Direct Fix - Dataset Loading", layout="wide")

# Initialize session state
if 'diabetes_data' not in st.session_state:
    st.session_state.diabetes_data = None
if 'heart_data' not in st.session_state:
    st.session_state.heart_data = None
if 'diabetes_inputs' not in st.session_state:
    st.session_state.diabetes_inputs = {
        'pregnancies': 0,
        'glucose': 120.0,
        'blood_pressure': 70.0,
        'skin_thickness': 20.0,
        'insulin': 80.0,
        'bmi': 25.0,
        'pedigree': 0.5,
        'age': 30,
        'gender': "Male"
    }
if 'heart_inputs' not in st.session_state:
    st.session_state.heart_inputs = {
        'age': 55,
        'sex': "Male",
        'cp': 0,
        'trestbps': 130.0,
        'chol': 200.0,
        'fbs': "No",
        'restecg': 0,
        'thalach': 150.0,
        'exang': "No",
        'oldpeak': 0.0,
        'slope': 0,
        'ca': 0,
        'thal': 0
    }

# Create sample datasets
def create_diabetes_dataset():
    """Create a sample diabetes dataset"""
    np.random.seed(42)
    n_samples = 20
    
    data = {
        'Pregnancies': np.random.randint(0, 10, n_samples),
        'Glucose': np.random.randint(70, 200, n_samples),
        'BloodPressure': np.random.randint(50, 130, n_samples),
        'SkinThickness': np.random.randint(10, 50, n_samples),
        'Insulin': np.random.randint(0, 200, n_samples),
        'BMI': np.random.uniform(18, 40, n_samples).round(1),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 1.5, n_samples).round(3),
        'Age': np.random.randint(20, 70, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Make sure females have pregnancies and males don't
    for i in range(n_samples):
        if df.loc[i, 'Outcome'] == 1:
            df.loc[i, 'Glucose'] = np.random.randint(140, 200)
            df.loc[i, 'BMI'] = np.random.uniform(30, 40).round(1)
    
    # Set pregnancies to 0 for half the samples (representing males)
    for i in range(n_samples // 2):
        df.loc[i, 'Pregnancies'] = 0
    
    return df

def create_heart_dataset():
    """Create a sample heart disease dataset"""
    np.random.seed(42)
    n_samples = 20
    
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(100, 180, n_samples),
        'chol': np.random.randint(150, 300, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(100, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 4, n_samples).round(1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    return df

# Main app
st.title("Direct Fix - Dataset Loading")

# Create sidebar menu
selected = option_menu(
    "Disease Prediction System",
    ["Diabetes Prediction", "Heart Disease Prediction"],
    icons=["activity", "heart"],
    menu_icon="hospital-fill",
    default_index=0,
    orientation="horizontal"
)

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.header("Diabetes Dataset Loading")
    
    # Create dataset if not already created
    if st.session_state.diabetes_data is None:
        st.session_state.diabetes_data = create_diabetes_dataset()
        st.success("Created diabetes dataset")
    
    # Show dataset
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.diabetes_data.head(10))
    
    # Row selection
    total_samples = len(st.session_state.diabetes_data)
    selected_row = st.number_input("Select Row", 0, total_samples - 1, 0, key="diabetes_row")
    
    # Load row button
    if st.button("Load Selected Row", key="load_diabetes_row"):
        try:
            # Get row data
            row = st.session_state.diabetes_data.iloc[selected_row]
            sample = row.to_dict()
            
            # Process row data
            pregnancies = int(float(sample.get('Pregnancies', 0)))
            glucose = float(sample.get('Glucose', 120.0))
            blood_pressure = float(sample.get('BloodPressure', 70.0))
            skin_thickness = float(sample.get('SkinThickness', 20.0))
            insulin = float(sample.get('Insulin', 80.0))
            bmi = float(sample.get('BMI', 25.0))
            pedigree = float(sample.get('DiabetesPedigreeFunction', 0.5))
            age = int(float(sample.get('Age', 30)))
            
            # Determine gender based on pregnancies
            gender = "Female" if pregnancies > 0 else "Male"
            
            # If male, ensure pregnancies is 0
            if gender == "Male":
                pregnancies = 0
            
            # Update session state
            st.session_state.diabetes_inputs = {
                'pregnancies': pregnancies,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'skin_thickness': skin_thickness,
                'insulin': insulin,
                'bmi': bmi,
                'pedigree': pedigree,
                'age': age,
                'gender': gender
            }
            
            st.success(f"✅ Successfully loaded row {selected_row} into input fields")
            
            # Show updated inputs
            st.subheader("Updated Input Fields")
            st.json(st.session_state.diabetes_inputs)
        except Exception as e:
            st.error(f"Error loading row: {e}")
            st.exception(e)
    
    # Show current input values
    st.subheader("Current Input Values")
    st.json(st.session_state.diabetes_inputs)
    
    # Input fields
    st.subheader("Input Fields")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.diabetes_inputs['gender'] == "Male" else 1)
        
        # Only show pregnancies field if gender is Female
        if gender == "Female":
            pregnancies = st.number_input("Pregnancies", 0, 20, value=st.session_state.diabetes_inputs['pregnancies'])
        else:
            # If male, force pregnancies to 0 and hide the field
            pregnancies = 0
            st.info("Pregnancies field not applicable for males")
    
    with col2:
        glucose = st.number_input("Glucose", 0.0, 300.0, value=float(st.session_state.diabetes_inputs['glucose']))
        blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, value=float(st.session_state.diabetes_inputs['blood_pressure']))
        skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, value=float(st.session_state.diabetes_inputs['skin_thickness']))
    
    with col3:
        insulin = st.number_input("Insulin", 0.0, 500.0, value=float(st.session_state.diabetes_inputs['insulin']))
        bmi = st.number_input("BMI", 0.0, 60.0, value=float(st.session_state.diabetes_inputs['bmi']))
        pedigree = st.number_input("Diabetes Pedigree", 0.0, 2.0, value=float(st.session_state.diabetes_inputs['pedigree']))
        age = st.number_input("Age", 0, 120, value=st.session_state.diabetes_inputs['age'])

# Heart Disease Prediction
elif selected == "Heart Disease Prediction":
    st.header("Heart Disease Dataset Loading")
    
    # Create dataset if not already created
    if st.session_state.heart_data is None:
        st.session_state.heart_data = create_heart_dataset()
        st.success("Created heart disease dataset")
    
    # Show dataset
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.heart_data.head(10))
    
    # Row selection
    total_samples = len(st.session_state.heart_data)
    selected_row = st.number_input("Select Row", 0, total_samples - 1, 0, key="heart_row")
    
    # Load row button
    if st.button("Load Selected Row", key="load_heart_row"):
        try:
            # Get row data
            row = st.session_state.heart_data.iloc[selected_row]
            sample = row.to_dict()
            
            # Process row data
            age = int(sample.get('age', 55))
            sex = "Male" if int(sample.get('sex', 1)) == 1 else "Female"
            cp = int(sample.get('cp', 0))
            trestbps = float(sample.get('trestbps', 130.0))
            chol = float(sample.get('chol', 200.0))
            fbs = "Yes" if int(sample.get('fbs', 0)) == 1 else "No"
            restecg = int(sample.get('restecg', 0))
            thalach = float(sample.get('thalach', 150.0))
            exang = "Yes" if int(sample.get('exang', 0)) == 1 else "No"
            oldpeak = float(sample.get('oldpeak', 0.0))
            slope = int(sample.get('slope', 0))
            ca = int(sample.get('ca', 0))
            thal = int(sample.get('thal', 0))
            
            # Update session state
            st.session_state.heart_inputs = {
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }
            
            st.success(f"✅ Successfully loaded row {selected_row} into input fields")
            
            # Show updated inputs
            st.subheader("Updated Input Fields")
            st.json(st.session_state.heart_inputs)
        except Exception as e:
            st.error(f"Error loading row: {e}")
            st.exception(e)
    
    # Show current input values
    st.subheader("Current Input Values")
    st.json(st.session_state.heart_inputs)
    
    # Input fields
    st.subheader("Input Fields")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 0, 120, value=st.session_state.heart_inputs['age'])
        sex = st.selectbox("Sex", ["Male", "Female"], index=0 if st.session_state.heart_inputs['sex'] == "Male" else 1)
        cp = st.number_input("Chest Pain Type", 0, 3, value=st.session_state.heart_inputs['cp'])
    
    with col2:
        trestbps = st.number_input("Resting Blood Pressure", 0.0, 300.0, value=float(st.session_state.heart_inputs['trestbps']))
        chol = st.number_input("Cholesterol", 0.0, 600.0, value=float(st.session_state.heart_inputs['chol']))
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0 if st.session_state.heart_inputs['fbs'] == "No" else 1)
        restecg = st.number_input("Resting ECG", 0, 2, value=st.session_state.heart_inputs['restecg'])
    
    with col3:
        thalach = st.number_input("Max Heart Rate", 0.0, 300.0, value=float(st.session_state.heart_inputs['thalach']))
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0 if st.session_state.heart_inputs['exang'] == "No" else 1)
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, value=float(st.session_state.heart_inputs['oldpeak']))
        slope = st.number_input("Slope", 0, 2, value=st.session_state.heart_inputs['slope'])
        ca = st.number_input("Number of Major Vessels", 0, 4, value=st.session_state.heart_inputs['ca'])
        thal = st.number_input("Thalassemia", 0, 3, value=st.session_state.heart_inputs['thal'])

# Footer
st.markdown("---")
st.write("Direct fix for dataset loading issues")
