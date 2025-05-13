import pandas as pd
import numpy as np
import os
import streamlit as st

# Create a reliable sample dataset
def create_diabetes_dataset():
    """Create a reliable diabetes dataset"""
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
    """Create a reliable heart disease dataset"""
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

# Function to ensure datasets are loaded
def ensure_datasets_loaded():
    """Make sure datasets are loaded in session state"""
    if 'diabetes_data' not in st.session_state or st.session_state.diabetes_data is None:
        st.session_state.diabetes_data = create_diabetes_dataset()
    
    if 'heart_data' not in st.session_state or st.session_state.heart_data is None:
        st.session_state.heart_data = create_heart_dataset()

# Function to get a sample from dataset
def get_sample(disease_type, index=0):
    """Get a reliable sample from dataset"""
    ensure_datasets_loaded()
    
    if disease_type == 'diabetes':
        df = st.session_state.diabetes_data
        default_sample = {
            'Pregnancies': 2,
            'Glucose': 140.0,
            'BloodPressure': 80.0,
            'SkinThickness': 25.0,
            'Insulin': 120.0,
            'BMI': 28.5,
            'DiabetesPedigreeFunction': 0.45,
            'Age': 35,
            'Outcome': 0
        }
    elif disease_type == 'heart':
        df = st.session_state.heart_data
        default_sample = {
            'age': 55,
            'sex': 1,
            'cp': 2,
            'trestbps': 140.0,
            'chol': 220.0,
            'fbs': 0,
            'restecg': 1,
            'thalach': 160.0,
            'exang': 0,
            'oldpeak': 1.5,
            'slope': 1,
            'ca': 1,
            'thal': 2,
            'target': 0
        }
    else:
        return None
    
    try:
        if df is not None and index >= 0 and index < len(df):
            row = df.iloc[index]
            return row.to_dict()
        else:
            return default_sample
    except:
        return default_sample

# Function to process diabetes sample
def process_diabetes_sample(sample):
    """Process a diabetes sample to ensure it's valid for input"""
    try:
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
        
        return {
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
    except Exception as e:
        print(f"Error processing diabetes sample: {e}")
        return {
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

# Function to process heart sample
def process_heart_sample(sample):
    """Process a heart disease sample to ensure it's valid for input"""
    try:
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
        
        return {
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
    except Exception as e:
        print(f"Error processing heart sample: {e}")
        return {
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
