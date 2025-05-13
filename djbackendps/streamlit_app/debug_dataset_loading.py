import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Debug Dataset Loading", layout="wide")

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
st.title("Debug Dataset Loading")

# Create tabs for different disease types
tab1, tab2 = st.tabs(["Diabetes", "Heart Disease"])

# Diabetes tab
with tab1:
    st.header("Diabetes Dataset Loading Debug")
    
    # Debug info
    st.subheader("Debug Information")
    debug_info = st.empty()
    
    # Create dataset button
    if st.button("Create Diabetes Dataset", key="create_diabetes"):
        st.session_state.diabetes_data = create_diabetes_dataset()
        debug_info.success(f"Created diabetes dataset with {len(st.session_state.diabetes_data)} rows")
    
    # Show dataset if available
    if st.session_state.diabetes_data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.diabetes_data.head(10))
        
        # Row selection
        row_index = st.number_input("Select Row", 0, len(st.session_state.diabetes_data)-1, 0, key="diabetes_row")
        
        # Show selected row
        st.subheader("Selected Row Data")
        selected_row = st.session_state.diabetes_data.iloc[row_index]
        st.write(selected_row)
        
        # Load row button
        if st.button("Load Selected Row", key="load_diabetes_row"):
            try:
                # Get row data
                row_dict = selected_row.to_dict()
                
                # Debug output
                st.write("Row Dictionary:")
                st.json(row_dict)
                
                # Process row data
                pregnancies = int(float(row_dict.get('Pregnancies', 0)))
                glucose = float(row_dict.get('Glucose', 120.0))
                blood_pressure = float(row_dict.get('BloodPressure', 70.0))
                skin_thickness = float(row_dict.get('SkinThickness', 20.0))
                insulin = float(row_dict.get('Insulin', 80.0))
                bmi = float(row_dict.get('BMI', 25.0))
                pedigree = float(row_dict.get('DiabetesPedigreeFunction', 0.5))
                age = int(float(row_dict.get('Age', 30)))
                
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
                
                st.success(f"✅ Successfully loaded row {row_index} into input fields")
                
                # Show updated inputs
                st.subheader("Updated Input Fields")
                st.json(st.session_state.diabetes_inputs)
                
                # Force UI update
                st.rerun()
            except Exception as e:
                st.error(f"Error loading row: {e}")
                st.exception(e)
    else:
        st.info("Please create a dataset first")
    
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
    
    # Update button
    if st.button("Update Input Values", key="update_diabetes"):
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
        
        st.success("✅ Input values updated")
        st.rerun()

# Heart Disease tab
with tab2:
    st.header("Heart Disease Dataset Loading Debug")
    
    # Debug info
    st.subheader("Debug Information")
    debug_info = st.empty()
    
    # Create dataset button
    if st.button("Create Heart Disease Dataset", key="create_heart"):
        st.session_state.heart_data = create_heart_dataset()
        debug_info.success(f"Created heart disease dataset with {len(st.session_state.heart_data)} rows")
    
    # Show dataset if available
    if st.session_state.heart_data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.heart_data.head(10))
        
        # Row selection
        row_index = st.number_input("Select Row", 0, len(st.session_state.heart_data)-1, 0, key="heart_row")
        
        # Show selected row
        st.subheader("Selected Row Data")
        selected_row = st.session_state.heart_data.iloc[row_index]
        st.write(selected_row)
        
        # Load row button
        if st.button("Load Selected Row", key="load_heart_row"):
            try:
                # Get row data
                row_dict = selected_row.to_dict()
                
                # Debug output
                st.write("Row Dictionary:")
                st.json(row_dict)
                
                # Process row data
                age = int(row_dict.get('age', 55))
                sex = "Male" if int(row_dict.get('sex', 1)) == 1 else "Female"
                cp = int(row_dict.get('cp', 0))
                trestbps = float(row_dict.get('trestbps', 130.0))
                chol = float(row_dict.get('chol', 200.0))
                fbs = "Yes" if int(row_dict.get('fbs', 0)) == 1 else "No"
                restecg = int(row_dict.get('restecg', 0))
                thalach = float(row_dict.get('thalach', 150.0))
                exang = "Yes" if int(row_dict.get('exang', 0)) == 1 else "No"
                oldpeak = float(row_dict.get('oldpeak', 0.0))
                slope = int(row_dict.get('slope', 0))
                ca = int(row_dict.get('ca', 0))
                thal = int(row_dict.get('thal', 0))
                
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
                
                st.success(f"✅ Successfully loaded row {row_index} into input fields")
                
                # Show updated inputs
                st.subheader("Updated Input Fields")
                st.json(st.session_state.heart_inputs)
                
                # Force UI update
                st.rerun()
            except Exception as e:
                st.error(f"Error loading row: {e}")
                st.exception(e)
    else:
        st.info("Please create a dataset first")
    
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
    
    # Update button
    if st.button("Update Input Values", key="update_heart"):
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
        
        st.success("✅ Input values updated")
        st.rerun()

# Footer
st.markdown("---")
st.write("Debug tool for dataset loading issues")
