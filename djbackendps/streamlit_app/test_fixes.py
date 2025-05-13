import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime

# Import our fix modules
from fix_dataset_loading import ensure_datasets_loaded, get_sample, process_diabetes_sample, process_heart_sample
from fix_session_persistence import initialize_session, check_url_params, login_user, logout_user, authenticate_user
from fix_prediction_type import save_prediction, determine_prediction_type
from fix_gender_pregnancy import validate_diabetes_inputs, handle_gender_pregnancy_ui, render_diabetes_inputs

# Set page config
st.set_page_config(page_title="Disease Prediction System - Test Fixes", layout="wide")

# Initialize session
initialize_session()
check_url_params()

# Initialize datasets
ensure_datasets_loaded()

# Initialize input state with default values
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

# Main app
st.title("Disease Prediction System - Test Fixes")

# Login/logout section
if not st.session_state.session['logged_in']:
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if authenticate_user(username, password):
                st.success(f"Logged in as {username}")
                st.rerun()
            else:
                st.error("Invalid username or password")
else:
    # Show logged in user
    st.write(f"Logged in as: {st.session_state.session['username']} (Role: {st.session_state.session['user_role']})")
    
    # Logout button
    if st.button("Logout"):
        logout_user()
        st.success("Logged out successfully")
        st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Loading", "Gender-Pregnancy", "Prediction Type", "Session Persistence"])
    
    # Tab 1: Dataset Loading
    with tab1:
        st.subheader("Test Dataset Loading")
        
        # Select disease type
        disease_type = st.selectbox("Select Disease Type", ["diabetes", "heart"])
        
        # Load dataset button
        if st.button("Load Dataset"):
            if disease_type == "diabetes":
                df = st.session_state.diabetes_data
            else:
                df = st.session_state.heart_data
                
            st.success(f"Loaded {len(df)} rows from {disease_type} dataset")
            st.dataframe(df.head(10))
            
            # Select row
            row_index = st.number_input("Select Row", 0, len(df)-1, 0)
            
            # Load row button
            if st.button("Load Selected Row"):
                sample = get_sample(disease_type, row_index)
                
                if disease_type == "diabetes":
                    processed = process_diabetes_sample(sample)
                    st.session_state.diabetes_inputs = processed
                else:
                    processed = process_heart_sample(sample)
                    st.session_state.heart_inputs = processed
                
                st.success(f"Loaded row {row_index} successfully")
                st.json(processed)
    
    # Tab 2: Gender-Pregnancy
    with tab2:
        st.subheader("Test Gender-Pregnancy Handling")
        
        # Render diabetes inputs with proper gender-pregnancy handling
        inputs = render_diabetes_inputs()
        
        # Show the validated inputs
        st.subheader("Validated Inputs")
        st.json(inputs)
        
        # Test button
        if st.button("Test Gender Change"):
            # Toggle gender
            current_gender = st.session_state.diabetes_inputs['gender']
            new_gender = "Male" if current_gender == "Female" else "Female"
            
            # Update gender
            st.session_state.diabetes_inputs['gender'] = new_gender
            
            # If changing to male, ensure pregnancies is 0
            if new_gender == "Male":
                st.session_state.diabetes_inputs['pregnancies'] = 0
            
            st.success(f"Changed gender to {new_gender}")
            st.rerun()
    
    # Tab 3: Prediction Type
    with tab3:
        st.subheader("Test Prediction Type")
        
        # Create test records
        diabetes_record = {
            'prediction_data': {
                'pregnancies': 2,
                'glucose': 140.0,
                'blood_pressure': 80.0,
                'skin_thickness': 25.0,
                'insulin': 120.0,
                'bmi': 28.5,
                'pedigree': 0.45,
                'age': 35,
                'gender': "Female"
            },
            'prediction_result': "Positive for Diabetes",
            'risk_score': 0.75,
            'recommendation': "Please consult with a doctor",
            'username': st.session_state.session['username']
        }
        
        heart_record = {
            'prediction_data': {
                'age': 55,
                'sex': "Male",
                'cp': 2,
                'trestbps': 140.0,
                'chol': 220.0,
                'fbs': "No",
                'restecg': 1,
                'thalach': 160.0,
                'exang': "No",
                'oldpeak': 1.5,
                'slope': 1,
                'ca': 1,
                'thal': 2
            },
            'prediction_result': "Positive for Heart Disease",
            'risk_score': 0.65,
            'recommendation': "Please consult with a cardiologist",
            'username': st.session_state.session['username']
        }
        
        # Select record type
        record_type = st.selectbox("Select Record Type", ["diabetes", "heart"])
        
        # Show record
        if record_type == "diabetes":
            st.json(diabetes_record)
        else:
            st.json(heart_record)
        
        # Test prediction type detection
        if st.button("Test Prediction Type Detection"):
            if record_type == "diabetes":
                detected_type = determine_prediction_type(diabetes_record)
                st.success(f"Detected type: {detected_type}")
            else:
                detected_type = determine_prediction_type(heart_record)
                st.success(f"Detected type: {detected_type}")
        
        # Test save prediction
        if st.button("Test Save Prediction"):
            if record_type == "diabetes":
                record_id = save_prediction("diabetes", diabetes_record)
            else:
                record_id = save_prediction("heart", heart_record)
                
            if record_id:
                st.success(f"Saved prediction with ID: {record_id}")
            else:
                st.error("Failed to save prediction")
    
    # Tab 4: Session Persistence
    with tab4:
        st.subheader("Test Session Persistence")
        
        # Show session state
        st.write("Session State:")
        st.json({
            'session_id': st.session_state.session_id,
            'username': st.session_state.session['username'],
            'logged_in': st.session_state.session['logged_in'],
            'user_role': st.session_state.session['user_role'],
            'login_time': str(st.session_state.session['login_time']),
            'last_activity': str(st.session_state.session['last_activity'])
        })
        
        # Show URL parameters
        st.write("URL Parameters:")
        st.json(dict(st.query_params))
        
        # Test URL with session ID
        session_url = f"http://localhost:8501/?session_id={st.session_state.session_id}"
        st.write("Session URL:")
        st.code(session_url)
        
        # Test URL with username
        username_url = f"http://localhost:8501/?username={st.session_state.session['username']}"
        st.write("Username URL:")
        st.code(username_url)
        
        # Test both
        both_url = f"http://localhost:8501/?username={st.session_state.session['username']}&session_id={st.session_state.session_id}"
        st.write("Both Parameters URL:")
        st.code(both_url)
        
        st.info("Open any of these URLs in a new tab to test session persistence")

# Footer
st.markdown("---")
st.write("Test app for fixing issues in Disease Prediction System")
