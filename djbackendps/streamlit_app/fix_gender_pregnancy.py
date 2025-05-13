import streamlit as st

def validate_diabetes_inputs(inputs):
    """Validate diabetes inputs and ensure gender-pregnancy consistency"""
    # Make a copy to avoid modifying the original
    validated = inputs.copy()
    
    # Ensure gender is valid
    if validated['gender'] not in ["Male", "Female"]:
        validated['gender'] = "Male"  # Default to male
    
    # If male, ensure pregnancies is 0
    if validated['gender'] == "Male" and validated['pregnancies'] > 0:
        validated['pregnancies'] = 0
    
    # Ensure numeric values are within reasonable ranges
    validated['glucose'] = max(0, min(500, validated['glucose']))
    validated['blood_pressure'] = max(0, min(300, validated['blood_pressure']))
    validated['skin_thickness'] = max(0, min(100, validated['skin_thickness']))
    validated['insulin'] = max(0, min(1000, validated['insulin']))
    validated['bmi'] = max(10, min(70, validated['bmi']))
    validated['pedigree'] = max(0, min(3, validated['pedigree']))
    validated['age'] = max(0, min(120, validated['age']))
    
    return validated

def handle_gender_pregnancy_ui(col1):
    """Create UI elements for gender and pregnancy with proper handling"""
    with col1:
        # Get current values from session state
        current_gender = st.session_state.diabetes_inputs.get('gender', 'Male')
        current_pregnancies = st.session_state.diabetes_inputs.get('pregnancies', 0)
        
        # Gender selection
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if current_gender == "Male" else 1, key="gender_select")
        
        # Only show pregnancies field if gender is Female
        if gender == "Female":
            pregnancies = st.number_input("Pregnancies", 0, 20, value=current_pregnancies, key="pregnancies_input")
        else:
            # If male, force pregnancies to 0 and hide the field
            pregnancies = 0
            st.info("Pregnancies field not applicable for males")
        
        # Update session state
        st.session_state.diabetes_inputs['gender'] = gender
        st.session_state.diabetes_inputs['pregnancies'] = pregnancies
        
        return gender, pregnancies

def render_diabetes_inputs():
    """Render diabetes input fields with proper gender-pregnancy handling"""
    # Input fields
    col1, col2, col3 = st.columns(3)
    
    # Handle gender and pregnancy in first column
    gender, pregnancies = handle_gender_pregnancy_ui(col1)
    
    # Other inputs in remaining columns
    with col2:
        glucose = st.number_input("Glucose", 0.0, 300.0, value=float(st.session_state.diabetes_inputs['glucose']))
        blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, value=float(st.session_state.diabetes_inputs['blood_pressure']))
        skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, value=float(st.session_state.diabetes_inputs['skin_thickness']))
    
    with col3:
        insulin = st.number_input("Insulin", 0.0, 500.0, value=float(st.session_state.diabetes_inputs['insulin']))
        bmi = st.number_input("BMI", 0.0, 60.0, value=float(st.session_state.diabetes_inputs['bmi']))
        pedigree = st.number_input("Diabetes Pedigree", 0.0, 2.0, value=float(st.session_state.diabetes_inputs['pedigree']))
        age = st.number_input("Age", 0, 120, value=st.session_state.diabetes_inputs['age'])
    
    # Update session state with all values
    st.session_state.diabetes_inputs.update({
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'pedigree': pedigree,
        'age': age,
        'gender': gender
    })
    
    # Return the validated inputs
    return validate_diabetes_inputs(st.session_state.diabetes_inputs)
