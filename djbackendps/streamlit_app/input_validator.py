import streamlit as st

def validate_diabetes_inputs(inputs):
    """Validate diabetes prediction inputs and return errors if any"""
    errors = []
    
    # Check for required fields
    required_fields = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                      'insulin', 'bmi', 'pedigree', 'age', 'gender']
    
    for field in required_fields:
        if field not in inputs or inputs[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate ranges
    if 'pregnancies' in inputs and (inputs['pregnancies'] < 0 or inputs['pregnancies'] > 20):
        errors.append("Pregnancies should be between 0 and 20")
    
    if 'glucose' in inputs and (inputs['glucose'] < 0 or inputs['glucose'] > 300):
        errors.append("Glucose should be between 0 and 300")
    
    if 'blood_pressure' in inputs and (inputs['blood_pressure'] < 0 or inputs['blood_pressure'] > 200):
        errors.append("Blood pressure should be between 0 and 200")
    
    if 'skin_thickness' in inputs and (inputs['skin_thickness'] < 0 or inputs['skin_thickness'] > 100):
        errors.append("Skin thickness should be between 0 and 100")
    
    if 'insulin' in inputs and (inputs['insulin'] < 0 or inputs['insulin'] > 1000):
        errors.append("Insulin should be between 0 and 1000")
    
    if 'bmi' in inputs and (inputs['bmi'] < 0 or inputs['bmi'] > 70):
        errors.append("BMI should be between 0 and 70")
    
    if 'pedigree' in inputs and (inputs['pedigree'] < 0 or inputs['pedigree'] > 2.5):
        errors.append("Pedigree function should be between 0 and 2.5")
    
    if 'age' in inputs and (inputs['age'] < 1 or inputs['age'] > 120):
        errors.append("Age should be between 1 and 120")
    
    if 'gender' in inputs and inputs['gender'] not in ["Male", "Female"]:
        errors.append("Gender should be either Male or Female")
    
    # Check for gender-pregnancy consistency
    if 'gender' in inputs and 'pregnancies' in inputs:
        if inputs['gender'] == "Male" and inputs['pregnancies'] > 0:
            errors.append("Males cannot have pregnancies > 0")
    
    return errors

def validate_heart_inputs(inputs):
    """Validate heart disease prediction inputs and return errors if any"""
    errors = []
    
    # Check for required fields
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    for field in required_fields:
        if field not in inputs or inputs[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate ranges
    if 'age' in inputs and (inputs['age'] < 1 or inputs['age'] > 120):
        errors.append("Age should be between 1 and 120")
    
    if 'sex' in inputs and inputs['sex'] not in ["Male", "Female"]:
        errors.append("Sex should be either Male or Female")
    
    if 'cp' in inputs and (inputs['cp'] < 0 or inputs['cp'] > 3):
        errors.append("Chest pain type should be between 0 and 3")
    
    if 'trestbps' in inputs and (inputs['trestbps'] < 0 or inputs['trestbps'] > 300):
        errors.append("Resting blood pressure should be between 0 and 300")
    
    if 'chol' in inputs and (inputs['chol'] < 0 or inputs['chol'] > 600):
        errors.append("Cholesterol should be between 0 and 600")
    
    if 'fbs' in inputs and inputs['fbs'] not in ["Yes", "No"]:
        errors.append("Fasting blood sugar should be either Yes or No")
    
    if 'restecg' in inputs and (inputs['restecg'] < 0 or inputs['restecg'] > 2):
        errors.append("Resting ECG should be between 0 and 2")
    
    if 'thalach' in inputs and (inputs['thalach'] < 0 or inputs['thalach'] > 300):
        errors.append("Max heart rate should be between 0 and 300")
    
    if 'exang' in inputs and inputs['exang'] not in ["Yes", "No"]:
        errors.append("Exercise angina should be either Yes or No")
    
    if 'oldpeak' in inputs and (inputs['oldpeak'] < 0 or inputs['oldpeak'] > 10):
        errors.append("ST depression should be between 0 and 10")
    
    if 'slope' in inputs and (inputs['slope'] < 0 or inputs['slope'] > 2):
        errors.append("ST slope should be between 0 and 2")
    
    if 'ca' in inputs and (inputs['ca'] < 0 or inputs['ca'] > 3):
        errors.append("Number of major vessels should be between 0 and 3")
    
    if 'thal' in inputs and (inputs['thal'] < 0 or inputs['thal'] > 3):
        errors.append("Thalassemia should be between 0 and 3")
    
    return errors

def process_diabetes_sample(sample):
    """Process a diabetes sample from the dataset to match the input format"""
    # Convert dataset column names to input field names
    field_mapping = {
        'Pregnancies': 'pregnancies',
        'Glucose': 'glucose',
        'BloodPressure': 'blood_pressure',
        'SkinThickness': 'skin_thickness',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'DiabetesPedigreeFunction': 'pedigree',
        'Age': 'age'
    }
    
    # Create a new dictionary with the mapped fields
    processed = {}
    for old_key, new_key in field_mapping.items():
        if old_key in sample:
            processed[new_key] = sample[old_key]
    
    # Determine gender based on pregnancies
    if 'pregnancies' in processed:
        processed['gender'] = "Female" if processed['pregnancies'] > 0 else "Male"
    else:
        processed['gender'] = "Unknown"
    
    return processed

def process_heart_sample(sample):
    """Process a heart sample from the dataset to match the input format"""
    # Convert dataset column names to input field names
    field_mapping = {
        'age': 'age',
        'sex': 'sex',
        'cp': 'cp',
        'trestbps': 'trestbps',
        'chol': 'chol',
        'fbs': 'fbs',
        'restecg': 'restecg',
        'thalach': 'thalach',
        'exang': 'exang',
        'oldpeak': 'oldpeak',
        'slope': 'slope',
        'ca': 'ca',
        'thal': 'thal'
    }
    
    # Create a new dictionary with the mapped fields
    processed = {}
    for old_key, new_key in field_mapping.items():
        if old_key in sample:
            processed[new_key] = sample[old_key]
    
    # Convert numeric values to string representations
    if 'sex' in processed:
        processed['sex'] = "Male" if processed['sex'] == 1 else "Female"
    
    if 'fbs' in processed:
        processed['fbs'] = "Yes" if processed['fbs'] == 1 else "No"
    
    if 'exang' in processed:
        processed['exang'] = "Yes" if processed['exang'] == 1 else "No"
    
    return processed
