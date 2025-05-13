import streamlit as st
import requests
import datetime
import json

# API URL
API_URL = "http://localhost:8000/api"

def save_prediction(prediction_type, payload):
    """Save prediction to Django backend with explicit type information"""
    print(f"\n===== SAVING {prediction_type.upper()} PREDICTION =====")
    
    try:
        # Force the correct prediction type
        payload['prediction_type'] = prediction_type
        
        # Add additional type information to help with identification
        if prediction_type == 'diabetes':
            payload['model_type'] = 'diabetes_mellitus'
            payload['model_info'] = 'Diabetes Prediction Model'
            payload['disease_category'] = 'diabetes'
        elif prediction_type == 'heart':
            payload['model_type'] = 'heart_disease'
            payload['model_info'] = 'Heart Disease Prediction Model'
            payload['disease_category'] = 'heart'
        
        # Add timestamp and session ID
        payload['timestamp'] = str(datetime.datetime.now())
        if 'session_id' in st.session_state:
            payload['session_id'] = st.session_state.session_id
        
        # Add a unique identifier
        payload['prediction_id'] = f"{prediction_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Make the API request
        print(f"Sending prediction to {API_URL}/predictions/")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        res = requests.post(f"{API_URL}/predictions/", json=payload, timeout=10)
        
        # Check response
        if res.status_code == 201:
            response_json = res.json()
            print(f"✅ Successfully saved prediction with ID: {response_json.get('id')}")
            return response_json.get('id')
        else:
            print(f"❌ Failed to save prediction. Status code: {res.status_code}")
            try:
                response_json = res.json()
                if 'error' in response_json:
                    print(f"Error message: {response_json['error']}")
            except:
                pass
            return None
    except Exception as e:
        print(f"❌ Error saving prediction: {e}")
        return None

def determine_prediction_type(record):
    """Determine prediction type from record data"""
    # Always set a default prediction type
    default_type = "diabetes"
    
    # Check explicit type fields
    if record.get("prediction_type") in ["diabetes", "heart"]:
        return record.get("prediction_type")
    
    if record.get("model_type"):
        if "diabetes" in record.get("model_type").lower():
            return "diabetes"
        elif "heart" in record.get("model_type").lower():
            return "heart"
    
    if record.get("disease_category") in ["diabetes", "heart"]:
        return record.get("disease_category")
    
    # Check prediction data fields
    pred_data = record.get("prediction_data", {})
    if pred_data:
        # Check for diabetes-specific fields
        diabetes_fields = ["glucose", "pregnancies", "insulin", "bmi", "pedigree"]
        heart_fields = ["trestbps", "chol", "thalach", "cp", "exang"]
        
        diabetes_matches = sum(1 for field in diabetes_fields if field in pred_data)
        heart_matches = sum(1 for field in heart_fields if field in pred_data)
        
        if diabetes_matches > heart_matches:
            return "diabetes"
        elif heart_matches > diabetes_matches:
            return "heart"
    
    # Return default if no determination can be made
    return default_type
