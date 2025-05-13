import requests
import json
import streamlit as st
import datetime

# API configuration
API_URL = "http://localhost:8000/api"

def save_prediction(prediction_type, payload):
    """Save a prediction to the database via API"""
    print(f"\n===== SAVING PREDICTION =====")
    print(f"Prediction type: {prediction_type}")
    print(f"Payload: {payload}")

    # Determine the endpoint based on prediction type
    endpoint = f"{API_URL}/predictions/"  # Always use the main predictions endpoint

    try:
        # Add timestamp if not present
        if 'timestamp' not in payload:
            payload['timestamp'] = datetime.datetime.now().isoformat()

        # IMPORTANT: Make sure prediction_type and disease_type are explicitly set in the payload
        payload['prediction_type'] = prediction_type  # Force the correct type
        payload['disease_type'] = prediction_type  # Add disease_type to match Django model

        # Make sure gender is included in the payload at the top level
        if 'prediction_data' in payload and 'gender' in payload['prediction_data']:
            # Extract gender from prediction_data and add it to the top level
            payload['gender'] = payload['prediction_data']['gender']
        elif prediction_type == 'heart' and 'prediction_data' in payload and 'sex' in payload['prediction_data']:
            # For heart disease, use 'sex' field as gender
            payload['gender'] = payload['prediction_data']['sex']

        # Add a unique identifier to ensure uniqueness
        payload['prediction_id'] = f"{prediction_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Get authentication token from session state if available
        headers = {'Content-Type': 'application/json'}
        if 'session' in st.session_state and 'token' in st.session_state.session:
            token = st.session_state.session['token']
            if token:
                headers['Authorization'] = f'Token {token}'
                print(f"Using authentication token: {token}")

        # Print detailed debug information
        print(f"Endpoint: {endpoint}")
        print(f"Headers: {headers}")
        print(f"Payload: {json.dumps(payload, indent=2, default=str)}")

        # Make the API request
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers
            )

            # Print response details
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response content: {response.text}")

            # Check if the request was successful
            if response.status_code == 201:
                print(f"Successfully saved prediction: {response.json()}")
                return response.json().get('id')
            else:
                print(f"Failed to save prediction: {response.status_code} - {response.text}")
                print(f"Response: {response.text}")
                return None
        except Exception as req_error:
            print(f"Request error: {req_error}")
            return None
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return None

def determine_prediction_type(prediction_data):
    """Determine the type of prediction based on the data"""
    # First check if disease_type is set (from Django model)
    disease_type = prediction_data.get("disease_type", "")
    if disease_type:
        print(f"Found disease_type: {disease_type}")
        return disease_type.lower()

    # Then check if prediction_type is set
    prediction_type = prediction_data.get("prediction_type", "")
    if prediction_type:
        print(f"Found prediction_type: {prediction_type}")
        return prediction_type.lower()

    # Then check if model_type is set
    model_type = prediction_data.get("model_type", "")
    if model_type:
        print(f"Found model_type: {model_type}")
        if "diabetes" in model_type.lower():
            return "diabetes"
        elif "heart" in model_type.lower() or "cardiac" in model_type.lower():
            return "heart"

    # Check for diabetes-specific fields
    if any(key in prediction_data for key in ['glucose', 'insulin', 'bmi', 'pedigree', 'pregnancies']):
        return 'diabetes'

    # Check for heart-specific fields
    if any(key in prediction_data for key in ['cp', 'thalach', 'exang', 'oldpeak', 'ca', 'thal', 'chest_pain_type']):
        return 'heart'

    # Check for kidney-specific fields
    if any(key in prediction_data for key in ['blood_urea', 'creatinine', 'albumin', 'sodium', 'potassium']):
        return 'kidney'

    # Check prediction_data if it exists
    if 'prediction_data' in prediction_data:
        nested_data = prediction_data['prediction_data']
        # Recursively check the nested data
        nested_type = determine_prediction_type(nested_data)
        if nested_type != 'unknown':
            return nested_type

    # Default to unknown
    return 'unknown'

def fetch_predictions(username=None):
    """Fetch predictions from the database via API"""
    print(f"\n===== FETCHING PREDICTIONS =====")

    # Determine the endpoint
    endpoint = f"{API_URL}/predictions/"

    # Add username filter if provided
    if username:
        endpoint += f"?username={username}"

    try:
        # Get authentication token from session state if available
        headers = {'Content-Type': 'application/json'}
        if 'session' in st.session_state and 'token' in st.session_state.session:
            token = st.session_state.session['token']
            if token:
                headers['Authorization'] = f'Token {token}'
                print(f"Using authentication token: {token}")

        # Make the API request
        response = requests.get(endpoint, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            predictions = response.json()
            print(f"Successfully fetched {len(predictions)} predictions")

            # Process each prediction to ensure it has a prediction_type
            for pred in predictions:
                if 'prediction_type' not in pred or not pred['prediction_type']:
                    pred['prediction_type'] = determine_prediction_type(pred)

            return predictions
        else:
            print(f"Failed to fetch predictions: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        # Return empty list on error
        return []

def fetch_prediction_by_id(prediction_id):
    """Fetch a specific prediction by ID"""
    print(f"\n===== FETCHING PREDICTION BY ID =====")
    print(f"Prediction ID: {prediction_id}")

    # Determine the endpoint
    endpoint = f"{API_URL}/predictions/{prediction_id}/"

    try:
        # Get authentication token from session state if available
        headers = {'Content-Type': 'application/json'}
        if 'session' in st.session_state and 'token' in st.session_state.session:
            token = st.session_state.session['token']
            if token:
                headers['Authorization'] = f'Token {token}'
                print(f"Using authentication token: {token}")

        # Make the API request
        response = requests.get(endpoint, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            prediction = response.json()
            print(f"Successfully fetched prediction: {prediction}")

            # Ensure it has a prediction_type
            if 'prediction_type' not in prediction or not prediction['prediction_type']:
                prediction['prediction_type'] = determine_prediction_type(prediction)

            return prediction
        else:
            print(f"Failed to fetch prediction: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error fetching prediction: {e}")
        return None

def delete_prediction(prediction_id):
    """Delete a prediction from the database"""
    print(f"\n===== DELETING PREDICTION =====")
    print(f"Prediction ID: {prediction_id}")

    # Determine the endpoint
    endpoint = f"{API_URL}/predictions/{prediction_id}/"

    try:
        # Get authentication token from session state if available
        headers = {'Content-Type': 'application/json'}
        if 'session' in st.session_state and 'token' in st.session_state.session:
            token = st.session_state.session['token']
            if token:
                headers['Authorization'] = f'Token {token}'
                print(f"Using authentication token: {token}")

        # Make the API request
        response = requests.delete(endpoint, headers=headers)

        # Check if the request was successful
        if response.status_code == 204:
            print(f"Successfully deleted prediction")
            return True
        else:
            print(f"Failed to delete prediction: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error deleting prediction: {e}")
        return False

def delete_all_predictions():
    """Delete all predictions from the database"""
    print(f"\n===== DELETING ALL PREDICTIONS =====")

    # First fetch all predictions
    predictions = fetch_predictions()

    if not predictions:
        print("No predictions to delete")
        return True

    # Delete each prediction
    success = True
    for pred in predictions:
        pred_id = pred.get('id')
        if pred_id:
            if not delete_prediction(pred_id):
                success = False

    return success
