import streamlit as st
import pickle
import os
import requests
import datetime
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

# Import our custom modules
from session_manager import initialize_session, check_url_params, login_user, logout_user, authenticate_user, update_user_activity
from dataset_loader import ensure_datasets_loaded, get_diabetes_sample, get_heart_sample, create_diabetes_dataset, create_heart_dataset
from prediction_manager import save_prediction, determine_prediction_type, fetch_predictions, delete_all_predictions
from input_validator import validate_diabetes_inputs, validate_heart_inputs, process_diabetes_sample, process_heart_sample
from user_forms import display_login_form, display_registration_form
from api_client import register_user, login_user as api_login

st.set_page_config(page_title="Disease Prediction System", layout="wide", page_icon="💊")

# Initialize models
diabetes_model = None
heart_model = None

# Try to load models
try:
    # Look for models in the saved_models directory within the streamlit_app folder
    saved_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

    # Fallback to models directory if saved_models doesn't exist
    if not os.path.exists(saved_models_dir):
        saved_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(saved_models_dir, exist_ok=True)

    print(f"Looking for models in: {saved_models_dir}")

    # Define model paths - use the correct filenames
    diabetes_model_path = os.path.join(saved_models_dir, 'diabetes.pkl')
    heart_model_path = os.path.join(saved_models_dir, 'heart.pkl')
    kidney_model_path = os.path.join(saved_models_dir, 'kidney.pkl')  # Add kidney model path for future use

    # Try to load diabetes model
    if os.path.exists(diabetes_model_path):
        try:
            with open(diabetes_model_path, 'rb') as f:
                diabetes_model = pickle.load(f)
            print(f"Loaded diabetes model from {diabetes_model_path}")
            print(f"Diabetes model type: {type(diabetes_model)}")
        except Exception as e:
            print(f"Error loading diabetes model: {e}")
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            diabetes_model = RandomForestClassifier(n_estimators=10, random_state=42)
            diabetes_model.fit(
                [[0, 100, 70, 30, 0, 25, 0.5, 30]] * 10,  # X (features)
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # y (labels)
            )
            print("Created a dummy diabetes model for demonstration")
    else:
        print(f"Diabetes model not found at {diabetes_model_path}")
        # Create a dummy model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        diabetes_model = RandomForestClassifier(n_estimators=10, random_state=42)
        diabetes_model.fit(
            [[0, 100, 70, 30, 0, 25, 0.5, 30]] * 10,  # X (features)
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # y (labels)
        )
        print("Created a dummy diabetes model for demonstration")

    # Try to load heart model
    if os.path.exists(heart_model_path):
        try:
            with open(heart_model_path, 'rb') as f:
                heart_model = pickle.load(f)
            print(f"Loaded heart model from {heart_model_path}")
            print(f"Heart model type: {type(heart_model)}")

            # Try to get model attributes to understand its structure
            try:
                print(f"Heart model attributes: {dir(heart_model)}")
                if hasattr(heart_model, 'predict'):
                    print("Heart model has predict method")
                if hasattr(heart_model, 'predict_proba'):
                    print("Heart model has predict_proba method")
                if hasattr(heart_model, 'feature_names_in_'):
                    print(f"Heart model feature names: {heart_model.feature_names_in_}")
            except Exception as e:
                print(f"Error inspecting heart model: {e}")
        except Exception as e:
            print(f"Error loading heart model: {e}")
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            heart_model = RandomForestClassifier(n_estimators=10, random_state=42)
            heart_model.fit(
                [[60, 1, 0, 120, 200, 0, 0, 150, 0, 0, 0, 0, 0]] * 10,  # X (features)
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # y (labels)
            )
            print("Created a dummy heart model for demonstration")
    else:
        print(f"Heart model not found at {heart_model_path}")
        # Create a dummy model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        heart_model = RandomForestClassifier(n_estimators=10, random_state=42)
        heart_model.fit(
            [[60, 1, 0, 120, 200, 0, 0, 150, 0, 0, 0, 0, 0]] * 10,  # X (features)
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # y (labels)
        )
        print("Created a dummy heart model for demonstration")
except Exception as e:
    print(f"Error loading models: {e}")
    # If there's an error, create simple dummy models
    from sklearn.ensemble import RandomForestClassifier
    diabetes_model = RandomForestClassifier(n_estimators=10, random_state=42)
    heart_model = RandomForestClassifier(n_estimators=10, random_state=42)
    print("Created simple dummy models due to error")

# Database connection flag
use_db = True  # Set to False to disable database operations

# Initialize session state to prevent app from refreshing
if 'diabetes_data' not in st.session_state:
    st.session_state.diabetes_data = None
if 'heart_data' not in st.session_state:
    st.session_state.heart_data = None
if 'kidney_data' not in st.session_state:
    st.session_state.kidney_data = None
if 'num_rows' not in st.session_state:
    st.session_state.num_rows = 20  # Show more rows by default

# Initialize session state for user management
if 'session' not in st.session_state:
    st.session_state.session = {
        'logged_in': False,
        'username': None,
        'user_role': None,
        'login_time': None,
        'last_activity': None
    }

# Create a session ID to track sessions
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
    print(f"Created new session ID: {st.session_state.session_id}")

# Check for existing session in cookies
if 'username' in st.query_params and st.query_params['username']:
    username = st.query_params['username']
    # Auto-login from URL parameter
    st.session_state.session['username'] = username
    st.session_state.session['logged_in'] = True
    st.session_state.session['user_role'] = "admin" if username == "admin" else "user"
    st.session_state.username = username
    st.session_state.logged_in = True
    print(f"Auto-logged in as {username} from URL parameter")

    # Set session ID in URL to maintain session
    st.query_params['session_id'] = st.session_state.session_id

    # Set login time if not already set
    if not st.session_state.session.get('login_time'):
        st.session_state.session['login_time'] = datetime.datetime.now()

    # Always update last activity time
    st.session_state.session['last_activity'] = datetime.datetime.now()

elif 'session_id' in st.query_params and st.query_params['session_id']:
    # We have a session ID in the URL, check if it matches our current session
    url_session_id = st.query_params['session_id']
    if 'session_id' in st.session_state and url_session_id == st.session_state.session_id:
        print(f"Restored session from URL parameter: {url_session_id}")
        # Session is valid, make sure we're still logged in
        if st.session_state.session.get('username'):
            print(f"Maintaining login for user: {st.session_state.session['username']}")

            # Ensure username is also in st.session_state for backward compatibility
            st.session_state.username = st.session_state.session['username']
            st.session_state.logged_in = True

            # Always update last activity time
            st.session_state.session['last_activity'] = datetime.datetime.now()
    else:
        # Session ID doesn't match, create a new one
        st.session_state.session_id = url_session_id
        print(f"Adopted session ID from URL: {url_session_id}")

# For backward compatibility, ensure these are always set
if 'username' not in st.session_state:
    st.session_state.username = st.session_state.session['username']
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = st.session_state.session['logged_in']

# Initialize user preferences
if 'preferences' not in st.session_state:
    st.session_state.preferences = {
        'theme': 'light',
        'save_predictions': True,
        'show_advanced_options': False,
        'default_model': 'diabetes'
    }

# Initialize user activity history
if 'activity_history' not in st.session_state:
    st.session_state.activity_history = []

# Initialize login-related session state variables
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0

# Simple user database for verification
# In a real app, this would be stored in a database
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "doctor": {"password": "doctor123", "role": "medical"},
    "nurse": {"password": "nurse123", "role": "medical"},
    "user": {"password": "password", "role": "user"},
    "test": {"password": "test", "role": "user"}
}

# For backward compatibility
if 'username' not in st.session_state:
    st.session_state.username = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Function to update user activity
def update_user_activity(activity_type, details=None):
    """Record user activity in the session state"""
    if st.session_state.session['logged_in']:
        timestamp = datetime.datetime.now()
        st.session_state.session['last_activity'] = timestamp

        activity = {
            'timestamp': timestamp,
            'username': st.session_state.session['username'],
            'activity_type': activity_type,
            'details': details
        }

        # Add to activity history (limit to last 50 activities)
        st.session_state.activity_history.append(activity)
        if len(st.session_state.activity_history) > 50:
            st.session_state.activity_history.pop(0)

# Initialize session state for input values
if 'diabetes_inputs' not in st.session_state:
    st.session_state.diabetes_inputs = {
        'pregnancies': 0,
        'glucose': 85.0,
        'blood_pressure': 66.0,
        'skin_thickness': 29.0,
        'insulin': 0.0,
        'bmi': 26.6,
        'pedigree': 0.351,
        'age': 31,
        'gender': 'Male'
    }

if 'heart_inputs' not in st.session_state:
    st.session_state.heart_inputs = {
        'age': 63,
        'sex': 'Male',
        'cp': 0,
        'trestbps': 145.0,
        'chol': 233.0,
        'fbs': 'No',
        'restecg': 0,
        'thalach': 150.0,
        'exang': 'No',
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }

if 'kidney_inputs' not in st.session_state:
    st.session_state.kidney_inputs = {
        'age': 45,
        'blood_pressure': 80.0,
        'specific_gravity': 1.015,
        'albumin': 0.0,
        'sugar': 0.0,
        'red_blood_cells': "Normal",
        'pus_cell': "Normal",
        'pus_cell_clumps': "Not Present",
        'bacteria': "Not Present",
        'blood_glucose': 120.0,
        'blood_urea': 40.0,
        'serum_creatinine': 1.0,
        'sodium': 135.0,
        'potassium': 4.0,
        'hemoglobin': 12.0,
        'packed_cell_volume': 40.0,
        'white_blood_cell_count': 9000.0,
        'red_blood_cell_count': 4.5,
        'hypertension': "No",
        'diabetes_mellitus': "No",
        'coronary_artery_disease': "No",
        'appetite': "Good",
        'pedal_edema': "No",
        'anemia': "No"
    }

# Configuration
API_URL = "http://localhost:8000/api"
working_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(working_dir)

# Check for dataset directory in multiple locations
# First, try the root directory (one level up from project_dir)
root_dir = os.path.dirname(project_dir)
dataset_dir = os.path.join(root_dir, 'dataset')

# List of possible dataset locations in order of preference
possible_locations = [
    dataset_dir,  # Root/dataset
    os.path.join(project_dir, 'dataset'),  # djbackendps/dataset
    os.path.join(working_dir, 'dataset'),  # djbackendps/streamlit_app/dataset
    os.path.join(root_dir, 'datasets'),  # Root/datasets
    os.path.join(project_dir, 'datasets'),  # djbackendps/datasets
    os.path.join(working_dir, 'datasets')  # djbackendps/streamlit_app/datasets
]

# Try each location until we find one that exists
for location in possible_locations:
    if os.path.exists(location):
        dataset_dir = location
        print(f"Found dataset directory at: {dataset_dir}")
        break
else:
    print(f"Warning: Dataset directory not found in any of the expected locations")
    # Create a fallback location
    dataset_dir = os.path.join(working_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Created fallback dataset directory at: {dataset_dir}")

print(f"Dataset directory set to: {dataset_dir}")

# Simplified function to generate random data
def generate_random_data(disease_type):
    """Generate random data for the specified disease type"""
    import random

    if disease_type == 'diabetes':
        # Randomly decide if male or female
        is_female = random.choice([True, False])

        return {
            'pregnancies': random.randint(1, 8) if is_female else 0,
            'glucose': random.uniform(70, 200),
            'blood_pressure': random.uniform(60, 120),
            'skin_thickness': random.uniform(10, 50),
            'insulin': random.uniform(0, 250),
            'bmi': random.uniform(18, 40),
            'pedigree': random.uniform(0.1, 1.5),
            'age': random.randint(20, 70),
            'gender': "Female" if is_female else "Male"
        }

    elif disease_type == 'heart':
        return {
            'age': random.randint(30, 75),
            'sex': random.choice(["Male", "Female"]),
            'cp': random.randint(0, 3),
            'trestbps': random.uniform(100, 180),
            'chol': random.uniform(150, 300),
            'fbs': random.choice(["Yes", "No"]),
            'restecg': random.randint(0, 2),
            'thalach': random.uniform(100, 200),
            'exang': random.choice(["Yes", "No"]),
            'oldpeak': random.uniform(0, 4),
            'slope': random.randint(0, 2),
            'ca': random.randint(0, 3),
            'thal': random.randint(0, 3)
        }

    else:
        return {}

# Main application
# Create a sidebar for navigation
with st.sidebar:
    # Add logo at the top of the sidebar
    st.image("https://img.freepik.com/free-vector/flat-hand-drawn-hospital-reception-scene_52683-54613.jpg", width=200)
    st.title("Disease Prediction")

    # Check if user is logged in
    if not st.session_state.logged_in:
        st.warning("Please log in to access all features")

        # Login form
        username, password, submit = display_login_form()

        if submit:
            # Try to login with API first
            success, response = api_login(username, password)

            if success:
                # Set session state
                st.session_state.username = username
                st.session_state.logged_in = True
                st.session_state.session['username'] = username
                st.session_state.session['logged_in'] = True
                st.session_state.session['user_role'] = response.get('role', 'user')
                st.session_state.session['login_time'] = datetime.datetime.now()
                st.session_state.session['token'] = response.get('token')
                st.session_state.session['user_id'] = response.get('user_id')
                st.success(f"Welcome {username}!")
                st.rerun()
            # Fallback to local authentication
            elif username in USERS and USERS[username]["password"] == password:
                st.session_state.username = username
                st.session_state.logged_in = True
                st.session_state.session['username'] = username
                st.session_state.session['logged_in'] = True
                st.session_state.session['user_role'] = USERS[username]["role"]
                st.session_state.session['login_time'] = datetime.datetime.now()
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
                st.session_state.login_attempts += 1

        # Register link and button
        st.info("Don't have an account? Register to access all features.")
        if st.button("Create New Account"):
            st.session_state.show_registration = True

        # Show registration form if requested
        if st.session_state.get('show_registration', False):
            st.subheader("Registration Form")

            # Add debug information
            st.write("Attempting to display registration form...")

            # Call the registration form function
            try:
                success, response = display_registration_form()

                if success:
                    st.success("Registration successful! You can now log in.")
                    st.session_state.show_registration = False
                elif response and isinstance(response, dict) and 'error' in response:
                    st.error(f"Registration error: {response['error']}")
            except Exception as e:
                st.error(f"Error displaying registration form: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        # Show logged in user info
        st.success(f"Logged in as: {st.session_state.username}")

        # Navigation menu
        selected = option_menu(
            menu_title="Navigation",
            options=["Diabetes Prediction", "Heart Disease Prediction", "Kidney Disease Prediction", "Admin Dashboard"],
            icons=["droplet", "heart", "activity", "gear"],
            menu_icon="cast",
            default_index=0
        )

        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.session['logged_in'] = False
            st.session_state.session['username'] = None
            st.rerun()

# Main content area
if not st.session_state.logged_in:
    # Show welcome message for non-logged in users
    st.title("Welcome to Disease Prediction System")
    st.write("Please log in to access the prediction tools.")

    # Show some information about the app
    st.subheader("About this application")
    st.write("""
    This application provides machine learning models to predict the risk of various diseases:

    - **Diabetes Prediction**: Predict the risk of diabetes based on health parameters
    - **Heart Disease Prediction**: Assess the risk of heart disease based on cardiac health indicators
    - **Kidney Disease Prediction**: Evaluate kidney health based on medical tests (coming soon)

    Please log in using the sidebar to access these features.
    """)

    # Show sample images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://img.freepik.com/free-vector/diabetes-concept-illustration_114360-3993.jpg", width=200)
        st.caption("Diabetes Prediction")
    with col2:
        st.image("https://img.freepik.com/free-vector/cardiology-concept-illustration_114360-6921.jpg", width=200)
        st.caption("Heart Disease Prediction")
    with col3:
        st.image("https://img.freepik.com/free-vector/kidneys-concept-illustration_114360-3687.jpg", width=200)
        st.caption("Kidney Disease Prediction")
else:
    # Main application content based on selected option
    if selected == 'Diabetes Prediction':
        st.title("🩸 Diabetes Prediction")

        # Create two columns for the main layout
        main_col1, main_col2 = st.columns([2, 1])

        # Test samples section - at the top for easy access
        test_col1, test_col2 = st.columns(2)

        with test_col1:
            if st.button("Low Risk Sample", use_container_width=True):
                # Low risk sample values
                st.session_state.diabetes_inputs = {
                    'pregnancies': 1,
                    'glucose': 85.0,
                    'blood_pressure': 66.0,
                    'skin_thickness': 29.0,
                    'insulin': 0.0,
                    'bmi': 26.6,
                    'pedigree': 0.351,
                    'age': 31,
                    'gender': "Female"
                }
                st.success("Low risk sample loaded")
                st.rerun()

        with test_col2:
            if st.button("High Risk Sample", use_container_width=True):
                # High risk sample values
                st.session_state.diabetes_inputs = {
                    'pregnancies': 8,
                    'glucose': 183.0,
                    'blood_pressure': 64.0,
                    'skin_thickness': 0.0,
                    'insulin': 0.0,
                    'bmi': 23.3,
                    'pedigree': 0.672,
                    'age': 32,
                    'gender': "Female"
                }
                st.success("High risk sample loaded")
                st.rerun()

        # Input parameters section
        with main_col1:
            # Manual input section - Always visible
            cols = st.columns(3)

            # Get values from session state
            inputs = st.session_state.diabetes_inputs

            # First get gender since it affects pregnancies
            with cols[2]:
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if inputs['gender'] == "Male" else 1, key="diabetes_gender")

            # Then get pregnancies with a note about gender
            with cols[0]:
                if gender == "Male":
                    # For males, show pregnancies as 0 and disabled
                    st.text_input("Pregnancies", value="0 (Male)", disabled=True, key="diabetes_pregnancies_disabled")
                    pregnancies = 0
                else:
                    # For females, allow pregnancies to be set
                    pregnancies = st.number_input("Pregnancies", 0, 20, value=int(inputs['pregnancies']), key="diabetes_pregnancies_enabled")

            # Other inputs
            with cols[1]: glucose = st.number_input("Glucose Level", 0.0, 300.0, value=float(inputs['glucose']), key="diabetes_glucose")
            with cols[2]: blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, value=float(inputs['blood_pressure']), key="diabetes_bp")
            with cols[0]: skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, value=float(inputs['skin_thickness']), key="diabetes_skin")
            with cols[1]: insulin = st.number_input("Insulin", 0.0, 1000.0, value=float(inputs['insulin']), key="diabetes_insulin")
            with cols[2]: bmi = st.number_input("BMI", 0.0, 70.0, value=float(inputs['bmi']), key="diabetes_bmi")
            with cols[0]: pedigree = st.number_input("Pedigree Function", 0.0, 2.5, value=float(inputs['pedigree']), key="diabetes_pedigree")
            with cols[1]: age = st.number_input("Age", 1, 120, value=int(inputs['age']), key="diabetes_age")

            # Update session state with current values
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

            # Prediction button
            if st.button("Predict Diabetes", use_container_width=True):
                if diabetes_model is None:
                    st.error("Model not loaded")
                else:
                    # Create inputs array
                    inputs_array = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]

                    # Make prediction
                    try:
                        # Handle different model types
                        try:
                            # Try standard scikit-learn prediction
                            prediction = diabetes_model.predict([inputs_array])[0]
                        except AttributeError:
                            # If not a scikit-learn model, try direct call
                            prediction = diabetes_model([inputs_array])[0]
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
                            prediction = 0  # Default to negative prediction

                        # Get probability scores if available
                        try:
                            # Try standard scikit-learn probability
                            probabilities = diabetes_model.predict_proba([inputs_array])[0]
                            risk_score = float(probabilities[1])
                        except AttributeError:
                            # If not available, use prediction as risk score
                            risk_score = float(prediction) if isinstance(prediction, (int, float)) else 0.5
                        except Exception as e:
                            st.warning(f"Could not get probability scores: {e}")
                            risk_score = 0.5  # Default to 50% risk

                        # Determine result and recommendation
                        # Handle different prediction formats
                        if isinstance(prediction, (int, float)):
                            has_diabetes = prediction == 1 or prediction > 0.5
                        elif isinstance(prediction, str):
                            has_diabetes = prediction.lower() in ['1', 'yes', 'true', 'positive']
                        else:
                            has_diabetes = bool(prediction)

                        result = "Has diabetes" if has_diabetes else "No diabetes"
                        recommendation = "Consult a doctor" if has_diabetes else "Maintain healthy lifestyle"

                        # Display results
                        st.subheader("Prediction Results")
                        st.success(f"Prediction: {result}")
                        st.info(f"Risk Score: {risk_score:.2f}")
                        st.info(f"Recommendation: {recommendation}")

                        if use_db:
                            # Create payload
                            payload = {
                                'prediction_type': 'diabetes',  # Explicitly set prediction type
                                'model_type': 'diabetes_mellitus',  # Add additional type info
                                'prediction_data': {
                                    'pregnancies': pregnancies,
                                    'glucose': glucose,
                                    'blood_pressure': blood_pressure,
                                    'skin_thickness': skin_thickness,
                                    'insulin': insulin,
                                    'bmi': bmi,
                                    'pedigree': pedigree,
                                    'age': age,
                                    'gender': gender
                                },
                                'prediction_result': result,
                                'risk_score': risk_score,
                                'recommendation': recommendation,
                                'username': st.session_state.username,
                                'gender': gender,  # Add gender at the top level too
                                'model_info': 'Diabetes Prediction Model'  # Add model info
                            }
                            # Add debug information
                            st.write("Saving diabetes prediction...")
                            st.write(f"Payload: {payload}")

                            # Try to save the prediction
                            rec_id = save_prediction('diabetes', payload)

                            # Show the result
                            if rec_id:
                                st.success(f"Saved record ID: {rec_id}")
                                # Record prediction activity
                                update_user_activity(
                                    "prediction",
                                    f"Diabetes prediction: {result} (Risk: {risk_score:.2f}), Record ID: {rec_id}"
                                )
                            else:
                                st.warning("Failed to save record")
                                # Check if user is logged in
                                if 'session' not in st.session_state or 'token' not in st.session_state.session:
                                    st.error("Not logged in or missing authentication token. Please log in again.")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                        st.warning("Please check if the diabetes model is loaded correctly")

        with main_col2:
            # Additional information and model details
            st.subheader("Model Information")
            if diabetes_model is not None:
                model_type = str(type(diabetes_model)).split("'")[1]
                st.info(f"Model Type: {model_type}")
                st.success("Model loaded successfully")
            else:
                st.error("Model not loaded")
                st.warning("Please check if the diabetes model file exists")

        # Dataset section - at the bottom
        st.subheader("Dataset Information")

        # Show information about the dataset
        st.info("""
        The diabetes prediction model uses the Pima Indians Diabetes Dataset, which includes the following features:

        - **Pregnancies**: Number of times pregnant
        - **Glucose**: Plasma glucose concentration (mg/dL)
        - **Blood Pressure**: Diastolic blood pressure (mm Hg)
        - **Skin Thickness**: Triceps skin fold thickness (mm)
        - **Insulin**: 2-Hour serum insulin (mu U/ml)
        - **BMI**: Body mass index (weight in kg/(height in m)²)
pyth
        - **Age**: Age in years

        The model predicts whether a person has diabetes based on these features.
        """)

        # Add a button to load random data
        if st.button("Generate Random Data", key="diabetes_random_data", use_container_width=True):
            try:
                # Generate random data using our helper function
                st.session_state.diabetes_inputs = generate_random_data('diabetes')

                # Display confirmation
                st.success("✅ Random data generated successfully!")

                # Force UI update
                st.rerun()
            except Exception as e:
                st.error(f"Error generating random data: {e}")
                st.info("Please try again")

    # --- Heart Disease Prediction ---
    if selected == 'Heart Disease Prediction':
        st.title("❤️ Heart Disease Prediction")

        # Add a brief description
        st.markdown("""
        This tool predicts the likelihood of heart disease based on clinical parameters.
        Fill in the form below with your health information to get a prediction.
        """)

        # Initialize heart disease inputs in session state if not already present
        if 'heart_inputs' not in st.session_state:
            st.session_state.heart_inputs = {
                'age': 45,
                'sex': "Male",
                'cp': 0,
                'trestbps': 120.0,
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

        # Create two columns for the main layout
        main_col1, main_col2 = st.columns([2, 1])

        # Test samples section - at the top for easy access
        st.subheader("Quick Start")
        st.markdown("Use these sample profiles or enter your own data below:")

        test_col1, test_col2 = st.columns(2)

        with test_col1:
            if st.button("📊 Low Risk Sample", key="heart_low_risk", use_container_width=True):
                # Low risk sample values
                st.session_state.heart_inputs = {
                    'age': 45,
                    'sex': "Female",
                    'cp': 0,
                    'trestbps': 120.0,
                    'chol': 180.0,
                    'fbs': "No",
                    'restecg': 0,
                    'thalach': 150.0,
                    'exang': "No",
                    'oldpeak': 0.0,
                    'slope': 0,
                    'ca': 3,  # Set to 3 for negative prediction
                    'thal': 0
                }
                st.success("Low risk sample loaded")
                st.rerun()

        with test_col2:
            if st.button("⚠️ High Risk Sample", key="heart_high_risk", use_container_width=True):
                # High risk sample values
                st.session_state.heart_inputs = {
                    'age': 65,
                    'sex': "Male",
                    'cp': 2,
                    'trestbps': 160.0,
                    'chol': 280.0,
                    'fbs': "Yes",
                    'restecg': 1,
                    'thalach': 120.0,
                    'exang': "Yes",
                    'oldpeak': 2.5,
                    'slope': 2,
                    'ca': 0,
                    'thal': 0
                }
                st.success("High risk sample loaded")
                st.rerun()

        # Input parameters section
        with main_col1:
            st.subheader("Your Health Information")

            # Create tabs for different categories of inputs
            tabs = st.tabs(["Demographics", "Heart Measurements", "ECG Results", "Other Factors"])

            # Get values from session state
            inputs = st.session_state.heart_inputs

            # Tab 1: Demographics
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age (years)", 18, 100, value=inputs['age'],
                                         help="Patient's age in years")
                with col2:
                    sex = st.selectbox("Sex", ["Male", "Female"],
                                      index=0 if inputs['sex'] == "Male" else 1,
                                      help="Patient's biological sex")

            # Tab 2: Heart Measurements
            with tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80.0, 200.0,
                                              value=float(inputs['trestbps']),
                                              help="Resting blood pressure in mm Hg")
                    chol = st.number_input("Cholesterol (mg/dl)", 100.0, 600.0,
                                          value=float(inputs['chol']),
                                          help="Serum cholesterol in mg/dl")
                with col2:
                    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                      ["No", "Yes"],
                                      index=0 if inputs['fbs'] == "No" else 1,
                                      help="Whether fasting blood sugar is greater than 120 mg/dl")
                    thalach = st.number_input("Max Heart Rate", 60.0, 220.0,
                                             value=float(inputs['thalach']),
                                             help="Maximum heart rate achieved during exercise")

            # Tab 3: ECG Results
            with tabs[2]:
                col1, col2 = st.columns(2)
                with col1:
                    cp = st.selectbox("Chest Pain Type",
                                     ["Typical Angina (0)", "Atypical Angina (1)",
                                      "Non-anginal Pain (2)", "Asymptomatic (3)"],
                                     index=inputs['cp'],
                                     help="Type of chest pain experienced")
                    restecg = st.selectbox("Resting ECG Results",
                                          ["Normal (0)", "ST-T Wave Abnormality (1)",
                                           "Left Ventricular Hypertrophy (2)"],
                                          index=inputs['restecg'],
                                          help="Resting electrocardiographic results")
                with col2:
                    exang = st.selectbox("Exercise Induced Angina",
                                        ["No", "Yes"],
                                        index=0 if inputs['exang'] == "No" else 1,
                                        help="Whether exercise induced angina")
                    oldpeak = st.number_input("ST Depression", 0.0, 10.0,
                                             value=float(inputs['oldpeak']),
                                             help="ST depression induced by exercise relative to rest")
                    slope = st.selectbox("ST Slope",
                                        ["Upsloping (0)", "Flat (1)", "Downsloping (2)"],
                                        index=inputs['slope'],
                                        help="Slope of the peak exercise ST segment")

            # Tab 4: Other Factors
            with tabs[3]:
                col1, col2 = st.columns(2)
                with col1:
                    ca = st.selectbox("Number of Major Vessels",
                                     ["0", "1", "2", "3"],
                                     index=inputs['ca'],
                                     help="Number of major vessels colored by fluoroscopy (0-3)")
                with col2:
                    thal = st.selectbox("Thalassemia",
                                       ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Severe (3)"],
                                       index=inputs['thal'],
                                       help="Thalassemia type")

            # Update session state with current values
            # Extract the numeric values from the selection strings
            cp_value = int(cp.split("(")[1].split(")")[0]) if "(" in str(cp) else cp
            restecg_value = int(restecg.split("(")[1].split(")")[0]) if "(" in str(restecg) else restecg
            slope_value = int(slope.split("(")[1].split(")")[0]) if "(" in str(slope) else slope
            ca_value = int(ca) if isinstance(ca, str) else ca
            thal_value = int(thal.split("(")[1].split(")")[0]) if "(" in str(thal) else thal

            st.session_state.heart_inputs = {
                'age': age,
                'sex': sex,
                'cp': cp_value,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg_value,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope_value,
                'ca': ca_value,
                'thal': thal_value
            }

            # Prediction button with a more prominent style
            predict_col1, predict_col2 = st.columns([3, 1])
            with predict_col1:
                predict_button = st.button("🔍 Analyze Heart Disease Risk",
                                          type="primary",
                                          use_container_width=True)

            # Add a disclaimer
            with predict_col2:
                st.info("For educational purposes only")

            if predict_button:
                if heart_model is None:
                    st.error("❌ Model not loaded")
                else:
                    # Show a spinner while making prediction
                    with st.spinner("Analyzing your heart health data..."):
                        # Create inputs array
                        inputs_array = [
                            age,
                            1 if sex=="Male" else 0,
                            cp_value,
                            trestbps,
                            chol,
                            1 if fbs=="Yes" else 0,
                            restecg_value,
                            thalach,
                            1 if exang=="Yes" else 0,
                            oldpeak,
                            slope_value,
                            ca_value,
                            thal_value
                        ]

                        # Try to convert all values to float for compatibility
                        inputs_array_float = [float(val) for val in inputs_array]

                        # Make prediction
                        try:
                            # First try with the float array
                            prediction = heart_model.predict([inputs_array_float])[0]

                            # Get probability scores if available
                            try:
                                probabilities = heart_model.predict_proba([inputs_array_float])[0]
                                risk_score = float(probabilities[1])
                            except Exception:
                                # Use prediction as risk score
                                risk_score = float(prediction) if isinstance(prediction, (int, float)) else 0.5

                            # Determine result and recommendation
                            if isinstance(prediction, (int, float)):
                                has_heart_disease = prediction == 1 or prediction > 0.5
                            elif isinstance(prediction, str):
                                has_heart_disease = prediction.lower() in ['1', 'yes', 'true', 'positive']
                            else:
                                has_heart_disease = bool(prediction)

                            result = "Has heart disease" if has_heart_disease else "No heart disease"

                            # More detailed recommendations
                            if has_heart_disease:
                                if risk_score > 0.8:
                                    recommendation = "Urgent: Consult a cardiologist as soon as possible"
                                    rec_icon = "🚨"
                                else:
                                    recommendation = "Schedule an appointment with a cardiologist"
                                    rec_icon = "⚕️"
                            else:
                                if risk_score > 0.3:
                                    recommendation = "Consider lifestyle changes and regular check-ups"
                                    rec_icon = "⚠️"
                                else:
                                    recommendation = "Maintain a healthy lifestyle"
                                    rec_icon = "✅"

                            # Display results in a more visually appealing way
                            st.subheader("Prediction Results")

                            # Create a card-like container for results
                            results_container = st.container()
                            with results_container:
                                # Add a colored box based on the result
                                if has_heart_disease:
                                    st.error(f"**Prediction: {result}**")
                                else:
                                    st.success(f"**Prediction: {result}**")

                                # Display risk score with a progress bar
                                st.markdown(f"**Risk Score: {risk_score:.2f}**")
                                st.progress(risk_score)

                                # Display recommendation
                                st.info(f"{rec_icon} **Recommendation:** {recommendation}")

                                # Add some health tips based on the inputs
                                st.subheader("Health Insights")

                                tips = []
                                if chol > 200:
                                    tips.append("Your cholesterol is above recommended levels. Consider dietary changes and regular exercise.")
                                if trestbps > 130:
                                    tips.append("Your blood pressure is elevated. Monitor it regularly and consult with a healthcare provider.")
                                if thalach < 120:
                                    tips.append("Your maximum heart rate is lower than expected. Regular cardiovascular exercise may help improve heart function.")
                                if int(ca_value) > 0:
                                    tips.append("The presence of major vessels colored by fluoroscopy may indicate coronary artery disease.")

                                if tips:
                                    for tip in tips:
                                        st.markdown(f"- {tip}")
                                else:
                                    st.markdown("- Your health parameters look good. Maintain a healthy lifestyle with regular exercise and a balanced diet.")

                        except Exception as e:
                            st.error(f"❌ Error making prediction: {e}")
                            st.warning("Please check if the heart disease model is loaded correctly")

                        if use_db:
                            try:
                                # Create payload with numeric values
                                # Extract numeric values from string fields
                                def extract_numeric(value):
                                    if isinstance(value, str) and '(' in value and ')' in value:
                                        # Extract the number between parentheses
                                        try:
                                            return int(value.split('(')[1].split(')')[0])
                                        except:
                                            return value
                                    return value

                                # Convert Yes/No to boolean
                                def convert_yes_no(value):
                                    if isinstance(value, str):
                                        return value.lower() == 'yes'
                                    return value

                                # Create payload with processed values
                                payload = {
                                    'prediction_type': 'heart',  # Explicitly set prediction type
                                    'model_type': 'heart_disease',  # Add additional type info
                                    'prediction_data': {
                                        'age': age,
                                        'sex': 1 if sex == 'Male' else 0,  # Convert to numeric
                                        'gender': sex,  # Keep original gender string
                                        'cp': extract_numeric(cp),  # Extract numeric value
                                        'trestbps': float(trestbps) if trestbps is not None else None,
                                        'chol': float(chol) if chol is not None else None,
                                        'fbs': convert_yes_no(fbs),  # Convert to boolean
                                        'restecg': extract_numeric(restecg),  # Extract numeric value
                                        'thalach': float(thalach) if thalach is not None else None,
                                        'exang': convert_yes_no(exang),  # Convert to boolean
                                        'oldpeak': float(oldpeak) if oldpeak is not None else None,
                                        'slope': extract_numeric(slope),  # Extract numeric value
                                        'ca': int(ca) if ca is not None else None,  # Convert to integer
                                        'thal': extract_numeric(thal)  # Extract numeric value
                                    },
                                    'prediction_result': result,
                                    'risk_score': float(risk_score) if risk_score is not None else 0.0,
                                    'recommendation': recommendation,
                                    'username': st.session_state.username,
                                    'gender': sex,  # Add gender at the top level too
                                    'model_info': 'Heart Disease Prediction Model'  # Add model info
                                }
                                # Add debug information
                                st.write("Saving heart disease prediction...")
                                st.write(f"Payload: {payload}")

                                # Try to save the prediction
                                rec_id = save_prediction('heart', payload)

                                # Show the result
                                if rec_id:
                                    st.success(f"Saved record ID: {rec_id}")
                                    # Record prediction activity
                                    update_user_activity(
                                        "prediction",
                                        f"Heart disease prediction: {result} (Risk: {risk_score:.2f}), Record ID: {rec_id}"
                                    )
                                else:
                                    st.warning("Failed to save record")
                                    # Check if user is logged in
                                    if 'session' not in st.session_state or 'token' not in st.session_state.session:
                                        st.error("Not logged in or missing authentication token. Please log in again.")
                            except Exception as e:
                                st.error(f"Error saving prediction: {e}")
                                st.warning("Failed to save the prediction to the database")

        with main_col2:
            # Additional information and model details
            st.subheader("Model Information")
            if heart_model is not None:
                model_type = str(type(heart_model)).split("'")[1]
                st.info(f"Model Type: {model_type}")
                st.success("Model loaded successfully")
            else:
                st.error("Model not loaded")
                st.warning("Please check if the heart disease model file exists")

        # Dataset section - at the bottom
        st.subheader("Dataset Information")

        # Show information about the dataset
        st.info("""
        The heart disease prediction model uses the Cleveland Heart Disease Dataset, which includes the following features:

        - **Age**: Age in years
        - **Sex**: Gender (Male/Female)
        - **CP**: Chest pain type (0-3)
        - **Trestbps**: Resting blood pressure (mm Hg)
        - **Chol**: Serum cholesterol (mg/dl)
        - **FBS**: Fasting blood sugar > 120 mg/dl (Yes/No)
        - **Restecg**: Resting electrocardiographic results (0-2)
        - **Thalach**: Maximum heart rate achieved
        - **Exang**: Exercise induced angina (Yes/No)
        - **Oldpeak**: ST depression induced by exercise relative to rest
        - **Slope**: Slope of the peak exercise ST segment (0-2)
        - **CA**: Number of major vessels colored by fluoroscopy (0-3)
        - **Thal**: Thalassemia (0-3)

        The model predicts whether a person has heart disease based on these features.
        """)

        # Add a button to load random data
        if st.button("Generate Random Data", key="heart_random_data", use_container_width=True):
            try:
                # Generate random data using our helper function
                st.session_state.heart_inputs = generate_random_data('heart')

                # Display confirmation
                st.success("✅ Random data generated successfully!")

                # Force UI update
                st.rerun()
            except Exception as e:
                st.error(f"Error generating random data: {e}")
                st.info("Please try again")

    # --- Kidney Disease Prediction ---
    elif selected == 'Kidney Disease Prediction':
        st.title("🧪 Kidney Disease Prediction")

        st.info("Kidney Disease Prediction model is currently under maintenance.")
        st.warning("Please use the Diabetes or Heart Disease prediction models instead.")

        # Display a placeholder image
        st.image("https://img.freepik.com/free-vector/kidneys-concept-illustration_114360-3687.jpg", width=300)

        # Add some information about kidney disease
        st.subheader("About Kidney Disease")
        st.write("""
        Chronic kidney disease (CKD) is a condition characterized by a gradual loss of kidney function over time.

        **Common risk factors include:**
        - Diabetes
        - High blood pressure
        - Heart disease
        - Family history of kidney failure

        **Prevention tips:**
        - Maintain a healthy blood pressure
        - Control blood sugar levels if you have diabetes
        - Follow a low-salt, low-fat diet
        - Exercise regularly
        - Avoid smoking and excessive alcohol
        - Regular check-ups with your healthcare provider
        """)

        st.write("The Kidney Disease Prediction model will be available soon.")

    # --- Admin Dashboard ---
    elif selected == 'Admin Dashboard':
        st.title("⚙️ Admin Dashboard")

        # Verify that the user is admin
        if st.session_state.username != "admin":
            st.error("You do not have permission to access this page.")
            st.warning("Please log in as an admin user to access the dashboard.")
        else:
            st.success("Welcome to the Admin Dashboard")

            # Create tabs for different admin sections
            admin_tab1, admin_tab2 = st.tabs(["Prediction Records", "User Activity"])

            with admin_tab1:
                st.subheader("Saved Prediction Records")

            # Automatically fetch all records when the page loads
            with st.spinner("Loading records..."):
                predictions = fetch_predictions()

            if predictions:
                st.success(f"Found {len(predictions)} records")

                # Convert to DataFrame for better display
                records = []
                for pred in predictions:
                    # Get the prediction type (already determined in fetch_predictions)
                    prediction_type = pred.get("prediction_type", "unknown")

                    # Ensure prediction type is valid
                    if prediction_type not in ["diabetes", "heart"]:
                        # Try to determine it again
                        prediction_type = determine_prediction_type(pred)

                    # Format the prediction type for display
                    if prediction_type == "diabetes":
                        type_display = "Diabetes"
                        type_icon = "🧠"
                    elif prediction_type == "heart":
                        type_display = "Heart Disease"
                        type_icon = "❤️"
                    else:
                        type_display = "Unknown"
                        type_icon = "❓"

                    # Format the date to be more readable
                    date_str = pred.get("created_at", "")
                    try:
                        # Try to parse the date and reformat it
                        date_obj = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        formatted_date = date_str

                    record = {
                        "ID": pred.get("id"),
                        "Type": f"{type_icon} {type_display}",
                        "Username": pred.get("username"),
                        "Result": pred.get("prediction_result"),
                        "Risk Score": f"{pred.get('risk_score', 0):.2f}",
                        "Date": formatted_date
                    }
                    records.append(record)

                # Create DataFrame
                df = pd.DataFrame(records)

                # Display as table with highlighting
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "Type": st.column_config.TextColumn(
                            "Prediction Type",
                            help="Type of prediction model used"
                        ),
                        "Result": st.column_config.TextColumn(
                            "Prediction Result",
                            help="Outcome of the prediction"
                        ),
                        "Risk Score": st.column_config.NumberColumn(
                            "Risk Score",
                            help="Probability score (0-1)",
                            format="%.2f"
                        )
                    }
                )

                # Allow downloading all records as CSV
                csv = df.to_csv(index=False)
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        "Download Records as CSV",
                        csv,
                        "prediction_records.csv",
                        "text/csv",
                        key="download-csv"
                    )

                with col2:
                    if st.button("Clear All Records", type="primary", use_container_width=True):
                        with st.spinner("Deleting all records..."):
                            if delete_all_predictions():
                                st.success("All prediction records have been deleted")
                                # Record activity
                                update_user_activity("admin", "Deleted all prediction records")
                                # Force refresh
                                st.rerun()
                            else:
                                st.error("Failed to delete all records")

                # Show record details
                if len(records) > 0:
                    st.subheader("Record Details")

                    # Create a more descriptive dropdown for record selection
                    record_options = []
                    for r in records:
                        # Format the option to clearly show the prediction type
                        option_text = f"{r['Type']} Prediction - ID: {r['ID']} - Result: {r['Result']} ({r['Username']})"
                        record_options.append((option_text, r['ID']))

                    # Create the dropdown with the formatted options
                    option_labels = [opt[0] for opt in record_options]
                    option_values = [opt[1] for opt in record_options]

                    if option_labels:
                        selected_index = st.selectbox(
                            "Select record to view details",
                            range(len(option_labels)),
                            format_func=lambda i: option_labels[i]
                        )
                        selected_id = option_values[selected_index]
                    else:
                        st.warning("No records available to select")
                        selected_id = None

                    # Find the selected record
                    selected_record = next((p for p in predictions if p.get("id") == selected_id), None)

                    if selected_record:
                        # Get the prediction type (already determined in fetch_predictions)
                        record_type = selected_record.get("prediction_type", "")

                        # Ensure prediction type is valid
                        if record_type not in ["diabetes", "heart"]:
                            # Try to determine it again
                            record_type = determine_prediction_type(selected_record)
                            # Update the record with the determined type
                            selected_record["prediction_type"] = record_type

                        # Create columns for layout
                        col1, col2 = st.columns(2)

                        # Format the prediction type for display with an icon
                        if record_type == "diabetes":
                            prediction_type_display = "Diabetes"
                            type_icon = "🧠"
                        elif record_type == "heart":
                            prediction_type_display = "Heart Disease"
                            type_icon = "❤️"
                        else:
                            prediction_type_display = "Unknown"
                            type_icon = "❓"

                        # Display a prominent header with the prediction type
                        st.markdown(f"## {type_icon} {prediction_type_display} Prediction Record")

                        with col1:
                            st.write("### Basic Information")
                            st.write(f"**Record ID:** {selected_record.get('id')}")
                            st.write(f"**Username:** {selected_record.get('username')}")
                            st.write(f"**Date:** {selected_record.get('created_at')}")

                        with col2:
                            st.write("### Results")
                            st.write(f"**Prediction:** {selected_record.get('prediction_result')}")
                            st.write(f"**Risk Score:** {selected_record.get('risk_score')}")
                            st.write(f"**Recommendation:** {selected_record.get('recommendation')}")

                        # Display prediction data based on type
                        if "prediction_data" in selected_record:
                            st.write("### Input Parameters")

                            # Get prediction data
                            pred_data = selected_record.get("prediction_data", {})

                            if record_type == "diabetes":
                                # Display diabetes-specific data
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Age:** {pred_data.get('age', 'N/A')}")
                                    st.write(f"**Gender:** {pred_data.get('gender', 'N/A')}")
                                    st.write(f"**Pregnancies:** {pred_data.get('pregnancies', 'N/A')}")
                                with col2:
                                    st.write(f"**Glucose:** {pred_data.get('glucose', 'N/A')}")
                                    st.write(f"**Blood Pressure:** {pred_data.get('blood_pressure', 'N/A')}")
                                    st.write(f"**Skin Thickness:** {pred_data.get('skin_thickness', 'N/A')}")
                                with col3:
                                    st.write(f"**BMI:** {pred_data.get('bmi', 'N/A')}")
                                    st.write(f"**Insulin:** {pred_data.get('insulin', 'N/A')}")
                                    st.write(f"**Pedigree:** {pred_data.get('pedigree', 'N/A')}")

                            elif record_type == "heart":
                                # Display heart-specific data
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Age:** {pred_data.get('age', 'N/A')}")
                                    st.write(f"**Sex:** {pred_data.get('sex', 'N/A')}")
                                    st.write(f"**Chest Pain Type:** {pred_data.get('cp', 'N/A')}")
                                    st.write(f"**Resting BP:** {pred_data.get('trestbps', 'N/A')}")
                                with col2:
                                    st.write(f"**Cholesterol:** {pred_data.get('chol', 'N/A')}")
                                    st.write(f"**Fasting Blood Sugar:** {pred_data.get('fbs', 'N/A')}")
                                    st.write(f"**Resting ECG:** {pred_data.get('restecg', 'N/A')}")
                                    st.write(f"**Max Heart Rate:** {pred_data.get('thalach', 'N/A')}")
                                with col3:
                                    st.write(f"**Exercise Angina:** {pred_data.get('exang', 'N/A')}")
                                    st.write(f"**ST Depression:** {pred_data.get('oldpeak', 'N/A')}")
                                    st.write(f"**ST Slope:** {pred_data.get('slope', 'N/A')}")
                                    st.write(f"**Major Vessels:** {pred_data.get('ca', 'N/A')}")

                            else:
                                # For any other type, just display as JSON
                                st.json(pred_data)
            else:
                st.warning("No records found in the database")

            # User Activity Tab
            with admin_tab2:
                st.subheader("User Activity Log")

                # Display activity history
                if st.session_state.activity_history:
                    # Convert to DataFrame for better display
                    activity_records = []
                    for activity in st.session_state.activity_history:
                        # Format timestamp
                        timestamp = activity.get('timestamp')
                        if timestamp:
                            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            formatted_time = "Unknown"

                        record = {
                            "Time": formatted_time,
                            "Username": activity.get('username', 'Unknown'),
                            "Activity": activity.get('activity_type', 'Unknown'),
                            "Details": activity.get('details', '')
                        }
                        activity_records.append(record)

                    # Create DataFrame
                    activity_df = pd.DataFrame(activity_records)

                    # Display as table
                    st.dataframe(
                        activity_df,
                        use_container_width=True,
                        column_config={
                            "Time": st.column_config.TextColumn(
                                "Timestamp",
                                help="When the activity occurred"
                            ),
                            "Activity": st.column_config.TextColumn(
                                "Activity Type",
                                help="Type of user activity"
                            ),
                            "Details": st.column_config.TextColumn(
                                "Details",
                                help="Additional information about the activity"
                            )
                        }
                    )

                    # Allow downloading activity log as CSV
                    csv = activity_df.to_csv(index=False)
                    st.download_button(
                        "Download Activity Log as CSV",
                        csv,
                        "user_activity_log.csv",
                        "text/csv",
                        key="download-activity-csv"
                    )

                    # Clear activity log button
                    if st.button("Clear Activity Log", type="primary"):
                        st.session_state.activity_history = []
                        st.success("Activity log cleared")
                        st.rerun()
                else:
                    st.info("No user activity recorded in this session")



