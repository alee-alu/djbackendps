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
from dataset_loader import ensure_datasets_loaded, get_diabetes_sample, get_heart_sample, create_diabetes_dataset, create_heart_dataset, load_dataset, load_sample_from_dataset
from prediction_manager import save_prediction, determine_prediction_type, fetch_predictions
from input_validator import validate_diabetes_inputs, validate_heart_inputs, process_diabetes_sample, process_heart_sample

# Set page configuration
st.set_page_config(page_title="Disease Prediction System", layout="wide", page_icon="💊")

# Initialize models
diabetes_model = None
heart_model = None

# Try to load models
try:
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    diabetes_model_path = os.path.join(model_dir, 'diabetes_model.pkl')
    heart_model_path = os.path.join(model_dir, 'heart_model.pkl')

    if os.path.exists(diabetes_model_path):
        with open(diabetes_model_path, 'rb') as f:
            diabetes_model = pickle.load(f)
        print(f"Loaded diabetes model from {diabetes_model_path}")
    else:
        print(f"Diabetes model not found at {diabetes_model_path}")

    if os.path.exists(heart_model_path):
        with open(heart_model_path, 'rb') as f:
            heart_model = pickle.load(f)
        print(f"Loaded heart model from {heart_model_path}")
    else:
        print(f"Heart model not found at {heart_model_path}")
except Exception as e:
    print(f"Error loading models: {e}")

# Flag to determine if we should use the database
use_db = True

# Create sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Diabetes Prediction", "Heart Disease Prediction", "Kidney Disease Prediction", "Admin Dashboard"],
        icons=["droplet", "heart", "activity", "gear"],
        menu_icon="hospital",
        default_index=0
    )

# Initialize the session
initialize_session()
check_url_params()

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

# Ensure datasets are loaded
ensure_datasets_loaded()

# Main application content based on selected option
if selected == 'Diabetes Prediction':
    st.title("🩸 Diabetes Prediction")

    # Create two columns for the main layout
    main_col1, main_col2 = st.columns([2, 1])

    # Main content area
    st.subheader("Input Parameters")

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
            st.experimental_rerun()

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
            st.experimental_rerun()

    # Input parameters section
    with main_col1:
        # Manual input section - Always visible
        cols = st.columns(3)

        # Get values from session state
        inputs = st.session_state.diabetes_inputs

        # First get gender since it affects pregnancies
        with cols[2]:
            gender = st.selectbox("Gender", ["Male", "Female"],
                                 index=0 if inputs['gender'] == "Male" else 1,
                                 key="diabetes_gender")

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
        with cols[1]:
            glucose = st.number_input("Glucose Level", 0.0, 300.0, value=float(inputs['glucose']), key="diabetes_glucose")
        with cols[2]:
            blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, value=float(inputs['blood_pressure']), key="diabetes_bp")
        with cols[0]:
            skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, value=float(inputs['skin_thickness']), key="diabetes_skin")
        with cols[1]:
            insulin = st.number_input("Insulin", 0.0, 1000.0, value=float(inputs['insulin']), key="diabetes_insulin")
        with cols[2]:
            bmi = st.number_input("BMI", 0.0, 70.0, value=float(inputs['bmi']), key="diabetes_bmi")
        with cols[0]:
            pedigree = st.number_input("Pedigree Function", 0.0, 2.5, value=float(inputs['pedigree']), key="diabetes_pedigree")
        with cols[1]:
            age = st.number_input("Age", 1, 120, value=int(inputs['age']), key="diabetes_age")

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

                try:
                    # Make prediction
                    prediction = diabetes_model.predict([inputs_array])[0]

                    # Get probability scores if available
                    try:
                        probabilities = diabetes_model.predict_proba([inputs_array])[0]
                        risk_score = float(probabilities[1])
                    except Exception as e:
                        risk_score = float(prediction)

                    # Determine result and recommendation
                    result = "Has diabetes" if prediction == 1 else "No diabetes"
                    recommendation = "Consult a doctor" if prediction == 1 else "Maintain healthy lifestyle"

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
                            'model_info': 'Diabetes Prediction Model'  # Add model info
                        }
                        rec_id = save_prediction('diabetes', payload)
                        if rec_id:
                            st.success(f"Saved record ID: {rec_id}")
                            # Record prediction activity
                            update_user_activity(
                                "prediction",
                                f"Diabetes prediction: {result} (Risk: {risk_score:.2f}), Record ID: {rec_id}"
                            )
                        else:
                            st.warning("Failed to save record")
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
    st.subheader("Dataset Browser")

    # Load dataset button
    if st.button("Load Dataset", key="diabetes_load_dataset_btn", use_container_width=True):
        # Try to load the dataset
        diabetes_df = load_dataset('diabetes')

        # If dataset loading fails, create a sample dataset
        if diabetes_df is None:
            st.warning("Could not load dataset from file. Creating a sample dataset instead.")
            from dataset_loader import create_diabetes_dataset
            diabetes_df = create_diabetes_dataset()

        # Store in session state
        st.session_state.diabetes_data = diabetes_df
        total_samples = len(diabetes_df)
        st.info(f"Loaded {total_samples} samples from diabetes dataset")

        # Display controls in columns
        browse_col1, browse_col2 = st.columns([3, 1])

        with browse_col1:
            # Display the dataset with a fixed number of rows
            num_rows = min(10, total_samples)
            st.write(f"Showing {num_rows} sample rows from dataset")
            st.dataframe(diabetes_df.head(num_rows))

        with browse_col2:
            # Add a number input to select a specific row
            selected_row = st.number_input("Select row", 0, total_samples - 1, 0, key="diabetes_row_select")

            if st.button("Load Selected Row", key="diabetes_load_selected_row"):
                try:
                    # Get sample from dataset or use default
                    sample = load_sample_from_dataset(diabetes_df, selected_row, "diabetes")

                    # Always show success message
                    st.success(f"✅ Successfully loaded row {selected_row}")

                    # Determine gender based on pregnancies
                    pregnancies = int(sample.get('Pregnancies', 0))
                    gender = "Female" if pregnancies > 0 else "Male"

                    # If male, ensure pregnancies is 0
                    if gender == "Male":
                        pregnancies = 0

                    # Set values directly in session state
                    st.session_state.diabetes_inputs = {
                        'pregnancies': pregnancies,
                        'glucose': float(sample['Glucose']),
                        'blood_pressure': float(sample['BloodPressure']),
                        'skin_thickness': float(sample['SkinThickness']),
                        'insulin': float(sample['Insulin']),
                        'bmi': float(sample['BMI']),
                        'pedigree': float(sample['DiabetesPedigreeFunction']),
                        'age': int(sample['Age']),
                        'gender': gender
                    }

                    # Display confirmation
                    st.success(f"✅ Sample data loaded successfully!")

                    # Force UI update
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    st.info("Please try again")

        if st.button("Load Random Row", key="diabetes_load_random_row"):
            # Direct approach - hardcode random sample values
            try:
                # Generate random values
                import random

                # Randomly decide if male or female
                is_female = random.choice([True, False])

                # Create a random sample
                sample = {
                    'Pregnancies': random.randint(1, 8) if is_female else 0,
                    'Glucose': random.uniform(70, 200),
                    'BloodPressure': random.uniform(60, 120),
                    'SkinThickness': random.uniform(10, 50),
                    'Insulin': random.uniform(0, 250),
                    'BMI': random.uniform(18, 40),
                    'DiabetesPedigreeFunction': random.uniform(0.1, 1.5),
                    'Age': random.randint(20, 70)
                }

                # Ensure consistency between gender and pregnancies
                gender = "Female" if is_female else "Male"
                pregnancies = int(sample['Pregnancies'])

                # If male, ensure pregnancies is 0
                if gender == "Male" and pregnancies > 0:
                    pregnancies = 0

                # Set values directly in session state
                st.session_state.diabetes_inputs = {
                    'pregnancies': pregnancies,
                    'glucose': float(sample['Glucose']),
                    'blood_pressure': float(sample['BloodPressure']),
                    'skin_thickness': float(sample['SkinThickness']),
                    'insulin': float(sample['Insulin']),
                    'bmi': float(sample['BMI']),
                    'pedigree': float(sample['DiabetesPedigreeFunction']),
                    'age': int(sample['Age']),
                    'gender': gender
                }

                # Display confirmation
                st.success(f"✅ Random diabetes data loaded successfully!")

                # Record activity
                update_user_activity("data_load", "Loaded random diabetes data")

                # Force UI update
                st.rerun()
            except Exception as e:
                st.error(f"Error loading random data: {e}")
                st.info("Please try again")
    else:
        st.info("Click 'Load Dataset' to view and interact with the diabetes dataset.")

elif selected == 'Heart Disease Prediction':
    st.title("❤️ Heart Disease Prediction")

    # Create two columns for the main layout
    main_col1, main_col2 = st.columns([2, 1])

    # Main content area
    st.subheader("Input Parameters")

    # Test samples section - at the top for easy access
    test_col1, test_col2 = st.columns(2)

    with test_col1:
        if st.button("Low Risk Sample", key="heart_low_risk", use_container_width=True):
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
                'ca': 0,
                'thal': 0
            }
            st.success("Low risk sample loaded")
            st.experimental_rerun()

    with test_col2:
        if st.button("High Risk Sample", key="heart_high_risk", use_container_width=True):
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
                'ca': 2,
                'thal': 3
            }
            st.success("High risk sample loaded")
            st.experimental_rerun()

    # Input parameters section
    with main_col1:
        # Manual input section - Always visible
        cols = st.columns(3)

        # Get values from session state
        inputs = st.session_state.heart_inputs

        # Input fields
        with cols[0]: age = st.number_input("Age", 1, 120, value=inputs['age'])
        with cols[1]: sex = st.selectbox("Sex", ["Male", "Female"], index=0 if inputs['sex'] == "Male" else 1)
        with cols[2]: cp = st.selectbox("Chest Pain Type", [0,1,2,3], index=inputs['cp'])
        with cols[0]: trestbps = st.number_input("Resting BP", 0.0, 300.0, value=float(inputs['trestbps']))
        with cols[1]: chol = st.number_input("Cholesterol", 0.0, 600.0, value=float(inputs['chol']))
        with cols[2]: fbs = st.selectbox("Fasting Blood Sugar >120", ["No","Yes"], index=0 if inputs['fbs'] == "No" else 1)
        with cols[0]: restecg = st.number_input("Resting ECG", 0, 2, value=inputs['restecg'])
        with cols[1]: thalach = st.number_input("Max Heart Rate", 0.0, 300.0, value=float(inputs['thalach']))
        with cols[2]: exang = st.selectbox("Exercise Angina", ["No","Yes"], index=0 if inputs['exang'] == "No" else 1)
        with cols[0]: oldpeak = st.number_input("ST Depression", 0.0, 10.0, value=float(inputs['oldpeak']))
        with cols[1]: slope = st.selectbox("ST Slope", [0,1,2], index=inputs['slope'])
        with cols[2]: ca = st.selectbox("Major Vessels", [0,1,2,3], index=inputs['ca'])
        with cols[0]: thal = st.selectbox("Thalassemia", [0,1,2,3], index=inputs['thal'])

        # Update session state with current values
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

        # Prediction button
        if st.button("Predict Heart Disease", use_container_width=True):
            if heart_model is None:
                st.error("Model not loaded")
            else:
                # Create inputs array
                inputs_array = [
                    age,
                    1 if sex=="Male" else 0,
                    cp,
                    trestbps,
                    chol,
                    1 if fbs=="Yes" else 0,
                    restecg,
                    thalach,
                    1 if exang=="Yes" else 0,
                    oldpeak,
                    slope,
                    ca,
                    thal
                ]

                try:
                    # Make prediction
                    prediction = heart_model.predict([inputs_array])[0]

                    # Get probability scores if available
                    try:
                        probabilities = heart_model.predict_proba([inputs_array])[0]
                        risk_score = float(probabilities[1])
                    except Exception as e:
                        risk_score = float(prediction)

                    # Determine result and recommendation
                    result = "Has heart disease" if prediction == 1 else "No heart disease"
                    recommendation = "Consult a cardiologist" if prediction == 1 else "Healthy heart"

                    # Display results
                    st.subheader("Prediction Results")
                    st.success(f"Prediction: {result}")
                    st.info(f"Risk Score: {risk_score:.2f}")
                    st.info(f"Recommendation: {recommendation}")

                    if use_db:
                        # Create payload
                        payload = {
                            'prediction_type': 'heart',  # Explicitly set prediction type
                            'model_type': 'heart_disease',  # Add additional type info
                            'prediction_data': {
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
                            },
                            'prediction_result': result,
                            'risk_score': risk_score,
                            'recommendation': recommendation,
                            'username': st.session_state.username,
                            'model_info': 'Heart Disease Prediction Model'  # Add model info
                        }
                        rec_id = save_prediction('heart', payload)
                        if rec_id:
                            st.success(f"Saved record ID: {rec_id}")
                            # Record prediction activity
                            update_user_activity(
                                "prediction",
                                f"Heart disease prediction: {result} (Risk: {risk_score:.2f}), Record ID: {rec_id}"
                            )
                        else:
                            st.warning("Failed to save record")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.warning("Please check if the heart disease model is loaded correctly")

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
                    st.dataframe(df, use_container_width=True)
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
                        "User": activity.get('username', 'Unknown'),
                        "Action": activity.get('action_type', 'Unknown'),
                        "Details": activity.get('description', '')
                    }
                    activity_records.append(record)

                # Create DataFrame
                activity_df = pd.DataFrame(activity_records)

                # Display as table
                st.dataframe(activity_df, use_container_width=True)

                # Clear activity log button
                if st.button("Clear Activity Log", type="primary"):
                    st.session_state.activity_history = []
                    st.success("Activity log cleared")
                    st.experimental_rerun()
            else:
                st.info("No user activity recorded in this session")



# Function to determine prediction type from record data
def determine_prediction_type(record):
    """
    Determine the prediction type based on the record data.
    Returns: 'diabetes', 'heart', or 'unknown'
    """
    # Print record for debugging
    print(f"\n===== DETERMINING PREDICTION TYPE =====")
    print(f"Record keys: {record.keys() if record else 'None'}")

    # First check if disease_type is set (from Django model)
    disease_type = record.get("disease_type", "")
    if disease_type:
        print(f"Found disease_type: {disease_type}")
        return disease_type.lower()

    # Then check if model_type is set
    model_type = record.get("model_type", "")
    if model_type:
        print(f"Found model_type: {model_type}")
        if "diabetes" in model_type.lower():
            return "diabetes"
        elif "heart" in model_type.lower() or "cardiac" in model_type.lower():
            return "heart"

    # If neither disease_type nor model_type is set, return 'unknown'
    print("No valid prediction type found in record")
    return "unknown"

# Function to save prediction to Django backend
def save_prediction(prediction_type, payload):
    print(f"\n===== SAVING {prediction_type.upper()} PREDICTION TO DATABASE =====")
    try:
        # Show minimal information
        print(f"Sending prediction to {API_URL}/predictions/...")

        # IMPORTANT: Make sure prediction_type and disease_type are explicitly set in the payload
        payload['prediction_type'] = prediction_type  # Force the correct type
        payload['disease_type'] = prediction_type  # Add disease_type to match Django model

        # Add a unique identifier to ensure uniqueness
        payload['prediction_id'] = f"{prediction_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Send the prediction to the Django backend
        response = requests.post(f"{API_URL}/predictions/", json=payload)

        # Check the response
        if response.status_code == 201:
            print("Prediction saved successfully")
            return response.json().get('id')  # Return the ID of the saved prediction
        else:
            print(f"Failed to save prediction. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return None

# Function to load a sample from dataset into input fields
def load_sample_from_dataset(df, index, disease_type):
    """Load a sample from the dataset and return it as a dictionary"""
    print(f"\n===== LOADING SAMPLE FROM DATASET =====")
    print(f"Disease type: {disease_type}")
    print(f"Index: {index}")
    print(f"DataFrame shape: {df.shape if df is not None else 'None'}")
    print(f"DataFrame columns: {df.columns.tolist() if df is not None else 'None'}")

    # Try to get the actual sample from the dataset
    try:
        if df is not None and index >= 0 and index < len(df):
            # Get the row from the dataset
            row = df.iloc[index]
            sample = row.to_dict()
            print(f"Successfully loaded {disease_type} sample from row {index}")
            print(f"Sample data: {sample}")
            return sample
        else:
            print(f"Index {index} out of bounds or DataFrame is None. Using default sample.")
            # Use default samples as fallback
            if disease_type == 'diabetes':
                return {
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
                return {
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
                print(f"Error: Unknown disease type {disease_type}")
                return {}
    except Exception as e:
        print(f"Error loading sample: {e}")
        print(f"DataFrame info: {df.info() if df is not None else 'None'}")
        # Return default sample instead of failing
        if disease_type == 'diabetes':
            return {
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
            return {
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
            return {}
