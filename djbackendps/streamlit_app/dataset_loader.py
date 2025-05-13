import os
import pandas as pd
import numpy as np
import streamlit as st

def ensure_datasets_loaded():
    """Ensure all datasets are loaded into session state"""
    if 'diabetes_data' not in st.session_state or st.session_state.diabetes_data is None:
        st.session_state.diabetes_data = load_dataset('diabetes')
    
    if 'heart_data' not in st.session_state or st.session_state.heart_data is None:
        st.session_state.heart_data = load_dataset('heart')
    
    if 'kidney_data' not in st.session_state or st.session_state.kidney_data is None:
        st.session_state.kidney_data = load_dataset('kidney')

def load_dataset(disease_type):
    """Load a dataset from file or create a sample if not found"""
    print(f"\n===== LOADING DATASET =====")
    print(f"Disease type: {disease_type}")
    
    # Get the dataset directory
    working_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(working_dir)
    root_dir = os.path.dirname(project_dir)
    
    # List of possible dataset locations in order of preference
    possible_locations = [
        os.path.join(root_dir, 'dataset'),  # Root/dataset
        os.path.join(project_dir, 'dataset'),  # djbackendps/dataset
        os.path.join(working_dir, 'dataset'),  # djbackendps/streamlit_app/dataset
        os.path.join(root_dir, 'datasets'),  # Root/datasets
        os.path.join(project_dir, 'datasets'),  # djbackendps/datasets
        os.path.join(working_dir, 'datasets')  # djbackendps/streamlit_app/datasets
    ]
    
    # Try each location until we find one that exists
    dataset_dir = None
    for location in possible_locations:
        if os.path.exists(location):
            dataset_dir = location
            print(f"Found dataset directory at: {dataset_dir}")
            break
    
    if dataset_dir is None:
        print(f"Warning: Dataset directory not found in any of the expected locations")
        # Create a fallback location
        dataset_dir = os.path.join(working_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Created fallback dataset directory at: {dataset_dir}")
    
    # Determine file path based on disease type
    if disease_type == 'diabetes':
        file_path = os.path.join(dataset_dir, 'diabetes.csv')
    elif disease_type == 'heart':
        file_path = os.path.join(dataset_dir, 'heart.csv')
    elif disease_type == 'kidney':
        file_path = os.path.join(dataset_dir, 'kidney.csv')
    else:
        print(f"Unknown disease type: {disease_type}")
        return None
    
    # Try to load the dataset
    try:
        if os.path.exists(file_path):
            print(f"Loading dataset from: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {file_path}")
            return df
        else:
            print(f"Dataset file not found: {file_path}")
            # Create a sample dataset if file doesn't exist
            if disease_type == 'diabetes':
                return create_diabetes_dataset()
            elif disease_type == 'heart':
                return create_heart_dataset()
            else:
                return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_diabetes_sample(index=None):
    """Get a sample from the diabetes dataset"""
    # Ensure the dataset is loaded
    if 'diabetes_data' not in st.session_state or st.session_state.diabetes_data is None:
        st.session_state.diabetes_data = load_dataset('diabetes')
    
    # If still None, create a sample dataset
    if st.session_state.diabetes_data is None:
        st.session_state.diabetes_data = create_diabetes_dataset()
    
    # Get a random sample if index is None
    if index is None:
        import random
        index = random.randint(0, len(st.session_state.diabetes_data) - 1)
    
    # Get the sample
    sample = st.session_state.diabetes_data.iloc[index].to_dict()
    return sample

def get_heart_sample(index=None):
    """Get a sample from the heart dataset"""
    # Ensure the dataset is loaded
    if 'heart_data' not in st.session_state or st.session_state.heart_data is None:
        st.session_state.heart_data = load_dataset('heart')
    
    # If still None, create a sample dataset
    if st.session_state.heart_data is None:
        st.session_state.heart_data = create_heart_dataset()
    
    # Get a random sample if index is None
    if index is None:
        import random
        index = random.randint(0, len(st.session_state.heart_data) - 1)
    
    # Get the sample
    sample = st.session_state.heart_data.iloc[index].to_dict()
    return sample

def create_diabetes_dataset():
    """Create a sample diabetes dataset for testing purposes"""
    print("Creating sample diabetes dataset")
    
    # Create sample data
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
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Make sure females have pregnancies and males don't
    for i in range(n_samples):
        # If outcome is 1 (has diabetes), increase glucose and BMI
        if df.loc[i, 'Outcome'] == 1:
            df.loc[i, 'Glucose'] = np.random.randint(140, 200)
            df.loc[i, 'BMI'] = np.random.uniform(30, 40).round(1)
    
    # Set pregnancies to 0 for half the samples (representing males)
    for i in range(n_samples // 2):
        df.loc[i, 'Pregnancies'] = 0
    
    print(f"Created sample diabetes dataset with {n_samples} rows")
    return df

def create_heart_dataset():
    """Create a sample heart disease dataset for testing purposes"""
    print("Creating sample heart disease dataset")
    
    # Create sample data
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
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Make high risk samples more realistic
    for i in range(n_samples):
        if df.loc[i, 'target'] == 1:  # Has heart disease
            df.loc[i, 'age'] = np.random.randint(50, 80)
            df.loc[i, 'chol'] = np.random.randint(220, 300)
            df.loc[i, 'thalach'] = np.random.randint(100, 150)
            df.loc[i, 'exang'] = 1
            df.loc[i, 'oldpeak'] = np.random.uniform(2, 4).round(1)
    
    print(f"Created sample heart disease dataset with {n_samples} rows")
    return df

def load_sample_from_dataset(df, index, disease_type):
    """Load a sample from the dataset and return it as a dictionary"""
    print(f"\n===== LOADING SAMPLE FROM DATASET =====")
    print(f"Disease type: {disease_type}")
    print(f"Index: {index}")
    print(f"DataFrame shape: {df.shape if df is not None else 'None'}")
    
    if df is None or index >= len(df):
        print("Invalid dataframe or index")
        # Return a default sample
        if disease_type == 'diabetes':
            return get_diabetes_sample(0)
        elif disease_type == 'heart':
            return get_heart_sample(0)
        else:
            return {}
    
    # Get the sample
    sample = df.iloc[index].to_dict()
    print(f"Loaded sample: {sample}")
    return sample
