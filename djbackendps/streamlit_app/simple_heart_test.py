import os
import pickle

# Define the path to the heart model
model_path = os.path.join('saved_models', 'heart.pkl')
print(f"Looking for model at: {os.path.abspath(model_path)}")

# Check if the file exists
if os.path.exists(model_path):
    print(f"Model file found!")

    # Load the model
    with open(model_path, 'rb') as f:
        heart_model = pickle.load(f)

    print(f"Model loaded successfully. Type: {type(heart_model)}")

    # Print feature importances if available
    if hasattr(heart_model, 'feature_importances_'):
        importances = heart_model.feature_importances_
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        # Create a DataFrame for better visualization
        import pandas as pd
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\nFeature Importances:")
        print(importance_df)

        print("\nTop 3 important features:")
        top_features = importance_df.head(3)['Feature'].tolist()
        for feature in top_features:
            print(f"- {feature}: {importance_df[importance_df['Feature'] == feature]['Importance'].values[0]:.4f}")

    # Define test cases - these are the ones most likely to get a negative result
    test_cases = [
        # Extreme cases - super healthy young female
        [18, 0, 0, 90, 110, 0, 0, 200, 0, 0.0, 0, 0, 0],

        # Extreme cases - super healthy young male
        [18, 1, 0, 90, 110, 0, 0, 200, 0, 0.0, 0, 0, 0],

        # Try with different combinations of values
        [20, 0, 0, 80, 100, 0, 0, 210, 0, 0.0, 0, 0, 0],
        [20, 1, 0, 80, 100, 0, 0, 210, 0, 0.0, 0, 0, 0],

        # Try with even more extreme values
        [15, 0, 0, 70, 90, 0, 0, 220, 0, 0.0, 0, 0, 0],
        [15, 1, 0, 70, 90, 0, 0, 220, 0, 0.0, 0, 0, 0],

        # Try with extreme values for ca (number of major vessels)
        [25, 0, 0, 110, 160, 0, 0, 160, 0, 0.0, 0, 0, 0],  # Baseline
        [25, 0, 0, 110, 160, 0, 0, 160, 0, 0.0, 0, 3, 0],  # Max ca

        # Try with extreme values for thal (thalassemia)
        [25, 0, 0, 110, 160, 0, 0, 160, 0, 0.0, 0, 0, 0],  # Baseline
        [25, 0, 0, 110, 160, 0, 0, 160, 0, 0.0, 0, 0, 3],  # Max thal

        # Try with extreme values for cp (chest pain)
        [25, 0, 0, 110, 160, 0, 0, 160, 0, 0.0, 0, 0, 0],  # No chest pain
        [25, 0, 3, 110, 160, 0, 0, 160, 0, 0.0, 0, 0, 0],  # Max chest pain
    ]

    # Test each case
    for i, test_case in enumerate(test_cases):
        # Convert to float
        test_case_float = [float(val) for val in test_case]

        # Make prediction
        prediction = heart_model.predict([test_case_float])[0]

        # Get probability
        probability = heart_model.predict_proba([test_case_float])[0][1]

        # Format result
        result = "Has heart disease" if prediction == 1 else "No heart disease"

        # Print result
        print(f"Case {i}: {test_case}")
        print(f"  Prediction: {result}")
        print(f"  Probability: {probability:.4f}")
        print()
else:
    print(f"Model file not found at {model_path}")
    print(f"Current directory: {os.getcwd()}")

    # List files in the current directory
    print("Files in current directory:")
    for file in os.listdir('.'):
        print(f"  - {file}")

    # Check if saved_models directory exists
    if os.path.exists('saved_models'):
        print("\nFiles in saved_models directory:")
        for file in os.listdir('saved_models'):
            print(f"  - {file}")
    else:
        print("\nsaved_models directory not found")
